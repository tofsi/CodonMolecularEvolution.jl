"""
    apply_smoothing(reshaping_scheme, ambient_parameters, kernel_parameters;
        dims, smoothing_method=TruncatedGaussianConvolution())

Apply Gaussian smoothing to ambient category logits along the selected grid
dimensions. The effective bandwidth is `abs(kernel_parameters[1])`;
`smoothing_method` selects the smoothing geometry.
"""
function apply_smoothing(
    reshaping_scheme::ProbabilityVectorReshapingScheme,
    ambient_parameters::AbstractVector{<:Real},
    kernel_parameters::AbstractVector{<:Real};
    dims=ntuple(identity, length(reshaping_scheme.grid_sizes)),
    smoothing_method::GaussianSmoothingMethod=TruncatedGaussianConvolution(),
)
    return apply_smoothing(
        smoothing_method,
        reshaping_scheme,
        ambient_parameters,
        kernel_parameters;
        dims=dims,
    )
end

function apply_smoothing(
    ::TruncatedGaussianConvolution,
    reshaping_scheme::ProbabilityVectorReshapingScheme,
    ambient_parameters::AbstractVector{<:Real},
    kernel_parameters::AbstractVector{<:Real};
    dims=ntuple(identity, length(reshaping_scheme.grid_sizes)),
)
    ambient_parameter_array = reshape_probability_vector(reshaping_scheme, ambient_parameters)
    kernel = gaussian_kernel(5, kernel_parameters[1]^2)
    smoothed_parameter_array =
        apply_separable_convolution(ambient_parameter_array, kernel; dims=dims)
    return unreshape_probability_vector(reshaping_scheme, smoothed_parameter_array)
end

"""
    gaussian_covariance_matrix(n, variance; jitter=1e-6)

Return the full `n × n` Gaussian correlation matrix for a grid dimension. Its
support spans every possible grid offset (equivalent to a convolution width of
`2n - 1`). A diagonal nugget bounds the condition number, and division by
`1 + jitter` preserves unit marginal variance.
"""
function gaussian_covariance_matrix(
    n::Int,
    variance::Real;
    jitter::Real=1e-6,
)
    n > 0 || throw(ArgumentError("Gaussian covariance dimension must be positive; got $n"))
    jitter > zero(jitter) ||
        throw(ArgumentError("Gaussian covariance jitter must be positive; got $jitter"))

    stabilized_variance = variance + 1e-6
    scale = inv(one(stabilized_variance) + jitter)
    return [
        scale * (
            exp(-((i - j)^2) / (2 * stabilized_variance)) +
            ifelse(i == j, jitter, zero(jitter))
        )
        for i in 1:n, j in 1:n
    ]
end

"""
    gaussian_covariance_factor(n, variance; jitter=1e-6)

Return the lower Cholesky factor of [`gaussian_covariance_matrix`](@ref). The
factor is used to map independent standard-normal logits to logits with the
requested Gaussian covariance.
"""
function gaussian_covariance_factor(
    n::Int,
    variance::Real;
    jitter::Real=1e-6,
)
    covariance = gaussian_covariance_matrix(n, variance; jitter=jitter)
    return Matrix(cholesky(Symmetric(covariance)).L)
end

function apply_smoothing(
    smoothing_method::GaussianCovarianceSmoothing,
    reshaping_scheme::ProbabilityVectorReshapingScheme,
    ambient_parameters::AbstractVector{<:Real},
    kernel_parameters::AbstractVector{<:Real};
    dims=ntuple(identity, length(reshaping_scheme.grid_sizes)),
)
    smoothed_parameter_array =
        reshape_probability_vector(reshaping_scheme, ambient_parameters)
    variance = kernel_parameters[1]^2
    smoothed_parameter_array = apply_gaussian_covariance_factors(
        smoothed_parameter_array,
        variance,
        dims,
        smoothing_method.jitter,
    )

    return unreshape_probability_vector(reshaping_scheme, smoothed_parameter_array)
end

function apply_smoothing(
    smoothing_method::GaussianCovarianceSmoothing,
    reshaping_scheme::PermutedDimsReshapingScheme{N},
    ambient_parameters::AbstractVector{<:Real},
    kernel_parameters::AbstractVector{<:Real};
    dims=ntuple(identity, length(reshaping_scheme.grid_sizes)),
) where {N}
    temporary_sizes = ntuple(
        dim -> reshaping_scheme.grid_sizes[reshaping_scheme.invperm[dim]],
        N,
    )
    temporary_array = reshape(ambient_parameters, temporary_sizes)
    temporary_dims = map(dim -> reshaping_scheme.perm[dim], dims)
    smoothed_temporary_array = apply_gaussian_covariance_factors(
        temporary_array,
        kernel_parameters[1]^2,
        temporary_dims,
        smoothing_method.jitter,
    )
    return vec(smoothed_temporary_array)
end

function apply_gaussian_covariance_factors(
    parameter_array::AbstractArray{<:Real},
    variance::Real,
    dims,
    jitter::Real,
)
    isempty(dims) && return parameter_array

    maximum_dimension = maximum(map(dim -> size(parameter_array, dim), dims))
    full_factor = gaussian_covariance_factor(
        maximum_dimension,
        variance;
        jitter=jitter,
    )

    smoothed_parameter_array = parameter_array
    for dim in dims
        dimension = size(smoothed_parameter_array, dim)
        factor = @view full_factor[1:dimension, 1:dimension]
        smoothed_parameter_array =
            multiply_along_dim(smoothed_parameter_array, factor, dim)
    end
    return smoothed_parameter_array
end

"""
    gaussian_kernel(window_size, variance)

Return an L2-normalized, one-dimensional Gaussian convolution kernel.

The variance is floored slightly for numerical stability. When this kernel is
used with [`same_conv`](@ref), zero padding intentionally reduces marginal
variance near grid boundaries. smoothFLAVOR instead uses
[`GaussianCovarianceSmoothing`](@ref), which preserves unit marginal variance.
"""
function gaussian_kernel(window_size::Int64, variance::Real)
    radius = (window_size - 1) ÷ 2
    x = -radius:radius
    kernel = exp.(-(x .^ 2) ./ (2 * (variance + 1e-6)))
    kernel /= sqrt(sum(abs2, kernel))
    return kernel
end

"""
    apply_separable_convolution(x, kernel; dims)

Apply a separable convolution along the selected dimensions of `x`.
"""
function apply_separable_convolution(x::AbstractArray{<:Real}, kernel::AbstractVector{<:Real}; dims=ntuple(identity, ndims(x)))
    y = copy(x)
    for d in dims
        y = convolve_along_dim(y, kernel, d)
    end
    return y
end

"""
    convolve_along_dim(x, kernel, dim)

Convolve `x` with `kernel` along dimension `dim`, preserving the size of `x`.
"""
function convolve_along_dim(x::AbstractArray{<:Real}, kernel::AbstractVector{<:Real}, dim::Int)
    perm = (dim, filter(d -> d != dim, 1:ndims(x))...)
    x_perm = permutedims(x, perm)
    sz = size(x_perm)
    reshaped = reshape(x_perm, sz[1], :)
    result = map(col -> same_conv(col, kernel), eachcol(reshaped))
    result_mat = hcat(result...)
    result_array = reshape(result_mat, sz...)
    return permutedims(result_array, invperm(perm))
end

"""
    multiply_along_dim(x, matrix, dim)

Multiply every vector along dimension `dim` by `matrix`.
"""
function multiply_along_dim(
    x::AbstractArray{<:Real},
    matrix::AbstractMatrix{<:Real},
    dim::Int,
)
    size(matrix, 1) == size(x, dim) == size(matrix, 2) ||
        throw(DimensionMismatch(
            "matrix size $(size(matrix)) does not match array dimension $dim of length $(size(x, dim))",
        ))

    if dim == 1
        sz = size(x)
        result_mat = matrix * reshape(x, sz[1], :)
        return reshape(result_mat, sz)
    end

    perm = (dim, filter(d -> d != dim, 1:ndims(x))...)
    x_perm = permutedims(x, perm)
    sz = size(x_perm)
    result_mat = matrix * reshape(x_perm, sz[1], :)
    result_array = reshape(result_mat, sz...)
    return permutedims(result_array, invperm(perm))
end

"""
    conv_pure(x, kernel)

Compute the valid one-dimensional convolution of `x` with `kernel`.
"""
function conv_pure(x::AbstractVector{<:Real}, kernel::AbstractVector{<:Real})
    nx = length(x)
    nk = length(kernel)
    ny = nx - nk + 1
    return [sum(x[i:i+nk-1] .* kernel) for i in 1:ny]
end

"""
    same_conv(col, kernel)

Convolve a vector with zero padding and return a vector of the same length.
"""
function same_conv(col, kernel)
    klen = length(kernel)
    left = fld(klen - 1, 2)
    right = cld(klen - 1, 2)
    padded = vcat(zeros(left), col, zeros(right))
    conv_result = conv_pure(padded, kernel)
    return conv_result
end
