function apply_smoothing(grid_sizes::Tuple, ambient_parameters::AbstractVector{<:Real}, kernel_parameters::AbstractVector{<:Real})
    ambient_parameter_array = reshape_probability_vector(grid_sizes, ambient_parameters)
    kernel = gaussian_kernel(5, kernel_parameters[1]^2) #approximate_gaussian_kernel(kernel_parameters[1]^2, 4)
    smoothed_parameter_array = apply_separable_convolution(ambient_parameter_array, kernel)
    return unreshape_probability_vector(grid_sizes, smoothed_parameter_array)
end

function fubar_apply_smoothing(grid_sizes::Tuple, codon_param_index_vec::Vector{Vector{Int64}}, ambient_parameters::AbstractVector{<:Real}, kernel_parameters::AbstractVector{<:Real})
    ambient_parameter_array = reshape_probability_vector(grid_sizes, codon_param_index_vec, ambient_parameters)
    kernel = gaussian_kernel(5, kernel_parameters[1]^2) #approximate_gaussian_kernel(kernel_parameters[1]^2, 4)
    smoothed_parameter_array = apply_separable_convolution(ambient_parameter_array, kernel)
    #smoothed_parameter_array = ambient_parameter_array
    return unreshape_probability_vector(codon_param_index_vec, smoothed_parameter_array)
end

function gaussian_kernel(window_size::Int64, variance::Real)
    # window_size should be odd for symmetry, but not mandatory
    radius = (window_size - 1) ÷ 2
    x = -radius:radius                  # symmetric points centered at zero
    kernel = exp.(-(x .^ 2) ./ (2 * variance))  # element-wise Gaussian formula
    kernel /= sqrt(sum(kernel .^ 2))               # This is to make the variance not depend on smoothing TODO: But on the edges this is a bit of a problem?
    return kernel
end

# NOTE: THIS HAS ISSUES BECAUSE OF SDJFKLJSDKLFJKLSDF
# Based on an article: Theoretical foundations of Gaussian Convolution by Extended Box Filtering:
# use repeated box blur to approximate gaussian blur faaast
# Adapted from https://stackoverflow.com/questions/23489902/approximating-gaussian-blur-using-extended-box-blur
function approximate_gaussian_kernel(box_blur_variance::Float64, n_iterations::Int64)
    box_blur_length = sqrt(((12 * box_blur_variance) / n_iterations) + 1)
    box_blur_radius = (box_blur_length - 1) / 2
    # 'box_blur_radius_int' -> 'l' in the reference
    box_blur_radius_int = floor(Int64, box_blur_radius)
    # box_blur_radiusFrac   = box_blur_radius - box_blur_radius_int
    # The length of the "Integer" part of the filter.
    # 'box_blur_length_int' -> 'L' in the reference
    box_blur_length_int = 2 * box_blur_radius_int + 1

    a1 = ((2 * box_blur_radius_int) + 1)
    a2 = (box_blur_radius_int * (box_blur_radius_int + 1)) - ((3 * box_blur_variance) / n_iterations)
    a3 = (6 * ((box_blur_variance / n_iterations) - ((box_blur_radius_int + 1)^2)))
    alpha = a1 * (a2 / a3)
    ww = alpha / ((2 * box_blur_radius_int) + 1 + (2 * alpha))
    # The length of the "Extended Box Filter".
    # 'box_blur_length' -> '\Gamma' in the reference.
    box_blur_length = (2 * (alpha + box_blur_radius_int)) + 1

    # The "Single Box Filter" with Varaince - box_blur_variance / n_iterations
    # It is normalized by definition.
    v_single_box_blur_kernel = vcat(ww, (ones(Float64, box_blur_length_int) / box_blur_length), ww)
    # v_box_blur_kernel = v_box_blur_kernel / sum(v_box_blur_kernel)
    v_box_blur_kernel = v_single_box_blur_kernel
    # singleBoxKernelVar = sum(([-(box_blur_radius_int + 1):(box_blur_radius_int + 1)] .^ 2) .* boxBlurKernel)
    # boxKernelVar = n_iterations * singleBoxKernelVar
    if length(v_single_box_blur_kernel) < n_iterations
        return v_single_box_blur_kernel ./ sum(v_single_box_blur_kernel)
    end
    for _ = 2:n_iterations
        v_box_blur_kernel = conv_pure(v_box_blur_kernel, v_single_box_blur_kernel)
    end
    return v_box_blur_kernel ./ sqrt(sum(v_box_blur_kernel .^ 2)) # To make the variance not depend on smoothing
end



function apply_separable_convolution(x::AbstractArray{<:Real}, kernel::AbstractVector{<:Real})
    y = copy(x) # Zygote does not like mutation
    for d in 1:ndims(x)
        y = convolve_along_dim(y, kernel, d)
    end
    return y
end

#TODO: Really MAKE SURE these are correct. They do seem to give a gaussian blur 
function convolve_along_dim(x::AbstractArray{<:Real}, kernel::AbstractVector{<:Real}, dim::Int)
    perm = (dim, filter(d -> d != dim, 1:ndims(x))...)
    x_perm = permutedims(x, perm)
    sz = size(x_perm)
    reshaped = reshape(x_perm, sz[1], :)
    #result = map(col -> same_conv(col, kernel), eachcol(reshaped))
    result = map(col -> same_conv(col, kernel), eachcol(reshaped))
    result_mat = hcat(result...)
    result_array = reshape(result_mat, sz...)
    return permutedims(result_array, invperm(perm))
end

function conv_pure(x::AbstractVector{<:Real}, kernel::AbstractVector{<:Real})
    nx = length(x)
    nk = length(kernel)
    ny = nx - nk + 1
    return [sum(x[i:i+nk-1] .* kernel) for i in 1:ny]
end

function same_conv(col, kernel)
    klen = length(kernel)
    left = fld(klen - 1, 2)
    right = cld(klen - 1, 2)
    #padded = vcat(zeros(left), col, zeros(right)) # Padding with edge values makes sense here I think, because our best guess at the values directly outside of the grid should be the values at the edge of the grid.
    padded = vcat(reverse(col[2:left+1]), col, reverse(col[end-right:end-1])) # TODO: What is the best behaviour when padding? A bit of a cursed solution would be to pad with more theta values.

    conv_result = conv_pure(padded, kernel)
    return conv_result
end