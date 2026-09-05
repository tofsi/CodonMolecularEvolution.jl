"""
    ProbabilityVectorReshapingScheme

Abstract interface for mapping category vectors to parameter-grid arrays and
back. Implementations define [`reshape_probability_vector`](@ref) and
[`unreshape_probability_vector`](@ref).
"""
abstract type ProbabilityVectorReshapingScheme end

"""
    GeneralCategoricalReshapingScheme(grid_sizes, codon_param_index_vec)

Represent an arbitrary category order with explicit Cartesian grid indices.
This is more general, but slower, than a permutation-based reshaping scheme.
"""
struct GeneralCategoricalReshapingScheme{N,M} <: ProbabilityVectorReshapingScheme
    grid_sizes::NTuple{N,Int}
    codon_param_index_vec::NTuple{M,CartesianIndex{N}}
end

# Immutable Cartesian indices keep the general scheme compatible with AD.
function GeneralCategoricalReshapingScheme(
    grid_sizes::NTuple{N,Int},
    codon_param_index_vec::AbstractVector{<:AbstractVector{<:Integer}},
) where {N}
    inds = Tuple(CartesianIndex(Tuple(idx)) for idx in codon_param_index_vec)
    return GeneralCategoricalReshapingScheme{N,length(inds)}(grid_sizes, inds)
end


"""
    PermutedDimsReshapingScheme(grid_sizes, perm)

Fast AD-safe reshaping scheme for probability vectors produced by a nested loop.

`perm` is the permutation passed to `permutedims(tmp, perm)` to turn the
temporary reshaped array into the desired logical axis order.
"""
struct PermutedDimsReshapingScheme{N} <: ProbabilityVectorReshapingScheme
    grid_sizes::NTuple{N,Int}
    perm::NTuple{N,Int}      # tmp -> logical array order
    invperm::NTuple{N,Int}   # logical array -> tmp order
end

_tuple_invperm(p::NTuple{N,Int}) where {N} =
    ntuple(i -> findfirst(==(i), p), N)

function PermutedDimsReshapingScheme(
    grid_sizes::NTuple{N,Int},
    perm::NTuple{N,Int},
) where {N}
    sort(collect(perm)) == collect(1:N) ||
        throw(ArgumentError("perm must be a permutation of 1:$N"))
    return PermutedDimsReshapingScheme{N}(grid_sizes, perm, _tuple_invperm(perm))
end

PermutedDimsReshapingScheme(
    grid_sizes::NTuple{N,Int},
    perm::AbstractVector{<:Integer},
) where {N} = PermutedDimsReshapingScheme(grid_sizes, Tuple(Int.(perm)))

"""
    reshape_probability_vector(scheme, probability_vector)

Map a category-ordered probability vector to its parameter-grid array.
"""
function reshape_probability_vector(reshaping_scheme::GeneralCategoricalReshapingScheme{N,M}, probability_vector::AbstractVector{T}) where {N,M,T<:Real}
    probability_array = Array{T}(undef, reshaping_scheme.grid_sizes)
    @inbounds for i = 1:M
        probability_array[reshaping_scheme.codon_param_index_vec[i]] = probability_vector[i]
    end
    return probability_array
end

"""
    unreshape_probability_vector(scheme, probability_array)

Map a parameter-grid array back to the category order represented by `scheme`.
"""
function unreshape_probability_vector(reshaping_scheme::GeneralCategoricalReshapingScheme{N,M}, probability_array::AbstractArray{T}) where {N,M,T<:Real}
    probability_vector = Vector{T}(undef, M)
    @inbounds for i = 1:M
        probability_vector[i] = probability_array[reshaping_scheme.codon_param_index_vec[i]]
    end
    return probability_vector
end

function reshape_probability_vector(
    s::PermutedDimsReshapingScheme{N},
    probability_vector::AbstractVector{T},
) where {N,T<:Real}
    tmp = reshape(probability_vector,
        ntuple(i -> s.grid_sizes[s.invperm[i]], N)...)
    return permutedims(tmp, s.perm)
end

function unreshape_probability_vector(
    s::PermutedDimsReshapingScheme{N},
    probability_array::AbstractArray{T,N},
) where {N,T<:Real}
    return vec(permutedims(probability_array, s.invperm))
end

"""
    FLAVORReshapingScheme(grid_sizes)

Return the fast, AD-safe scheme for FLAVOR category order.

FLAVOR enumerates categories with `alpha` varying fastest, followed by `shape`,
`mu`, and finally `capped`. The logical array axes remain
`(mu, shape, alpha, capped)`. Use [`GeneralCategoricalReshapingScheme`](@ref) for
categories in any other order.
"""
FLAVORReshapingScheme(grid_sizes::NTuple{4,Int}) =
    PermutedDimsReshapingScheme(grid_sizes, (3, 2, 1, 4))

"""
    DifFUBARReshapingScheme(grid_sizes)

Fast reshaping scheme for category vectors in difFUBAR grid order.
"""
struct DifFUBARReshapingScheme{N} <: ProbabilityVectorReshapingScheme
    grid_sizes::NTuple{N,Int}
end


"""
    reshape_probability_vector(scheme::DifFUBARReshapingScheme, probability_vector)

Map a difFUBAR-ordered category vector to its parameter-grid array. This fast,
AD-safe method requires the category order produced by `difFUBAR_grid`.
"""
function reshape_probability_vector(reshaping_scheme::DifFUBARReshapingScheme, probability_vector::AbstractVector{<:Real})
    return permutedims(reshape(probability_vector, reverse(reshaping_scheme.grid_sizes)...),
        reverse(1:length(reshaping_scheme.grid_sizes)))
end

"""
    unreshape_probability_vector(scheme::DifFUBARReshapingScheme, probability_array)

Map a parameter-grid array back to the category order produced by
`difFUBAR_grid`.
"""
function unreshape_probability_vector(reshaping_scheme::DifFUBARReshapingScheme, probability_array::AbstractArray{<:Real})
    return vec(permutedims(probability_array, reverse(1:length(reshaping_scheme.grid_sizes))))
end
