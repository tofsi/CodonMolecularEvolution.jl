"""
ProbabilityVectorReshapingScheme
This type is used to determine how (un)reshaping probability vector to(from) probability array works.
(un)reshaping then works by calling (un_)reshape_probability_vector(reshaping_scheme::ProbabilityVectorReshapingScheme, probability_vector(array))
"""
abstract type ProbabilityVectorReshapingScheme end

"""
GeneralCategoricalReshapingScheme
Works for a general indexation, but makes reshaping slower than for e.g. DifFUBARReshapingScheme
"""
struct GeneralCategoricalReshapingScheme{N,M} <: ProbabilityVectorReshapingScheme
    grid_sizes::NTuple{N,Int}
    codon_param_index_vec::NTuple{M,CartesianIndex{N}}
end

# Constructor, We convert the codon_param_index_vec to an immutable tuple representation because otherwise it is not handled well by AD
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
temporary reshaped array into the desired logical array axis order.
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
# reshape_probability_vector(grid_sizes::Tuple, codon_param_index_vec::Vector{Vector{Int64}}, probability_vector::AbstractVector{<:Real})
Takes a vector with index according to codon_param_index_vec 
and returns the corresponding multidimensional array with shape according to grid sizes.
"""
function reshape_probability_vector(reshaping_scheme::GeneralCategoricalReshapingScheme{N,M}, probability_vector::AbstractVector{T}) where {N,M,T<:Real}
    probability_array = Array{T}(undef, reshaping_scheme.grid_sizes)
    @inbounds for i = 1:M
        probability_array[reshaping_scheme.codon_param_index_vec[i]] = probability_vector[i]
    end
    return probability_array
end

"""
# unreshape_probability_vector(codon_param_index_vec::Vector{Vector{Int64}}, probability_array::AbstractArray{<:Real})
Takes a multidimensional array and returns the corresponding vector with index according to codon_param_index_vec.
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

FLAVORReshapingScheme(grid_sizes::NTuple{4,Int}) =
    PermutedDimsReshapingScheme(grid_sizes, (3, 2, 1, 4))

# TODO: Verify that the following struct works for difFUBAR.
#= difFUBARReshapingScheme(grid_sizes::NTuple{N,Int}) where {N} =
PermutedDimsReshapingScheme(grid_sizes, ntuple(i -> N - i + 1, N)) =#
"""
DifFUBARReshapingScheme
Used specifically in the case when the codon_param_index_vec matches the difFUBAR indexation
"""
struct DifFUBARReshapingScheme{N} <: ProbabilityVectorReshapingScheme
    grid_sizes::NTuple{N,Int}
end


"""
# reshape_probability_vector(grid_sizes::Tuple, probability_vector::AbstractVector{<:Real})
Takes a vector with index according to codon_param_index_vec and reshapes it into a multidimensional array based on the grid sizes.
NOTE: Only works when the index matches con_lik_mat from difFUBAR_grid(), but is fast and AD safe
"""
function reshape_probability_vector(reshaping_scheme::DifFUBARReshapingScheme, probability_vector::AbstractVector{<:Real})
    return permutedims(reshape(probability_vector, reverse(reshaping_scheme.grid_sizes)...),
        reverse(1:length(reshaping_scheme.grid_sizes)))
end

"""
# unreshape_probability_vector()
Takes a multidimensional array and returns the corresponding vector with index according to codon_param_index_vec.
NOTE: Only works when the index matches con_lik_mat from difFUBAR_grid, but is fast and AD safe
"""
function unreshape_probability_vector(reshaping_scheme::DifFUBARReshapingScheme, probability_array::AbstractArray{<:Real})
    return vec(permutedims(probability_array, reverse(1:length(reshaping_scheme.grid_sizes))))
end
