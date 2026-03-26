using Random
using Distributions

# Reuse the generic categorical-model machinery from skbDifFUBAR.jl.
if !isdefined(@__MODULE__, :GeneralizedFUBARModel)
    include("skbDifFUBAR.jl")
end

# Reuse FLAVOR's own grid object and helper functions when available.
# The adapter is still usable if the caller has already loaded FLAVOR.jl before this file.
if !isdefined(@__MODULE__, :gamma_slices) || !isdefined(@__MODULE__, :get_pos_sel_mask)
    try
        include("FLAVOR.jl")
    catch
        # Allow loading to succeed in contexts where FLAVOR.jl is included elsewhere.
    end
end

"""
    FLAVORCategoricalData

A categorical representation of a `FLAVORgrid` suitable for downstream Bayesian
inference on the category weights `theta`.

Fields
- `parameter_grids`: parameter grids in the order `(mu, shape, alpha, capped)`.
- `parameter_names`: names corresponding to `parameter_grids`.
- `codon_param_vec`: parameter values for each category.
- `codon_param_index_vec`: integer grid indices for each category.
- `hypothesis_masks`: boolean masks indexed as `[hypothesis, category]`.
- `con_lik_matrix`: conditional likelihood matrix indexed by `(category, site)`.
- `log_con_lik_matrix`: `log.(con_lik_matrix)`.
- `positive_selection_mask`: copy of the first hypothesis mask.
- `category_labels`: named tuple metadata for each category.
- `grid_sizes`: tuple of grid dimensions.
"""
struct FLAVORCategoricalData
    parameter_grids::Vector{Vector{Float64}}
    parameter_names::Vector{String}
    codon_param_vec::Vector{Vector{Float64}}
    codon_param_index_vec::Vector{Vector{Int64}}
    hypothesis_masks::Matrix{Bool}
    con_lik_matrix::Matrix{Float64}
    log_con_lik_matrix::Matrix{Float64}
    positive_selection_mask::Vector{Bool}
    category_labels::Vector{NamedTuple{(:mu, :shape, :alpha, :capped), Tuple{Float64, Float64, Float64, Bool}}}
    grid_sizes::Tuple
end

"""
    DirichletLogitPrior(alpha)

Distribution on an unconstrained vector `η ∈ R^(K-1)` induced by the transform

```math
θ = softmax([η; 0])
```

with a Dirichlet prior on `θ`.
"""
struct DirichletLogitPrior <: ContinuousMultivariateDistribution
    alpha::Vector{Float64}
    dirichlet::Dirichlet

    function DirichletLogitPrior(alpha::AbstractVector{<:Real})
        length(alpha) >= 2 || throw(ArgumentError("DirichletLogitPrior needs at least two categories."))
        all(alpha .> 0) || throw(ArgumentError("All Dirichlet concentration parameters must be positive."))
        α = Float64.(collect(alpha))
        new(α, Dirichlet(α))
    end
end

Base.length(d::DirichletLogitPrior) = length(d.alpha) - 1
Base.eltype(::Type{DirichletLogitPrior}) = Float64
Distributions.minimum(d::DirichletLogitPrior) = fill(-Inf, length(d))
Distributions.maximum(d::DirichletLogitPrior) = fill(Inf, length(d))

"""
    logit_simplex_map(eta)

Map `eta ∈ R^(K-1)` to `theta ∈ Δ^(K-1)` using

```math
θ = softmax([η; 0]).
```
"""
function logit_simplex_map(eta::AbstractVector{<:Real})
    z = vcat(Float64.(eta), 0.0)
    zmax = maximum(z)
    ez = exp.(z .- zmax)
    return ez ./ sum(ez)
end

"""
    inverse_logit_simplex_map(theta)

Inverse of `logit_simplex_map`, returning `eta_i = log(theta_i / theta_K)`.
"""
function inverse_logit_simplex_map(theta::AbstractVector{<:Real})
    length(theta) >= 2 || throw(ArgumentError("theta must have at least two entries."))
    all(theta .> 0) || throw(ArgumentError("All entries of theta must be strictly positive."))
    s = sum(theta)
    isapprox(s, 1.0; atol=1e-8, rtol=1e-8) || throw(ArgumentError("theta must sum to 1."))
    return log.(Float64.(theta[1:end-1])) .- log(Float64(theta[end]))
end

function Random.rand(rng::Random.AbstractRNG, d::DirichletLogitPrior)
    θ = rand(rng, d.dirichlet)
    return inverse_logit_simplex_map(θ)
end

Random.rand(d::DirichletLogitPrior) = rand(Random.default_rng(), d)

function Distributions.logpdf(d::DirichletLogitPrior, eta::AbstractVector{<:Real})
    length(eta) == length(d) || throw(DimensionMismatch("Expected parameter vector of length $(length(d)); got $(length(eta))."))
    θ = logit_simplex_map(eta)
    # |det J_{eta -> theta[1:K-1]}| = prod(theta)
    return logpdf(d.dirichlet, θ) + sum(log, θ)
end

"""
    flavor_gamma_slices(mu, shape, slices=20)

Use FLAVOR's own Gamma discretization when available. Falls back to the same midpoint-
quantile construction if `gamma_slices` is not loaded.
"""
function flavor_gamma_slices(mu::Real, shape::Real, slices::Int=20)
    if isdefined(@__MODULE__, :gamma_slices)
        return Float64.(gamma_slices(mu, shape, slices))
    end
    shape > 0 || throw(ArgumentError("shape must be positive."))
    mu > 0 || throw(ArgumentError("mu must be positive."))
    slices >= 1 || throw(ArgumentError("slices must be at least 1."))
    c = 1 / (2 * slices)
    dist = Gamma(shape, mu / shape)
    return quantile.(Ref(dist), c:(2c):(1 - c))
end

"""
    flavor_positive_selection_mask(flavorgrid; slices=20)

Use FLAVOR's own positive-selection mask logic when available.
"""
function flavor_positive_selection_mask(flavorgrid; slices::Int=20)
    if isdefined(@__MODULE__, :get_pos_sel_mask)
        return Bool.(collect(get_pos_sel_mask(flavorgrid)))
    end

    mugrid = Float64.(collect(getproperty(flavorgrid, :mugrid)))
    shapegrid = Float64.(collect(getproperty(flavorgrid, :shapegrid)))
    alphagrid = Float64.(collect(getproperty(flavorgrid, :alphagrid)))

    mask = Bool[]
    for capped in (false, true)
        for mu in mugrid, shape in shapegrid, alpha in alphagrid
            _ = alpha
            positive = !capped && any(flavor_gamma_slices(mu, shape, slices) .> 1.0)
            push!(mask, positive)
        end
    end
    return mask
end

"""
    flavor_grid_sizes(flavorgrid)

Grid shape corresponding to the category ordering used by FLAVOR:
`(mu, shape, alpha, capped)`.
"""
function flavor_grid_sizes(flavorgrid)
    return (length(getproperty(flavorgrid, :mugrid)),
        length(getproperty(flavorgrid, :shapegrid)),
        length(getproperty(flavorgrid, :alphagrid)),
        2)
end

"""
    flavor_parameter_metadata(flavorgrid; slices=20)

Construct categorical metadata in the same category order as FLAVOR itself:
all uncapped grid points first, then all capped grid points. This reuses
`flavorgrid.gridpoints` and `get_pos_sel_mask` when available rather than rebuilding
FLAVOR's category logic from scratch.
"""
function flavor_parameter_metadata(flavorgrid; slices::Int=20)
    mugrid = Float64.(collect(getproperty(flavorgrid, :mugrid)))
    shapegrid = Float64.(collect(getproperty(flavorgrid, :shapegrid)))
    alphagrid = Float64.(collect(getproperty(flavorgrid, :alphagrid)))
    gridpoints = collect(getproperty(flavorgrid, :gridpoints))

    parameter_grids = [mugrid, shapegrid, alphagrid, [0.0, 1.0]]
    parameter_names = ["mu", "shape", "alpha", "capped"]
    grid_sizes = flavor_grid_sizes(flavorgrid)

    mu_index = Dict(mu => i for (i, mu) in enumerate(mugrid))
    shape_index = Dict(shape => i for (i, shape) in enumerate(shapegrid))
    alpha_index = Dict(alpha => i for (i, alpha) in enumerate(alphagrid))

    codon_param_vec = Vector{Float64}[]
    codon_param_index_vec = Vector{Int64}[]
    category_labels = NamedTuple{(:mu, :shape, :alpha, :capped), Tuple{Float64, Float64, Float64, Bool}}[]

    for (cap_index, capped) in enumerate((false, true))
        for gp in gridpoints
            mu, shape, alpha = Float64.(Tuple(gp))
            push!(codon_param_vec, [mu, shape, alpha, Float64(capped)])
            push!(codon_param_index_vec, [mu_index[mu], shape_index[shape], alpha_index[alpha], cap_index])
            push!(category_labels, (mu=mu, shape=shape, alpha=alpha, capped=capped))
        end
    end

    positive_selection_mask = flavor_positive_selection_mask(flavorgrid; slices=slices)
    hypothesis_masks = reshape(copy(positive_selection_mask), 1, :)

    return parameter_grids, parameter_names, codon_param_vec, codon_param_index_vec,
        hypothesis_masks, positive_selection_mask, category_labels, grid_sizes
end

"""
    flavor_con_lik_matrix(flavorgrid; normalized=true)

Extract the conditional likelihood matrix indexed by `(category, site)`.

With `normalized=true` (the default), this returns `flavorgrid.prob_matrix`, whose
columns sum to 1. This differs from FLAVOR's original conditional likelihood matrix
only by a site-specific multiplicative constant, so it yields the same posterior over
`theta` and the same likelihood up to an additive constant.

With `normalized=false`, the function reconstructs the unnormalized matrix using
`site_scalers`.
"""
function flavor_con_lik_matrix(flavorgrid; normalized::Bool=true)
    prob_matrix = Float64.(Matrix(getproperty(flavorgrid, :prob_matrix)))
    if normalized
        return prob_matrix
    end

    if !hasproperty(flavorgrid, :site_scalers)
        throw(ArgumentError("normalized=false requires the FLAVOR object to have a `site_scalers` field."))
    end

    site_scalers = Float64.(collect(getproperty(flavorgrid, :site_scalers)))
    return prob_matrix .* reshape(exp.(site_scalers), 1, :)
end

"""
    flavor_categorical_data(flavorgrid; slices=20, normalized=true)

Convert a FLAVOR grid object into the categorical representation needed by
`GeneralizedFUBARModel`.
"""
function flavor_categorical_data(flavorgrid; slices::Int=20, normalized::Bool=true)
    parameter_grids, parameter_names, codon_param_vec, codon_param_index_vec,
    hypothesis_masks, positive_selection_mask, category_labels, grid_sizes =
        flavor_parameter_metadata(flavorgrid; slices=slices)

    con_lik_matrix = flavor_con_lik_matrix(flavorgrid; normalized=normalized)
    log_con_lik_matrix = log.(con_lik_matrix)

    n_categories_expected = 2 * length(getproperty(flavorgrid, :gridpoints))
    n_categories_observed = size(con_lik_matrix, 1)
    n_categories_observed == n_categories_expected || throw(DimensionMismatch(
        "Expected $n_categories_expected FLAVOR categories, but con_lik_matrix has $n_categories_observed rows."
    ))

    return FLAVORCategoricalData(
        parameter_grids,
        parameter_names,
        codon_param_vec,
        codon_param_index_vec,
        hypothesis_masks,
        con_lik_matrix,
        log_con_lik_matrix,
        positive_selection_mask,
        category_labels,
        grid_sizes,
    )
end

"""
    GeneralizedFUBARModel(data::FLAVORCategoricalData; dirichlet_alpha=1.0)

Construct a `GeneralizedFUBARModel` from the categorical FLAVOR representation using
an unconstrained additive-log-ratio parameterization and a true Dirichlet prior on
`theta`.
"""
function GeneralizedFUBARModel(data::FLAVORCategoricalData; dirichlet_alpha::Union{Real, AbstractVector{<:Real}}=1.0)
    n_categories = size(data.con_lik_matrix, 1)
    α = dirichlet_alpha isa Real ? fill(Float64(dirichlet_alpha), n_categories) : Float64.(collect(dirichlet_alpha))
    length(α) == n_categories || throw(DimensionMismatch("dirichlet_alpha must have length $n_categories."))

    prior = DirichletLogitPrior(α)
    to_probability_vector = logit_simplex_map
    log_likelihood = eta -> sum(log.(data.con_lik_matrix' * to_probability_vector(eta)))

    return GeneralizedFUBARModel(
        n_categories - 1,
        n_categories,
        log_likelihood,
        to_probability_vector,
        prior,
        data.con_lik_matrix,
    )
end

"""
    GeneralizedFUBARModel_from_FLAVOR(flavorgrid; dirichlet_alpha=1.0, slices=20, normalized=true)

Convert a `FLAVORgrid` into categorical FLAVOR metadata and then into a
`GeneralizedFUBARModel`.

Returns a named tuple with:
- `model`: the `GeneralizedFUBARModel`
- `data`: the corresponding `FLAVORCategoricalData`
"""
function GeneralizedFUBARModel_from_FLAVOR(flavorgrid;
    dirichlet_alpha::Union{Real, AbstractVector{<:Real}}=1.0,
    slices::Int=20,
    normalized::Bool=true)

    data = flavor_categorical_data(flavorgrid; slices=slices, normalized=normalized)
    model = GeneralizedFUBARModel(data; dirichlet_alpha=dirichlet_alpha)
    return (model=model, data=data)
end

"""
    flavor_theta_samples(ambient_samples)

Convert posterior samples in the unconstrained parameterization to samples on the simplex.
"""
function flavor_theta_samples(ambient_samples)
    return [logit_simplex_map(sample) for sample in ambient_samples]
end

"""
    flavor_positive_selection_prior_mass(theta, data)

Given a simplex point `theta` and a `FLAVORCategoricalData` object, return the total
mass assigned to positive-selection-capable categories.
"""
function flavor_positive_selection_prior_mass(theta::AbstractVector{<:Real}, data::FLAVORCategoricalData)
    length(theta) == length(data.positive_selection_mask) || throw(DimensionMismatch("theta and positive_selection_mask must have the same length."))
    return sum(Float64.(theta) .* Float64.(data.positive_selection_mask))
end

# Backwards-compatible alias for the earlier name.
flavor_positive_selection_posterior(theta::AbstractVector{<:Real}, data::FLAVORCategoricalData) =
    flavor_positive_selection_prior_mass(theta, data)

function BAME(f::FLAVORgrid, outpath; pos_thresh=0.9, verbosity=1, plots = true)
    x = GeneralizedFUBARModel_from_FLAVOR(f; dirichlet_alpha=1.0)

    model = x.model
    data  = x.data

    ambient_samples, stats = sample_NUTS(model, iters, n_chains)

    # flatten chains, drop burnin
    samples = vcat([ambient_samples[c][burnin+1:end] for c in 1:length(ambient_samples)]...)

    theta_samples = [model.to_probability_vector(s) for s in samples]

    # posterior responsibilities
    K, S = size(model.con_lik_matrix)
    posterior_mat = zeros(K, S)

    for θ in theta_samples
        for s in 1:S
            w = θ .* model.con_lik_matrix[:, s]
            posterior_mat[:, s] .+= w ./ sum(w)
        end
    end
    posterior_mat ./= length(theta_samples)

    P_pos = vec(sum(posterior_mat[data.pos_sel_mask, :], dims=1))

    if plots
        gen_BAME_plots(P_pos, bfs)
    end
    
    if verbosity > 0
        for i in 1:length(posterior_pos)
            if posterior_pos[i] > pos_thresh
                println("Site $(i): P(β>α) on some branches = $(round(posterior_pos[i],digits=4))");
            end
        end
    end

end

function FLAVOR(f::FLAVORgrid, outpath; pos_thresh=0.9, verbosity=1, method = (sampler = :DirichletEM, concentration = 0.1, iterations = 2500), plots = true)
    return BAME(f, outpath, pos_thresh=pos_thresh, verbosity=verbosity, method=method, plots=plots)
end


