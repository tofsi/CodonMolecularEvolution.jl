using Distributions

if !isdefined(@__MODULE__, :SKBDIModel)
    include("skbDifFUBAR.jl")
end

if !isdefined(@__MODULE__, :FLAVORgrid)
    include("FLAVOR.jl")
end

function require_flavor_symbols()
    needed = (:FLAVORgrid, :get_pos_sel_mask)
    missing = [s for s in needed if !isdefined(@__MODULE__, s)]
    isempty(missing) || error("smoothFLAVOR.jl requires FLAVOR.jl to be loaded. Missing symbols: $(join(string.(missing), ", ")).")
    return nothing
end

"""
    flavor_con_lik_matrix(flavorgrid; normalized=true)

Return the category-by-site conditional likelihood matrix used by FLAVOR.

With `normalized=true`, this is `flavorgrid.prob_matrix`, which differs from the
unnormalized conditional likelihood matrix only by a site-specific multiplicative
constant and therefore induces the same posterior over category weights.
"""
function flavor_con_lik_matrix(flavorgrid::FLAVORgrid; normalized::Bool=true)
    prob_matrix = Float64.(Matrix(getproperty(flavorgrid, :prob_matrix)))
    if normalized
        return prob_matrix
    end

    hasproperty(flavorgrid, :site_scalers) || throw(ArgumentError("normalized=false requires flavorgrid.site_scalers."))
    site_scalers = Float64.(collect(getproperty(flavorgrid, :site_scalers)))
    return prob_matrix .* reshape(exp.(site_scalers), 1, :)
end

"""
    flavor_parameter_metadata(flavorgrid)

Construct the category metadata needed by `SKBDIModel` in FLAVOR's own category order:
all uncapped grid points first, then all capped grid points.
"""
function flavor_parameter_metadata(flavorgrid::FLAVORgrid)
    require_flavor_symbols()

    mugrid = Float64.(collect(getproperty(flavorgrid, :mugrid)))
    shapegrid = Float64.(collect(getproperty(flavorgrid, :shapegrid)))
    alphagrid = Float64.(collect(getproperty(flavorgrid, :alphagrid)))
    gridpoints = collect(getproperty(flavorgrid, :gridpoints))

    mu_index = Dict(mu => i for (i, mu) in enumerate(mugrid))
    shape_index = Dict(shape => i for (i, shape) in enumerate(shapegrid))
    alpha_index = Dict(alpha => i for (i, alpha) in enumerate(alphagrid))

    codon_param_vec = Vector{Float64}[]
    codon_param_index_vec = Vector{Int64}[]

    for (cap_index, capped) in enumerate((false, true))
        for gp in gridpoints
            mu, shape, alpha = Float64.(Tuple(gp))
            push!(codon_param_vec, [mu, shape, alpha, Float64(capped)])
            push!(codon_param_index_vec, [mu_index[mu], shape_index[shape], alpha_index[alpha], cap_index])
        end
    end

    positive_selection_mask = Bool.(collect(get_pos_sel_mask(flavorgrid)))
    length(positive_selection_mask) == length(codon_param_vec) || throw(DimensionMismatch("Positive-selection mask does not match number of categories."))

    return (
        parameter_grids=[mugrid, shapegrid, alphagrid, [0.0, 1.0]],
        parameter_names=["mu", "shape", "alpha", "capped"],
        codon_param_vec=codon_param_vec,
        codon_param_index_vec=codon_param_index_vec,
        hypothesis_masks=reshape(copy(positive_selection_mask), 1, :),
        grid_sizes=(length(mugrid), length(shapegrid), length(alphagrid), 2),
    )
end

"""
    SKBDIModel_from_FLAVOR(flavorgrid; kwargs...)

Construct an `SKBDIModel` directly from a `FLAVORgrid`.

The single hypothesis mask corresponds to FLAVOR's own positive-selection-capable
categories, as returned by `get_pos_sel_mask(flavorgrid)`.
"""
function SKBDIModel_from_FLAVOR(flavorgrid::FLAVORgrid;
    normalized::Bool=true,
    kernel_dim::Int=1,
    kernel_stddev::Real=4.0,
    suppression_stddev::Real=2.0,
    transition_function=s -> CodonMolecularEvolution.quintic_smooth_transition(s, 0.0, 1.0))

    meta = flavor_parameter_metadata(flavorgrid)
    con_lik_matrix = flavor_con_lik_matrix(flavorgrid; normalized=normalized)
    log_con_lik_matrix = log.(con_lik_matrix)

    n_categories = size(con_lik_matrix, 1)
    length(meta.codon_param_vec) == n_categories || throw(DimensionMismatch("Category metadata does not match con_lik_matrix."))
    size(meta.hypothesis_masks, 2) == n_categories || throw(DimensionMismatch("Hypothesis mask does not match con_lik_matrix."))

    ambient_to_parameter_transform = ambient_sample -> fubar_ambient_to_parameter_transform(
        ambient_sample,
        meta.grid_sizes,
        meta.codon_param_index_vec,
        kernel_dim,
        size(meta.hypothesis_masks, 1),
        Float64(kernel_stddev),
        Float64(suppression_stddev),
    )

    return SKBDIModel(
        meta.parameter_grids,
        meta.parameter_names,
        meta.hypothesis_masks,
        transition_function,
        log_con_lik_matrix,
        con_lik_matrix,
        meta.codon_param_vec,
        meta.codon_param_index_vec,
        ambient_to_parameter_transform,
        kernel_dim,
        meta.grid_sizes,
    )
end

"""
    GeneralizedFUBARModel_from_FLAVOR(flavorgrid; kwargs...)

Convenience constructor returning `GeneralizedFUBARModel(SKBDIModel_from_FLAVOR(flavorgrid; kwargs...))`.
"""
function GeneralizedFUBARModel_from_FLAVOR(flavorgrid::FLAVORgrid; kwargs...)
    return GeneralizedFUBARModel(SKBDIModel_from_FLAVOR(flavorgrid; kwargs...))
end
