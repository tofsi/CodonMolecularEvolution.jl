

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

    positive_selection_mask = Bool.(collect(CodonMolecularEvolution.get_pos_sel_mask(flavorgrid)))
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

# The following is for debugging purposes
struct FlavorIdentityTransform
    kernel_dim::Int
    suppression_dim::Int
    kernel_stddev::Float64
    suppression_stddev::Float64
end
# and this...
function (t::FlavorIdentityTransform)(ambient_sample::AbstractVector{<:Real})
    kernel_parameters = ambient_sample[1:t.kernel_dim]
    suppression_parameters = ambient_sample[t.kernel_dim+1:t.kernel_dim+t.suppression_dim]
    ambient_unsuppressed_parameters = ambient_sample[t.kernel_dim+t.suppression_dim+1:end]

    return vcat(
        t.kernel_stddev .* kernel_parameters,
        t.suppression_stddev .* suppression_parameters,
        ambient_unsuppressed_parameters,
    )
end

"""
    SKBDIModel_from_FLAVOR(flavorgrid; kwargs...)

Construct an `SKBDIModel` directly from a `FLAVORgrid`.

The single hypothesis mask corresponds to FLAVOR's own positive-selection-capable
categories, as returned by `CodonMolecularEvolution.get_pos_sel_mask(flavorgrid)`.
"""
function SKBDIModel_from_FLAVOR(flavorgrid::FLAVORgrid;
    normalized::Bool=true,
    kernel_dim::Int=1,
    kernel_stddev::Real=4.0,
    suppress::Bool=false,
    fast_reshaping::Bool=true,
    suppression_stddev::Real=2.0,
    transition_function=s -> quintic_smooth_transition(s, 0.0, 1.0))

    meta = flavor_parameter_metadata(flavorgrid)
    con_lik_matrix = flavor_con_lik_matrix(flavorgrid; normalized=normalized)
    log_con_lik_matrix = log.(con_lik_matrix)

    n_categories = size(con_lik_matrix, 1)
    length(meta.codon_param_vec) == n_categories || throw(DimensionMismatch("Category metadata does not match con_lik_matrix."))
    size(meta.hypothesis_masks, 2) == n_categories || throw(DimensionMismatch("Hypothesis mask does not match con_lik_matrix."))
    reshaping_scheme = fast_reshaping ? FLAVORReshapingScheme(meta.grid_sizes) : GeneralCategoricalReshapingScheme(meta.grid_sizes, meta.codon_param_index_vec)
    ambient_to_parameter_transform = AmbientToParameterTransform(
        reshaping_scheme,
        1,
        suppress ? 1 : 0,
        kernel_stddev,
        suppress ? suppression_stddev : 0.0,
    ) #TODO: grid_based_transform assumes diffubar ordering of codon_param_vec.

    # ambient_to_parameter_transform = identity
    #= ambient_to_parameter_transform = FlavorIdentityTransform(
    kernel_dim,
    0,                  # suppression_dim
    kernel_stddev,
    suppression_stddev,
    ) =#
    return SKBDIModel(
        meta.parameter_grids,
        meta.parameter_names,
        suppress ? meta.hypothesis_masks : nothing,
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



# same formula as in FLAVOR
bayes_factor_bame_analog(posterior, prior) = (posterior / (1 - posterior)) / (prior / (1 - prior))

function summarize_smoothFLAVOR_BAME(
    flavorgrid,
    fubar_model::GeneralizedFUBARModel,
    ambient_samples::Vector;
    burnin::Int,
    pos_thresh::Float64=0.9,
    sample_allocations::Bool=false,
    progress::Bool=false,
)
    con_lik = fubar_model.con_lik_matrix
    n_categories, n_sites = size(con_lik)

    pos_sel_mask = CodonMolecularEvolution.get_pos_sel_mask(flavorgrid)

    posterior_mat = zeros(Float64, n_categories, n_sites)
    θ_mean = zeros(Float64, n_categories)
    alloc_grid = sample_allocations ? zeros(Int, n_categories, n_sites) : nothing

    n_used = 0
    v = zeros(Float64, n_categories)

    p = progress ? ProgressMeter.Progress(sum(length(chain) - burnin for chain in ambient_samples);
        desc="Summarizing posterior") : nothing

    for chain in ambient_samples
        for t in burnin+1:length(chain)
            θ = fubar_model.to_probability_vector(chain[t])
            θ_mean .+= θ
            n_used += 1

            for s in 1:n_sites
                @inbounds v .= θ .* con_lik[:, s]
                z = sum(v)

                # Should not happen if con_lik columns are valid, but guard anyway
                if z <= 0
                    continue
                end

                @inbounds posterior_mat[:, s] .+= v ./ z

                if sample_allocations
                    k = sample(1:n_categories, Weights(v))
                    alloc_grid[k, s] += 1
                end
            end

            progress ? ProgressMeter.next!(p) : nothing
        end
    end

    θ_mean ./= n_used
    posterior_mat ./= n_used

    posterior_probs = vec(sum(posterior_mat[pos_sel_mask, :], dims=1))

    # BAME-like plug-in prior mass
    pos_prior = sum(θ_mean[pos_sel_mask])

    # avoid 0/1 blowups
    eps = 1e-12
    posterior_probs_clamped = clamp.(posterior_probs, eps, 1 - eps)
    pos_prior_clamped = clamp(pos_prior, eps, 1 - eps)

    bayes_factors = bayes_factor_bame_analog.(posterior_probs_clamped, pos_prior_clamped)

    df = DataFrame(
        site=1:n_sites,
        posterior_prob_positive=posterior_probs,
        bayes_factor=bayes_factors,
        threshold=posterior_probs .> pos_thresh,
    )

    return (
        df=df,
        posterior_mat=posterior_mat,
        posterior_probs=posterior_probs,
        bayes_factors=bayes_factors,
        θ_mean=θ_mean,
        pos_prior=pos_prior,
        pos_sel_mask=pos_sel_mask,
        alloc_grid=alloc_grid,
        n_used=n_used,
    )
end

function smoothFLAVOR_BAME(
    flavorgrid,
    outpath;
    pos_thresh=0.9,
    iters=10,
    burnin=div(iters, 4),
    n_chains=4,
    verbosity=1,
    exports=true,
    sample_allocations=false,
)
    sk_model = SKBDIModel_from_FLAVOR(flavorgrid)
    fubar_model = GeneralizedFUBARModel(sk_model)

    if verbosity > 0
        println("Sampling from smoothFLAVOR with NUTS.")
    end

    ambient_samples, stats = sample_NUTS(fubar_model, iters, n_chains; progress=verbosity > 0)

    summary = summarize_smoothFLAVOR_BAME(
        flavorgrid,
        fubar_model,
        ambient_samples;
        burnin=burnin,
        pos_thresh=pos_thresh,
        sample_allocations=sample_allocations,
        progress=verbosity > 0,
    )

    if exports
        CSV.write(outpath * "_smoothFLAVOR_BAME.csv", summary.df)
    end

    return summary.df, (
        ambient_samples=ambient_samples,
        fubar_model=fubar_model,
        sk_model=sk_model,
        posterior_mat=summary.posterior_mat,
        θ_mean=summary.θ_mean,
        posterior_probs=summary.posterior_probs,
        bayes_factors=summary.bayes_factors,
        pos_sel_mask=summary.pos_sel_mask,
        alloc_grid=summary.alloc_grid,
        stats=stats,
    )
end