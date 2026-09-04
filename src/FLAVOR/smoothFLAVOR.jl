

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
    covariance_jitter::Real=1e-6,
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
        (1, 3),  # μ, α; do not smooth shape, "capped"
        smoothing_method=GaussianCovarianceSmoothing(covariance_jitter),
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
                z = 0.0
                @inbounds @simd for k in 1:n_categories
                    weight = θ[k] * con_lik[k, s]
                    v[k] = weight
                    z += weight
                end

                # Should not happen if con_lik columns are valid, but guard anyway
                if z <= 0
                    continue
                end

                inverse_z = inv(z)
                @inbounds @simd for k in 1:n_categories
                    posterior_mat[k, s] += v[k] * inverse_z
                end

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

function _ambient_chain_array(ambient_samples::Vector)
    n_chains = length(ambient_samples)
    n_chains > 0 || error("No NUTS chains were returned")
    n_iterations = length(first(ambient_samples))
    n_iterations > 0 || error("NUTS returned an empty chain")
    n_parameters = length(first(first(ambient_samples)))

    samples = Array{Float64}(undef, n_iterations, n_parameters, n_chains)
    for chain_index in eachindex(ambient_samples)
        chain = ambient_samples[chain_index]
        length(chain) == n_iterations ||
            throw(DimensionMismatch("NUTS chains have unequal iteration counts"))
        for iteration in eachindex(chain)
            length(chain[iteration]) == n_parameters ||
                throw(DimensionMismatch("NUTS samples have unequal parameter counts"))
            samples[iteration, :, chain_index] .= Float64.(chain[iteration])
        end
    end
    return samples
end

function _ambient_parameter_names(sk_model::SKBDIModel)
    names = String[]
    append!(names, ["kernel_$(i)" for i in 1:sk_model.kernel_dim])
    append!(names, ["suppression_$(i)" for i in 1:sk_model.suppression_dim])
    append!(names, ["ambient_weight_$(i)" for i in 1:sk_model.unsuppressed_dim])
    length(names) == sk_model.total_dim ||
        throw(DimensionMismatch("Parameter names do not match the NUTS model dimension"))
    return names
end

function save_smoothFLAVOR_chain_artifacts!(
    outpath::AbstractString,
    ambient_samples::Vector,
    parameter_names::AbstractVector{<:AbstractString};
    burnin::Int,
    n_adapts::Int,
    save_chain_samples::Bool=true,
)
    samples = _ambient_chain_array(ambient_samples)
    n_iterations, n_parameters, n_chains = size(samples)
    0 <= n_adapts <= burnin < n_iterations || throw(ArgumentError(
        "Require 0 <= n_adapts <= burnin < chain length; got " *
        "n_adapts=$n_adapts, burnin=$burnin, chain length=$n_iterations",
    ))

    length(parameter_names) == n_parameters ||
        throw(DimensionMismatch("Saved sample dimension does not match parameter metadata"))
    retained_samples = samples[burnin+1:end, :, :]
    chains = MCMCChains.Chains(retained_samples, Symbol.(parameter_names))
    diagnostics = DataFrame(MCMCChains.summarystats(chains))
    diagnostics[!, :parameter_group] = [
        startswith(String(name), "kernel_") ? "kernel" :
        startswith(String(name), "suppression_") ? "suppression" : "ambient_weight"
        for name in diagnostics.parameters
    ]
    diagnostics[!, :n_chains] .= n_chains
    diagnostics[!, :iterations_per_chain] .= n_iterations
    diagnostics[!, :burnin] .= burnin
    diagnostics[!, :n_adapts] .= n_adapts
    diagnostics[!, :retained_per_chain] .= n_iterations - burnin

    finite_values(values) = filter(isfinite, Float64[value for value in skipmissing(values)])
    finite_rhat = finite_values(diagnostics.rhat)
    finite_bulk = finite_values(diagnostics.ess_bulk)
    finite_tail = finite_values(diagnostics.ess_tail)
    summary = DataFrame([(
        n_parameters=n_parameters,
        n_chains=n_chains,
        iterations_per_chain=n_iterations,
        burnin=burnin,
        n_adapts=n_adapts,
        retained_per_chain=n_iterations - burnin,
        max_rhat=isempty(finite_rhat) ? NaN : maximum(finite_rhat),
        n_rhat_above_1p01=count(>(1.01), finite_rhat),
        min_ess_bulk=isempty(finite_bulk) ? NaN : minimum(finite_bulk),
        min_ess_tail=isempty(finite_tail) ? NaN : minimum(finite_tail),
    )])

    CSV.write(outpath * "_chain_diagnostics.csv", diagnostics)
    CSV.write(outpath * "_chain_diagnostics_summary.csv", summary)

    samples_path = nothing
    if save_chain_samples
        samples_path = outpath * "_chain_samples.jld2"
        JLD2.jldsave(
            samples_path,
            true;
            samples=samples,
            parameter_names=parameter_names,
            burnin=burnin,
            n_adapts=n_adapts,
        )
    end

    return (
        diagnostics=diagnostics,
        diagnostics_summary=summary,
        samples_path=samples_path,
    )
end

function _nuts_statistic(stat, name::Symbol)
    hasproperty(stat, name) ||
        throw(ArgumentError("NUTS statistic is missing required field :$name"))
    return getproperty(stat, name)
end

function save_NUTS_sampler_artifacts!(
    outpath::AbstractString,
    chain_stats::Vector;
    burnin::Int,
    n_adapts::Int,
    max_tree_depth::Int=10,
)
    n_chains = length(chain_stats)
    n_chains > 0 || throw(ArgumentError("No NUTS sampler statistics were returned"))
    n_iterations = length(first(chain_stats))
    0 <= n_adapts <= burnin < n_iterations || throw(ArgumentError(
        "Require 0 <= n_adapts <= burnin < sampler-stat length; got " *
        "n_adapts=$n_adapts, burnin=$burnin, sampler-stat length=$n_iterations",
    ))
    max_tree_depth > 0 || throw(ArgumentError("max_tree_depth must be positive"))

    trace_rows = NamedTuple[]
    for chain_index in eachindex(chain_stats)
        stats = chain_stats[chain_index]
        length(stats) == n_iterations ||
            throw(DimensionMismatch("NUTS chains have unequal sampler-stat counts"))

        for iteration in eachindex(stats)
            stat = stats[iteration]
            is_adapt = Bool(_nuts_statistic(stat, :is_adapt))
            phase = is_adapt ? "adaptation" :
                iteration <= burnin ? "post_adaptation_burnin" : "retained"
            push!(trace_rows, (
                chain=chain_index,
                iteration=iteration,
                phase=phase,
                is_adapt=is_adapt,
                log_density=Float64(_nuts_statistic(stat, :log_density)),
                hamiltonian_energy=Float64(_nuts_statistic(stat, :hamiltonian_energy)),
                hamiltonian_energy_error=Float64(_nuts_statistic(stat, :hamiltonian_energy_error)),
                max_hamiltonian_energy_error=Float64(_nuts_statistic(stat, :max_hamiltonian_energy_error)),
                acceptance_rate=Float64(_nuts_statistic(stat, :acceptance_rate)),
                n_steps=Int(_nuts_statistic(stat, :n_steps)),
                tree_depth=Int(_nuts_statistic(stat, :tree_depth)),
                numerical_error=Bool(_nuts_statistic(stat, :numerical_error)),
                step_size=Float64(_nuts_statistic(stat, :step_size)),
                nominal_step_size=Float64(_nuts_statistic(stat, :nom_step_size)),
            ))
        end
    end

    trace = DataFrame(trace_rows)
    chain_rows = NamedTuple[]
    for chain_index in 1:n_chains
        chain_trace = trace[trace.chain .== chain_index, :]
        adaptation = chain_trace[chain_trace.phase .== "adaptation", :]
        extra_burnin = chain_trace[chain_trace.phase .== "post_adaptation_burnin", :]
        retained = chain_trace[chain_trace.phase .== "retained", :]
        retained_energy = retained.hamiltonian_energy
        ebfmi = length(retained_energy) > 1 ? AdvancedHMC.EBFMI(retained_energy) : NaN
        tree_depth_hits = count(>=(max_tree_depth), retained.tree_depth)

        push!(chain_rows, (
            chain=chain_index,
            iterations=nrow(chain_trace),
            adaptation_iterations=nrow(adaptation),
            post_adaptation_burnin_iterations=nrow(extra_burnin),
            retained_iterations=nrow(retained),
            adaptation_numerical_errors=count(adaptation.numerical_error),
            post_adaptation_burnin_numerical_errors=count(extra_burnin.numerical_error),
            retained_numerical_errors=count(retained.numerical_error),
            retained_max_tree_depth_hits=tree_depth_hits,
            retained_max_tree_depth_hit_rate=tree_depth_hits / nrow(retained),
            retained_max_tree_depth=maximum(retained.tree_depth),
            retained_mean_acceptance_rate=mean(retained.acceptance_rate),
            retained_min_acceptance_rate=minimum(retained.acceptance_rate),
            retained_mean_n_steps=mean(retained.n_steps),
            retained_max_n_steps=maximum(retained.n_steps),
            retained_mean_step_size=mean(retained.step_size),
            final_step_size=last(chain_trace.step_size),
            retained_ebfmi=ebfmi,
            retained_log_density_mean=mean(retained.log_density),
            retained_log_density_std=std(retained.log_density),
            retained_log_density_min=minimum(retained.log_density),
            retained_log_density_max=maximum(retained.log_density),
            retained_nonfinite_log_density=count(value -> !isfinite(value), retained.log_density),
            retained_max_abs_hamiltonian_energy_error=maximum(abs.(retained.hamiltonian_energy_error)),
        ))
    end
    diagnostics = DataFrame(chain_rows)

    retained_count = n_iterations - burnin
    log_density_array = Array{Float64}(undef, retained_count, 1, n_chains)
    for chain_index in 1:n_chains
        retained = trace[(trace.chain .== chain_index) .& (trace.phase .== "retained"), :]
        log_density_array[:, 1, chain_index] .= retained.log_density
    end

    log_density_rhat = NaN
    log_density_ess_bulk = NaN
    log_density_ess_tail = NaN
    if retained_count >= 4 && n_chains >= 2 && all(isfinite, log_density_array)
        log_density_chains = MCMCChains.Chains(log_density_array, [:log_density])
        log_density_diagnostics = DataFrame(MCMCChains.summarystats(log_density_chains))
        log_density_rhat = Float64(only(log_density_diagnostics.rhat))
        log_density_ess_bulk = Float64(only(log_density_diagnostics.ess_bulk))
        log_density_ess_tail = Float64(only(log_density_diagnostics.ess_tail))
    end

    finite_ebfmi = filter(isfinite, diagnostics.retained_ebfmi)
    summary = DataFrame([(
        n_chains=n_chains,
        iterations_per_chain=n_iterations,
        burnin=burnin,
        n_adapts=n_adapts,
        retained_per_chain=retained_count,
        max_tree_depth=max_tree_depth,
        total_adaptation_numerical_errors=sum(diagnostics.adaptation_numerical_errors),
        total_post_adaptation_burnin_numerical_errors=sum(diagnostics.post_adaptation_burnin_numerical_errors),
        total_retained_numerical_errors=sum(diagnostics.retained_numerical_errors),
        total_retained_max_tree_depth_hits=sum(diagnostics.retained_max_tree_depth_hits),
        max_retained_max_tree_depth_hit_rate=maximum(diagnostics.retained_max_tree_depth_hit_rate),
        min_retained_ebfmi=isempty(finite_ebfmi) ? NaN : minimum(finite_ebfmi),
        min_retained_mean_acceptance_rate=minimum(diagnostics.retained_mean_acceptance_rate),
        max_retained_mean_acceptance_rate=maximum(diagnostics.retained_mean_acceptance_rate),
        log_density_rhat=log_density_rhat,
        log_density_ess_bulk=log_density_ess_bulk,
        log_density_ess_tail=log_density_ess_tail,
        log_density_chain_mean_range=maximum(diagnostics.retained_log_density_mean) -
            minimum(diagnostics.retained_log_density_mean),
        total_retained_nonfinite_log_density=sum(diagnostics.retained_nonfinite_log_density),
    )])

    CSV.write(outpath * "_sampler_trace.csv", trace)
    CSV.write(outpath * "_sampler_diagnostics.csv", diagnostics)
    CSV.write(outpath * "_sampler_diagnostics_summary.csv", summary)

    return (trace=trace, diagnostics=diagnostics, diagnostics_summary=summary)
end

function save_smoothFLAVOR_chain_artifacts!(
    outpath::AbstractString,
    ambient_samples::Vector,
    sk_model::SKBDIModel;
    kwargs...,
)
    return save_smoothFLAVOR_chain_artifacts!(
        outpath,
        ambient_samples,
        _ambient_parameter_names(sk_model);
        kwargs...,
    )
end

function smoothFLAVOR_BAME(
    flavorgrid,
    outpath;
    pos_thresh=0.9,
    iters=10,
    burnin=div(iters, 4),
    n_adapts=burnin,
    kernel_stddev=4.0,
    covariance_jitter=1e-6,
    n_chains=4,
    max_tree_depth=10,
    verbosity=1,
    exports=true,
    sample_allocations=false,
    fast_reshaping=true,
    save_chain_diagnostics=false,
    save_chain_samples=false,
)
    0 <= n_adapts < iters ||
        throw(ArgumentError("n_adapts must satisfy 0 <= n_adapts < iters; got n_adapts=$n_adapts, iters=$iters"))
    n_adapts <= burnin < iters || throw(ArgumentError(
        "burnin must satisfy n_adapts <= burnin < iters; got n_adapts=$n_adapts, burnin=$burnin, iters=$iters",
    ))

    sk_model = SKBDIModel_from_FLAVOR(
        flavorgrid;
        kernel_stddev=kernel_stddev,
        covariance_jitter=covariance_jitter,
        fast_reshaping=fast_reshaping,
    )
    fubar_model = GeneralizedFUBARModel(sk_model)

    if verbosity > 0
        println("Sampling from smoothFLAVOR with NUTS.")
    end

    ambient_samples, stats = sample_NUTS(
        fubar_model,
        iters,
        n_chains;
        n_adapts=n_adapts,
        max_tree_depth=max_tree_depth,
        progress=verbosity > 0,
    )

    chain_artifacts = if save_chain_diagnostics || save_chain_samples
        posterior_artifacts = save_smoothFLAVOR_chain_artifacts!(
            outpath,
            ambient_samples,
            sk_model;
            burnin=burnin,
            n_adapts=n_adapts,
            save_chain_samples=save_chain_samples,
        )
        sampler_artifacts = save_NUTS_sampler_artifacts!(
            outpath,
            stats;
            burnin=burnin,
            n_adapts=n_adapts,
            max_tree_depth=max_tree_depth,
        )
        (posterior=posterior_artifacts, sampler=sampler_artifacts)
    else
        nothing
    end

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
        chain_artifacts=chain_artifacts,
    )
end
