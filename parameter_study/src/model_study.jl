function _existing_summary(path::AbstractString)
    return isfile(path) ? DataFrame(CSV.File(path)) : DataFrame()
end

function _summary_has_method(summary_df::DataFrame, method::AbstractString; kernel_stddev=nothing)
    nrow(summary_df) == 0 && return false
    nms = Symbol.(names(summary_df))
    (:method in nms) || return false

    method_col = String.(summary_df[!, :method])
    matches = method_col .== method

    if kernel_stddev !== nothing
        (:kernel_stddev in nms) || return false
        matches .&= Float64.(summary_df[!, :kernel_stddev]) .== Float64(kernel_stddev)
    end

    return any(matches)
end

function _write_summary(path::AbstractString, rows::Vector{NamedTuple})
    mkpath(dirname(path))
    df = isempty(rows) ? DataFrame() : DataFrame(rows)
    CSV.write(path, df)
    return df
end

function _load_summary_rows(summary_path::AbstractString)
    if !isfile(summary_path)
        return NamedTuple[]
    end
    df = DataFrame(CSV.File(summary_path))
    return [NamedTuple(row) for row in eachrow(df)]
end

function _extract_smooth_posterior(df::DataFrame)
    nms = Symbol.(names(df))
    if :posterior_prob_positive in nms
        return Float64.(df[!, :posterior_prob_positive])
    elseif "posterior_prob_positive" in names(df)
        return Float64.(df[!, "posterior_prob_positive"])
    elseif "P(β>α)" in names(df)
        return Float64.(df[!, "P(β>α)"])
    else
        error("Could not find posterior probability column in smoothFLAVOR output. Columns were: $(names(df))")
    end
end

function _extract_bayes_factor(df::DataFrame)
    nms = Symbol.(names(df))
    if :bayes_factor in nms
        return Float64.(df[!, :bayes_factor])
    elseif "BayesFactor" in names(df)
        return Float64.(df[!, "BayesFactor"])
    elseif "bayes_factor" in names(df)
        return Float64.(df[!, "bayes_factor"])
    else
        return fill(NaN, nrow(df))
    end
end

function run_existing_BAME_baseline!(
    flavorgrid,
    truth_vec::Vector{Bool},
    outdir::AbstractString;
    pos_thresh::Float64=0.9,
    verbosity::Int=1,
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
)
    prefix = joinpath(outdir, "original_BAME")

    raw_df = CodonMolecularEvolution.BAME(
        flavorgrid,
        prefix;
        pos_thresh=pos_thresh,
        verbosity=verbosity,
        method=bame_method,
        plots=false,
    )

    posterior = Float64.(raw_df[!, "P(β>α)"])
    bayes_factors = Float64.(raw_df[!, "BayesFactor"])

    site_df = DataFrame(
        site = Int.(raw_df[!, "site"]),
        posterior_prob_positive = posterior,
        bayes_factor = bayes_factors,
        threshold = posterior .>= pos_thresh,
        true_positive = truth_vec,
        method = fill("original_BAME", length(posterior)),
        kernel_stddev = fill(NaN, length(posterior)),
    )

    roc_df, auc = roc_curve_from_scores(site_df.posterior_prob_positive, truth_vec)
    pr_df, auprc = pr_curve_from_scores(site_df.posterior_prob_positive, truth_vec)

    roc_df[!, :method] .= "original_BAME"
    roc_df[!, :kernel_stddev] .= NaN

    pr_df[!, :method] .= "original_BAME"
    pr_df[!, :kernel_stddev] .= NaN

    th = threshold_summary(site_df.posterior_prob_positive, truth_vec, pos_thresh)

    CSV.write(prefix * "_site_posteriors.csv", site_df)
    CSV.write(prefix * "_roc.csv", roc_df)
    CSV.write(prefix * "_pr.csv", pr_df)

    return (
        method = "original_BAME",
        kernel_stddev = NaN,
        auc = auc,
        auprc = auprc,
        n_sites = length(truth_vec),
        n_true_positive_sites = count(truth_vec),
        n_called_at_pos_thresh = th.n_called,
        pos_thresh = pos_thresh,
        tpr_at_pos_thresh = th.tpr,
        fpr_at_pos_thresh = th.fpr,
        precision_at_pos_thresh = th.precision,
    )
end

function run_smoothFLAVOR_for_kernel_stddev!(
    flavorgrid,
    truth_vec::Vector{Bool},
    outdir::AbstractString,
    kernel_stddev::Real;
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    verbosity::Int=1,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
)
    σk = Float64(kernel_stddev)
    σk > 0 || throw(ArgumentError("kernel_stddev must be strictly positive; no-smoothing runs are intentionally excluded."))

    prefix = joinpath(outdir, "kernel_stddev_$(slugify_real(σk))")

    df, results = CodonMolecularEvolution.smoothFLAVOR_BAME(
        flavorgrid,
        prefix;
        pos_thresh=pos_thresh,
        iters=iters,
        burnin=burnin,
        n_chains=n_chains,
        verbosity=verbosity,
        exports=false,
        sample_allocations=sample_allocations,
        fast_reshaping=fast_reshaping,
        kernel_stddev=σk,
    )

    posterior = _extract_smooth_posterior(df)
    bayes_factors = _extract_bayes_factor(df)

    site_df = copy(df)
    if !(:site in Symbol.(names(site_df)))
        site_df[!, :site] = collect(1:nrow(site_df))
    end
    site_df[!, :posterior_prob_positive] = posterior
    site_df[!, :bayes_factor] = bayes_factors
    site_df[!, :threshold] = posterior .>= pos_thresh
    site_df[!, :true_positive] = truth_vec
    site_df[!, :method] .= "smoothFLAVOR_BAME"
    site_df[!, :kernel_stddev] .= σk

    roc_df, auc = roc_curve_from_scores(site_df.posterior_prob_positive, truth_vec)
    pr_df, auprc = pr_curve_from_scores(site_df.posterior_prob_positive, truth_vec)

    roc_df[!, :method] .= "smoothFLAVOR_BAME"
    roc_df[!, :kernel_stddev] .= σk

    pr_df[!, :method] .= "smoothFLAVOR_BAME"
    pr_df[!, :kernel_stddev] .= σk

    th = threshold_summary(site_df.posterior_prob_positive, truth_vec, pos_thresh)

    CSV.write(prefix * "_site_posteriors.csv", site_df)
    CSV.write(prefix * "_roc.csv", roc_df)
    CSV.write(prefix * "_pr.csv", pr_df)

    return (
        method = "smoothFLAVOR_BAME",
        kernel_stddev = σk,
        auc = auc,
        auprc = auprc,
        n_sites = length(truth_vec),
        n_true_positive_sites = count(truth_vec),
        n_called_at_pos_thresh = th.n_called,
        pos_thresh = pos_thresh,
        tpr_at_pos_thresh = th.tpr,
        fpr_at_pos_thresh = th.fpr,
        precision_at_pos_thresh = th.precision,
    )
end

function run_parameter_sweep_on_flavorgrid!(
    flavorgrid,
    truth,
    outdir::AbstractString;
    kernel_stddevs=Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    include_original_bame::Bool=true,
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    verbosity::Int=1,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    base_seed::Union{Nothing,Int}=nothing,
    skip_completed::Bool=true,
)
    mkpath(outdir)

    any(σ -> Float64(σ) <= 0, kernel_stddevs) &&
        throw(ArgumentError("All kernel_stddevs must be strictly positive. Remove 0.0/no-smoothing values."))

    n_sites = size(getproperty(flavorgrid, :prob_matrix), 2)
    truth_vec = truth_to_bool_vector(truth, n_sites)

    summary_path = joinpath(outdir, "method_sweep_summary.csv")
    summary_df = _existing_summary(summary_path)
    summary_rows = _load_summary_rows(summary_path)

    if include_original_bame
        already_done = skip_completed && _summary_has_method(summary_df, "original_BAME")
        if !already_done
            verbosity > 0 && println("Running original BAME baseline.")
            elapsed = @elapsed bame_summary = run_existing_BAME_baseline!(
                flavorgrid,
                truth_vec,
                outdir;
                pos_thresh=pos_thresh,
                verbosity=verbosity,
                bame_method=bame_method,
            )
            bame_summary = merge(bame_summary, (elapsed_seconds=elapsed,))
            push!(summary_rows, bame_summary)
            summary_df = _write_summary(summary_path, summary_rows)
        else
            verbosity > 0 && println("Skipping original BAME baseline; summary row already exists.")
        end
    end

    for (j, σ) in enumerate(Float64.(kernel_stddevs))
        already_done = skip_completed && _summary_has_method(summary_df, "smoothFLAVOR_BAME"; kernel_stddev=σ)
        if already_done
            verbosity > 0 && println("Skipping smoothFLAVOR_BAME kernel_stddev=$σ; summary row already exists.")
            continue
        end

        if base_seed !== nothing
            Random.seed!(base_seed + j - 1)
        end

        verbosity > 0 && println("Running smoothFLAVOR_BAME with kernel_stddev=$σ.")
        elapsed = @elapsed smooth_summary = run_smoothFLAVOR_for_kernel_stddev!(
            flavorgrid,
            truth_vec,
            outdir,
            σ;
            pos_thresh=pos_thresh,
            iters=iters,
            burnin=burnin,
            n_chains=n_chains,
            verbosity=verbosity,
            fast_reshaping=fast_reshaping,
            sample_allocations=sample_allocations,
        )

        smooth_summary = merge(smooth_summary, (elapsed_seconds=elapsed,))
        push!(summary_rows, smooth_summary)
        summary_df = _write_summary(summary_path, summary_rows)
    end

    return summary_df
end
