using CodonMolecularEvolution
using DataFrames
using CSV
using Random

"""
    truth_to_bool_vector(truth, n_sites)

Convert ground-truth site labels to a Bool vector of length `n_sites`.
Accepted inputs:
- Bool vector of length `n_sites`
- vector of 1-based site indices that are true positives
"""
function truth_to_bool_vector(truth, n_sites::Int)
    if truth isa AbstractVector{Bool}
        length(truth) == n_sites || throw(DimensionMismatch("truth has length $(length(truth)); expected $(n_sites)"))
        return collect(truth)
    elseif truth isa AbstractVector{<:Integer}
        y = falses(n_sites)
        for i in truth
            1 <= i <= n_sites || throw(BoundsError(y, i))
            y[i] = true
        end
        return y
    else
        throw(ArgumentError("truth must be a Bool vector or a vector of site indices"))
    end
end

"""
    roc_curve_from_scores(scores, truth)

Compute an empirical ROC curve and trapezoidal AUC from site-level scores.
"""
function roc_curve_from_scores(scores::AbstractVector{<:Real}, truth)
    n = length(scores)
    y = truth_to_bool_vector(truth, n)

    thresholds = vcat(Inf, sort(unique(Float64.(scores)); rev=true), -Inf)
    n_pos = count(y)
    n_neg = n - n_pos
    n_pos > 0 || throw(ArgumentError("ROC is undefined with zero positive sites"))
    n_neg > 0 || throw(ArgumentError("ROC is undefined with zero negative sites"))

    threshold_vals = Float64[]
    tpr_vals = Float64[]
    fpr_vals = Float64[]
    tp_vals = Int[]
    fp_vals = Int[]
    tn_vals = Int[]
    fn_vals = Int[]

    s = Float64.(scores)
    for thr in thresholds
        pred = s .>= thr
        tp = count(pred .& y)
        fp = count(pred .& .!y)
        fn = n_pos - tp
        tn = n_neg - fp

        push!(threshold_vals, thr)
        push!(tpr_vals, tp / n_pos)
        push!(fpr_vals, fp / n_neg)
        push!(tp_vals, tp)
        push!(fp_vals, fp)
        push!(tn_vals, tn)
        push!(fn_vals, fn)
    end

    roc_df = DataFrame(
        threshold=threshold_vals,
        tpr=tpr_vals,
        fpr=fpr_vals,
        tp=tp_vals,
        fp=fp_vals,
        tn=tn_vals,
        fn=fn_vals,
    )

    order = sortperm(roc_df.fpr)
    x = roc_df.fpr[order]
    yroc = roc_df.tpr[order]
    auc = sum((x[2:end] .- x[1:end-1]) .* (yroc[2:end] .+ yroc[1:end-1]) ./ 2)

    return roc_df, auc
end

function slugify_real(x::Real)
    s = replace(string(round(Float64(x), digits=6)), '-' => "m", '.' => "p")
    return replace(s, r"[^A-Za-z0-9_]" => "_")
end

"""
    smoothFLAVOR_kernel_stddev_study(flavorgrid, truth, outdir; kwargs...)

Run a parameter study over `kernel_stddevs`, using the updated
`CodonMolecularEvolution.smoothFLAVOR_BAME(...; kernel_stddev=...)` interface.
For each setting, this function:
- runs smoothFLAVOR_BAME,
- stores the site-level posterior probabilities,
- computes an ROC curve and AUC for `posterior_prob_positive`,
- writes CSV outputs to `outdir`.

Arguments
---------
- `flavorgrid`: precomputed FLAVOR grid for one alignment/tree.
- `truth`: either a Bool vector of length `n_sites` or a vector of 1-based site indices.
- `outdir`: directory where CSV outputs are written.

Keyword arguments
-----------------
- `kernel_stddevs`: values to sweep over.
- `iters`, `burnin`, `n_chains`, `pos_thresh`, `verbosity`, `sample_allocations`,
  `fast_reshaping`: forwarded to `smoothFLAVOR_BAME`.
- `base_seed`: if provided, seeds each run as `base_seed + j - 1` for reproducibility.

Returns
-------
`summary_df, per_stddev`
where `summary_df` is one row per kernel_stddev and `per_stddev` stores detailed outputs.
"""
function smoothFLAVOR_kernel_stddev_study(
    flavorgrid,
    truth,
    outdir;
    kernel_stddevs=Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    pos_thresh::Float64=0.9,
    verbosity::Int=1,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    base_seed::Union{Nothing,Int}=nothing,
)
    mkpath(outdir)

    n_sites = size(getproperty(flavorgrid, :prob_matrix), 2)
    truth_vec = truth_to_bool_vector(truth, n_sites)

    summary_df = DataFrame(
        kernel_stddev=Float64[],
        auc=Float64[],
        n_sites=Int[],
        n_true_positive_sites=Int[],
        n_called_at_pos_thresh=Int[],
        pos_thresh=Float64[],
        tpr_at_pos_thresh=Float64[],
        fpr_at_pos_thresh=Float64[],
        posterior_mean_positive_prior_mass=Float64[],
    )

    per_stddev = Dict{Float64, NamedTuple}()

    for (j, σk) in enumerate(Float64.(kernel_stddevs))
        σk > 0 || throw(ArgumentError("All kernel_stddev values must be > 0"))

        if base_seed !== nothing
            Random.seed!(base_seed + j - 1)
        end

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

        site_df = copy(df)
        site_df[!, :true_positive] = truth_vec
        site_df[!, :kernel_stddev] .= σk

        roc_df, auc = roc_curve_from_scores(site_df.posterior_prob_positive, truth_vec)
        roc_df[!, :kernel_stddev] .= σk

        pred = site_df.posterior_prob_positive .>= pos_thresh
        tp = count(pred .& truth_vec)
        fp = count(pred .& .!truth_vec)
        n_pos = count(truth_vec)
        n_neg = length(truth_vec) - n_pos
        tpr_at_pos_thresh = n_pos > 0 ? tp / n_pos : NaN
        fpr_at_pos_thresh = n_neg > 0 ? fp / n_neg : NaN

        CSV.write(prefix * "_site_posteriors.csv", site_df)
        CSV.write(prefix * "_roc.csv", roc_df)

        push!(summary_df, (
            σk,
            auc,
            n_sites,
            n_pos,
            count(pred),
            pos_thresh,
            tpr_at_pos_thresh,
            fpr_at_pos_thresh,
            results.pos_prior,
        ))

        per_stddev[σk] = (
            site_df=site_df,
            roc_df=roc_df,
            auc=auc,
            posterior_results=results,
        )
    end

    CSV.write(joinpath(outdir, "kernel_stddev_study_summary.csv"), summary_df)

    return summary_df, per_stddev
end

"""
    run_smoothFLAVOR_kernel_stddev_study(seqnames, seqs, treestring, truth, outdir; kwargs...)

Convenience wrapper: build the `FLAVORgrid` once, then run the
`kernel_stddev` parameter study on it.
"""
function run_smoothFLAVOR_kernel_stddev_study(
    seqnames,
    seqs,
    treestring,
    truth,
    outdir;
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    flavorgrid_kwargs...,
    study_kwargs...,
)
    mkpath(outdir)

    flavorgrid = CodonMolecularEvolution.FLAVORgrid(
        seqnames,
        seqs,
        treestring;
        verbosity=flavorgrid_verbosity,
        optimize_branch_lengths=optimize_branch_lengths,
        flavorgrid_kwargs...,
    )

    summary_df, per_stddev = smoothFLAVOR_kernel_stddev_study(
        flavorgrid,
        truth,
        outdir;
        study_kwargs...,
    )

    return flavorgrid, summary_df, per_stddev
end
