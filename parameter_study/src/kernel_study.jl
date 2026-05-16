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
        auprc=Float64[],
        n_sites=Int[],
        n_true_positive_sites=Int[],
        n_called_at_pos_thresh=Int[],
        pos_thresh=Float64[],
        tpr_at_pos_thresh=Float64[],
        fpr_at_pos_thresh=Float64[],
        precision_at_pos_thresh=Float64[],
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
        pr_df, auprc = pr_curve_from_scores(site_df.posterior_prob_positive, truth_vec)
        roc_df[!, :kernel_stddev] .= σk
        pr_df[!, :kernel_stddev] .= σk

        th = threshold_summary(site_df.posterior_prob_positive, truth_vec, pos_thresh)

        CSV.write(prefix * "_site_posteriors.csv", site_df)
        CSV.write(prefix * "_roc.csv", roc_df)
        CSV.write(prefix * "_pr.csv", pr_df)

        push!(summary_df, (
            σk,
            auc,
            auprc,
            n_sites,
            count(truth_vec),
            th.n_called,
            pos_thresh,
            th.tpr,
            th.fpr,
            th.precision,
            results.pos_prior,
        ))

        per_stddev[σk] = (
            site_df=site_df,
            roc_df=roc_df,
            pr_df=pr_df,
            auc=auc,
            auprc=auprc,
            posterior_results=results,
        )

        CSV.write(joinpath(outdir, "kernel_stddev_study_summary.csv"), summary_df)
    end

    return summary_df, per_stddev
end

function run_smoothFLAVOR_kernel_stddev_study(
    seqnames,
    seqs,
    treestring,
    truth,
    outdir;
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    flavorgrid_kwargs=NamedTuple(),
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
