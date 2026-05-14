function summarize_original_BAME_EM(
    flavorgrid;
    pos_thresh::Float64=0.9,
    method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
)
    l = size(flavorgrid.prob_matrix, 1)
    n_sites = size(flavorgrid.prob_matrix, 2)

    θ = CodonMolecularEvolution.weightEM(
        flavorgrid.prob_matrix,
        ones(l) ./ l;
        conc=method.concentration,
        iters=method.iterations,
    )

    pos_sel_mask = CodonMolecularEvolution.get_pos_sel_mask(flavorgrid)
    pos_prior = sum(pos_sel_mask .* θ)

    posterior_pos = [
        sum(MolecularEvolution.sum2one(θ .* flavorgrid.prob_matrix[:, i]) .* pos_sel_mask)
        for i in 1:n_sites
    ]

    bfs = CodonMolecularEvolution.bayes_factor.(posterior_pos, pos_prior)

    df = DataFrame(
        site=1:n_sites,
        posterior_prob_positive=posterior_pos,
        bayes_factor=bfs,
        threshold=posterior_pos .> pos_thresh,
    )

    return (
        df=df,
        θ=θ,
        posterior_probs=posterior_pos,
        bayes_factors=bfs,
        pos_prior=pos_prior,
        pos_sel_mask=pos_sel_mask,
    )
end

function run_original_BAME_EM_baseline(
    flavorgrid,
    truth_vec::Vector{Bool},
    outdir;
    pos_thresh::Float64=0.9,
    method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
)
    prefix = joinpath(outdir, "original_BAME_EM")

    summary = summarize_original_BAME_EM(
        flavorgrid;
        pos_thresh=pos_thresh,
        method=method,
    )

    site_df = copy(summary.df)
    site_df[!, :true_positive] = truth_vec
    site_df[!, :method] .= "original_BAME_EM"
    site_df[!, :kernel_stddev] .= NaN
    site_df[!, :smoothing] .= false

    roc_df, auc = roc_curve_from_scores(site_df.posterior_prob_positive, truth_vec)
    pr_df, auprc = pr_curve_from_scores(site_df.posterior_prob_positive, truth_vec)

    roc_df[!, :method] .= "original_BAME_EM"
    roc_df[!, :kernel_stddev] .= NaN
    roc_df[!, :smoothing] .= false

    pr_df[!, :method] .= "original_BAME_EM"
    pr_df[!, :kernel_stddev] .= NaN
    pr_df[!, :smoothing] .= false

    th = threshold_summary(site_df.posterior_prob_positive, truth_vec, pos_thresh)

    CSV.write(prefix * "_site_posteriors.csv", site_df)
    CSV.write(prefix * "_roc.csv", roc_df)
    CSV.write(prefix * "_pr.csv", pr_df)

    return (
        site_df=site_df,
        roc_df=roc_df,
        pr_df=pr_df,
        auc=auc,
        auprc=auprc,
        threshold_summary=th,
        pos_prior=summary.pos_prior,
        θ=summary.θ,
    )
end

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
    include_original_bame::Bool=true,
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
)
    mkpath(outdir)

    n_sites = size(getproperty(flavorgrid, :prob_matrix), 2)
    truth_vec = truth_to_bool_vector(truth, n_sites)


    summary_df = DataFrame(
        method=String[],
        kernel_stddev=Float64[],
        smoothing=Bool[],
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
    extra_results = Dict{String, NamedTuple}()

    if include_original_bame
        bame_result = run_original_BAME_EM_baseline(
            flavorgrid,
            truth_vec,
            outdir;
            pos_thresh=pos_thresh,
            method=bame_method,
        )

        th = bame_result.threshold_summary

        push!(summary_df, (
            method="original_BAME_EM",
            kernel_stddev=NaN,
            smoothing=false,
            auc=bame_result.auc,
            auprc=bame_result.auprc,
            n_sites=n_sites,
            n_true_positive_sites=count(truth_vec),
            n_called_at_pos_thresh=th.n_called,
            pos_thresh=pos_thresh,
            tpr_at_pos_thresh=th.tpr,
            fpr_at_pos_thresh=th.fpr,
            precision_at_pos_thresh=th.precision,
            posterior_mean_positive_prior_mass=bame_result.pos_prior,
        ))

        extra_results["original_BAME_EM"] = bame_result

        CSV.write(joinpath(outdir, "kernel_stddev_study_summary.csv"), summary_df)
    end


    for (j, σk) in enumerate(Float64.(kernel_stddevs))
        method_name = σk == 0.0 ? "smoothFLAVOR_BAME_no_smoothing" : "smoothFLAVOR_BAME"
        smoothing_flag = σk > 0.0

        site_df[!, :method] .= method_name
        site_df[!, :smoothing] .= smoothing_flag

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
            kernel_stddev=max(σk, 0.0),
            smooth=σk > 0.0,
        )

        site_df = copy(df)
        site_df[!, :true_positive] = truth_vec
        site_df[!, :kernel_stddev] .= σk

        roc_df, auc = roc_curve_from_scores(site_df.posterior_prob_positive, truth_vec)
        pr_df, auprc = pr_curve_from_scores(site_df.posterior_prob_positive, truth_vec)
        roc_df[!, :kernel_stddev] .= σk
        pr_df[!, :kernel_stddev] .= σk

        roc_df[!, :method] .= method_name
        roc_df[!, :smoothing] .= smoothing_flag

        pr_df[!, :method] .= method_name
        pr_df[!, :smoothing] .= smoothing_flag

        th = threshold_summary(site_df.posterior_prob_positive, truth_vec, pos_thresh)

        CSV.write(prefix * "_site_posteriors.csv", site_df)
        CSV.write(prefix * "_roc.csv", roc_df)
        CSV.write(prefix * "_pr.csv", pr_df)

        push!(summary_df, (
            method=method_name,
            kernel_stddev=σk,
            smoothing=smoothing_flag,
            auc=auc,
            auprc=auprc,
            n_sites=n_sites,
            n_true_positive_sites=count(truth_vec),
            n_called_at_pos_thresh=th.n_called,
            pos_thresh=pos_thresh,
            tpr_at_pos_thresh=th.tpr,
            fpr_at_pos_thresh=th.fpr,
            precision_at_pos_thresh=th.precision,
            posterior_mean_positive_prior_mass=results.pos_prior,
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

    return summary_df, per_stddev, extra_results
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

    summary_df, per_stddev, extra_results = smoothFLAVOR_kernel_stddev_study(
        flavorgrid,
        truth,
        outdir;
        study_kwargs...,
    )

    return flavorgrid, summary_df, per_stddev, extra_results
end
