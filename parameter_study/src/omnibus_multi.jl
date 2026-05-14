function _find_true_rates_for_sim_rep(truth_files::Vector{String}, sim::Integer, rep::Integer)
    isempty(truth_files) && return nothing

    sim_rep_patterns = [
        Regex("sims[._-]$(sim)[._-].*replicate[._-]$(rep)(\\D|\$)", "i"),
        Regex("sim[._-]$(sim)[._-].*replicate[._-]$(rep)(\\D|\$)", "i"),
        Regex("sim[._-]$(sim)[._-].*rep[._-]$(rep)(\\D|\$)", "i"),
    ]

    for rx in sim_rep_patterns
        matches = filter(p -> occursin(rx, basename(p)), truth_files)
        length(matches) == 1 && return only(matches)
    end

    sim_patterns = [
        Regex("sims[._-]$(sim)(\\D|\$)", "i"),
        Regex("sim[._-]$(sim)(\\D|\$)", "i"),
    ]

    for rx in sim_patterns
        matches = filter(p -> occursin(rx, basename(p)), truth_files)

        if length(matches) == 1
            return only(matches)
        elseif length(matches) > 1
            rep_matches = filter(
                p -> occursin(Regex("replicate[._-]$(rep)(\\D|\$)", "i"), basename(p)) ||
                     occursin(Regex("rep[._-]$(rep)(\\D|\$)", "i"), basename(p)),
                matches,
            )
            length(rep_matches) == 1 && return only(rep_matches)
        end
    end

    length(truth_files) == 1 && return only(truth_files)

    return nothing
end

function _discover_omnibus_multi_from_named_files(all_files::Vector{String}; min_match_score::Float64=0.0)
    tree_rx = r"^sims\.(\d+)\.nwk$"i
    settings_rx = r"^sims\.(\d+)\.settings$"i
    rep_rx = r"^sims\.(\d+)\.settings\.replicate\.(\d+)$"i

    tree_map = Dict{Int,String}()
    settings_map = Dict{Int,String}()
    replicate_rows = NamedTuple[]

    true_rates_npz = filter(
        p -> lowercase(basename(p)) == "omnibus_multi_true_rates.npz",
        all_files,
    )

    true_rates_meta = filter(
        p -> lowercase(basename(p)) == "omnibus_multi_true_rates.meta.json",
        all_files,
    )

    isempty(true_rates_npz) && error("Could not find omnibus_multi_true_rates.npz")
    length(true_rates_npz) == 1 || error("Found multiple omnibus_multi_true_rates.npz files")

    true_rates_path = abspath(only(true_rates_npz))
    true_rates_meta_path = isempty(true_rates_meta) ? "" : abspath(only(true_rates_meta))

    for p in all_files
        b = basename(p)

        occursin(r"\.FEL\.json$"i, b) && continue

        mt = match(tree_rx, b)
        if mt !== nothing
            tree_map[parse(Int, mt.captures[1])] = abspath(p)
            continue
        end

        ms = match(settings_rx, b)
        if ms !== nothing
            settings_map[parse(Int, ms.captures[1])] = abspath(p)
            continue
        end

        mr = match(rep_rx, b)
        if mr !== nothing
            sim = parse(Int, mr.captures[1])
            rep = parse(Int, mr.captures[2])
            push!(replicate_rows, (
                sim=sim,
                replicate=rep,
                alignment_path=abspath(p),
            ))
        end
    end

    isempty(replicate_rows) && return DataFrame()
    isempty(tree_map) && error("No tree files matching sims.<sim>.nwk were found")

    rows = NamedTuple[]
    rep_df = sort!(DataFrame(replicate_rows), [:sim, :replicate])

    for row in eachrow(rep_df)
        haskey(tree_map, row.sim) || continue

        settings_path = get(settings_map, row.sim, "")

        push!(rows, (
            simulation_id="sim_$(row.sim)_replicate_$(row.replicate)",
            alignment_path=String(row.alignment_path),
            tree_path=tree_map[row.sim],
            true_rates_path=true_rates_path,
            true_rates_meta_path=true_rates_meta_path,
            settings_path=settings_path,
            truth_source="omnibus_multi_true_rates_npz",
            tree_match_score=1.0,
            truth_match_score=1.0,
            sim=row.sim,
            replicate=row.replicate,
        ))
    end

    return isempty(rows) ? DataFrame() : DataFrame(rows)
end

function _discover_omnibus_multi_generic(all_files::Vector{String}; min_match_score::Float64=0.2)
    aln_files = filter(p -> occursin(r"\.(fa|fna|fas|fasta)$"i, p), all_files)
    tree_files = filter(p -> occursin(r"\.(nwk|newick|tre|tree|txt)$"i, p), all_files)
    truth_files = filter(
        p -> occursin(r"true[_-]?rates"i, basename(p)) &&
             occursin(r"\.(csv|tsv|txt)$"i, p),
        all_files,
    )

    isempty(aln_files) && return DataFrame()
    isempty(tree_files) && error("No tree files were found under the provided omnibus directory")
    isempty(truth_files) && error("No omnibus_multi_true_rates files were found under the provided omnibus directory")

    rows = NamedTuple[]

    for aln in sort(aln_files)
        tree, tree_score = _best_match(aln, tree_files; min_score=min_match_score)
        truth, truth_score = _best_match(aln, truth_files; min_score=min_match_score)

        (tree === nothing || truth === nothing) && continue

        sim_id = _slugify_string(splitext(basename(aln))[1])

        push!(rows, (
            simulation_id=sim_id,
            alignment_path=abspath(aln),
            tree_path=abspath(tree),
            true_rates_path=abspath(truth),
            tree_match_score=tree_score,
            truth_match_score=truth_score,
        ))
    end

    return isempty(rows) ? DataFrame() : unique(DataFrame(rows), :simulation_id)
end

function discover_omnibus_multi_simulations(rootdir::AbstractString; min_match_score::Float64=0.2)
    all_files = String[]

    for (dir, _, files) in walkdir(rootdir)
        for f in files
            push!(all_files, joinpath(dir, f))
        end
    end

    isempty(all_files) && error("No files were found under $rootdir")

    println("Discovery root: ", abspath(rootdir))
    println("Number of files found: ", length(all_files))
    println("Example files:")
    for f in first(sort(all_files), min(20, length(all_files)))
        println("  ", basename(f))
    end
    manifest_df = _discover_omnibus_multi_from_named_files(all_files; min_match_score=min_match_score)
    println("Manifest columns: ", names(manifest_df))
    println("Manifest rows: ", nrow(manifest_df))
    if nrow(manifest_df) > 0
        println(first(manifest_df, min(5, nrow(manifest_df))))
    end
    if nrow(manifest_df) == 0
        manifest_df = _discover_omnibus_multi_generic(all_files; min_match_score=min_match_score)
    end

    nrow(manifest_df) > 0 || error(
        "Could not discover omnibus-multi simulations under $rootdir. " *
        "Expected files like sims.<sim>.nwk and sims.<sim>.settings.replicate.<rep>, " *
        "plus omnibus_multi_true_rates files."
    )

    sortcols = Symbol[s for s in (:simulation_id, :sim, :replicate, :alignment_path) if s in names(manifest_df)]
    !isempty(sortcols) && sort!(manifest_df, sortcols)

    return manifest_df
end

function _resolve_manifest(rootdir::AbstractString; manifest=nothing)
    if manifest === nothing
        return discover_omnibus_multi_simulations(rootdir)
    elseif manifest isa AbstractString
        return DataFrame(CSV.File(manifest))
    elseif manifest isa DataFrame
        return copy(manifest)
    else
        throw(ArgumentError("manifest must be nothing, a CSV path, or a DataFrame"))
    end
end

function select_one_omnibus_multi_simulation(
    rootdir::AbstractString;
    manifest=nothing,
    simulation_id::Union{Nothing,AbstractString}=nothing,
    simulation_index::Int=1,
    random_choice::Bool=false,
    rng_seed::Union{Nothing,Int}=nothing,
)
    manifest_df = _resolve_manifest(rootdir; manifest=manifest)

    required = [:simulation_id, :alignment_path, :tree_path, :true_rates_path]
    manifest_names = Set(Symbol.(names(manifest_df)))

    all(req -> req in manifest_names, required) ||
    throw(ArgumentError("Manifest must contain columns $(required). Found columns: $(names(manifest_df))"))

    nrow(manifest_df) > 0 || error("No omnibus-multi simulations were found")

    row_ix = if simulation_id !== nothing
        matches = findall(manifest_df.simulation_id .== simulation_id)
        isempty(matches) && error("simulation_id='$(simulation_id)' was not found in the manifest")
        length(matches) == 1 || error("simulation_id='$(simulation_id)' matched multiple rows")
        only(matches)
    elseif random_choice
        rng = rng_seed === nothing ? Random.default_rng() : MersenneTwister(rng_seed)
        rand(rng, 1:nrow(manifest_df))
    else
        1 <= simulation_index <= nrow(manifest_df) ||
            throw(BoundsError(1:nrow(manifest_df), simulation_index))
        simulation_index
    end

    return manifest_df[row_ix, :], manifest_df
end

function recompute_aggregate_outputs!(outdir::AbstractString; save_all_site_scores::Bool=true)
    agg_outdir = joinpath(outdir, "aggregate")
    mkpath(agg_outdir)

    simulation_summary_path = joinpath(outdir, "omnibus_multi_simulation_summary.csv")
    all_site_scores_path = joinpath(outdir, "omnibus_multi_all_site_scores.csv")
    aggregate_summary_path = joinpath(outdir, "omnibus_multi_aggregate_summary.csv")

    if !(isfile(simulation_summary_path) && save_all_site_scores && isfile(all_site_scores_path))
        CSV.write(aggregate_summary_path, DataFrame(
            kernel_stddev=Float64[],
            auc_micro=Float64[],
            auprc_micro=Float64[],
            auc_macro_mean=Float64[],
            auprc_macro_mean=Float64[],
            n_sites_total=Int[],
            n_positive_total=Int[],
            truth_prevalence=Float64[],
            n_simulations=Int[],
        ))
        return DataFrame()
    end

    simulation_summary_df = DataFrame(CSV.File(simulation_summary_path))
    all_site_scores_df = DataFrame(CSV.File(all_site_scores_path))

    aggregate_summary_df = DataFrame(
        kernel_stddev=Float64[],
        auc_micro=Float64[],
        auprc_micro=Float64[],
        auc_macro_mean=Float64[],
        auprc_macro_mean=Float64[],
        n_sites_total=Int[],
        n_positive_total=Int[],
        truth_prevalence=Float64[],
        n_simulations=Int[],
    )

    for σk in sort(unique(Float64.(all_site_scores_df.kernel_stddev)))
        sub = all_site_scores_df[Float64.(all_site_scores_df.kernel_stddev) .== σk, :]
        roc_df, auc = roc_curve_from_scores(sub.posterior_prob_positive, sub.true_positive)
        pr_df, auprc = pr_curve_from_scores(sub.posterior_prob_positive, sub.true_positive)
        roc_df[!, :kernel_stddev] .= σk
        pr_df[!, :kernel_stddev] .= σk

        prefix = joinpath(agg_outdir, "kernel_stddev_$(slugify_real(σk))")
        CSV.write(prefix * "_aggregate_roc.csv", roc_df)
        CSV.write(prefix * "_aggregate_pr.csv", pr_df)

        sim_sub = simulation_summary_df[Float64.(simulation_summary_df.kernel_stddev) .== σk, :]
        push!(aggregate_summary_df, (
            σk,
            auc,
            auprc,
            mean(Float64.(sim_sub.auc)),
            mean(Float64.(sim_sub.auprc)),
            nrow(sub),
            count(sub.true_positive),
            mean(Bool.(sub.true_positive)),
            length(unique(String.(sub.simulation_id))),
        ))
    end

    CSV.write(aggregate_summary_path, aggregate_summary_df)
    return aggregate_summary_df
end

_seed_for_run(base_seed::Union{Nothing,Int}, sim_index::Int, sigma_index::Int, n_sigmas::Int) =
    base_seed === nothing ? nothing : base_seed + (sim_index - 1) * n_sigmas + sigma_index - 1

function strip_tree_group_tags_for_flavor(treestring::AbstractString)
    # Remove branch-set tags used by Contrast-FEL / omnibus simulations,
    # e.g. N66{GROUP_0} -> N66.
    #
    # This is only for the temporary tree string passed to FLAVORgrid.
    # Do not overwrite the original .nwk file.
    return replace(String(treestring), r"\{GROUP_[0-9A-Za-z_:-]*\}" => "")
end

function run_one_simulation!(
    row,
    sim_index::Int,
    mode::AbstractString,
    outdir::AbstractString;
    kernel_stddevs,
    base_seed::Union{Nothing,Int}=nothing,
    save_all_site_scores::Bool=true,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    skip_completed::Bool=true,
    continue_on_error::Bool=true,
    update_aggregate_each_simulation::Bool=true,
    flavorgrid_kwargs...,
)
    progress_path = joinpath(outdir, "progress.csv")
    global_summary_path = joinpath(outdir, "omnibus_multi_simulation_summary.csv")
    global_site_scores_path = joinpath(outdir, "omnibus_multi_all_site_scores.csv")
    simulation_runs_path = joinpath(outdir, "simulation_runs.csv")

    sim_id = _slugify_string(String(row.simulation_id))
    sim_outdir = joinpath(outdir, "simulations", sim_id)
    mkpath(sim_outdir)

    completed_keys = skip_completed ? collect_completed_keys(progress_path) : Set{Tuple{String,String}}()
    sigma_keys = [slugify_real(Float64(σk)) for σk in kernel_stddevs]
    pending_mask = [!((sim_id, sigma_key) in completed_keys) for sigma_key in sigma_keys]
    if !any(pending_mask)
        append_csv_row(simulation_runs_path, (
            timestamp=read_timestamp(),
            mode=mode,
            simulation_id=sim_id,
            simulation_index=sim_index,
            status="skipped_all_completed",
            n_kernel_stddev_total=length(kernel_stddevs),
            n_kernel_stddev_completed=length(kernel_stddevs),
            n_kernel_stddev_failed=0,
            flavorgrid_seconds=0.0,
            total_simulation_seconds=0.0,
            alignment_path=String(row.alignment_path),
            tree_path=String(row.tree_path),
            true_rates_path=String(row.true_rates_path),
            error_message="",
        ))
        return (simulation_id=sim_id, status="skipped_all_completed", flavorgrid_seconds=0.0, total_simulation_seconds=0.0, n_kernel_stddev_completed=length(kernel_stddevs), n_kernel_stddev_failed=0)
    end

    seqnames, seqs = read_fasta_simple(String(row.alignment_path))
    validate_alignment_for_flavor(seqnames, seqs; source=String(row.alignment_path))
    println("Read ", length(seqnames), " sequences")
    println("Alignment nucleotide length: ", length(first(seqs)))
    println("Alignment codon length: ", div(length(first(seqs)), 3))
    println("First names: ", first(seqnames, min(5, length(seqnames))))
    raw_treestring = read(String(row.tree_path), String)
    treestring = strip_tree_group_tags_for_flavor(raw_treestring)
    truth_vec, truth_df = load_omnibus_multi_truth(
    String(row.true_rates_path);
    sim = (:sim in propertynames(row)) ? Int(row.sim) : nothing,
    replicate = (:replicate in propertynames(row)) ? Int(row.replicate) : nothing,
    meta_path = (:true_rates_meta_path in propertynames(row)) ? String(row.true_rates_meta_path) : nothing,
    )
    CSV.write(joinpath(sim_outdir, "true_rates_normalized.csv"), truth_df)

    sim_start = time()
    fg_timed = @timed CodonMolecularEvolution.FLAVORgrid(
        seqnames,
        seqs,
        treestring;
        verbosity=flavorgrid_verbosity,
        optimize_branch_lengths=optimize_branch_lengths,
        flavorgrid_kwargs...,
    )
    flavorgrid = fg_timed.value
    flavorgrid_seconds = fg_timed.time

    n_sites = size(getproperty(flavorgrid, :prob_matrix), 2)
    n_sequences = length(seqnames)
    n_pos = count(truth_vec)
    truth_prevalence = mean(truth_vec)

    failed_count = 0
    completed_count = 0

    for (sigma_index, σk_raw) in enumerate(Float64.(kernel_stddevs))
        σk = Float64(σk_raw)
        sigma_key = slugify_real(σk)
        skip_completed && ((sim_id, sigma_key) in completed_keys) && continue

        sigma_seed = _seed_for_run(base_seed, sim_index, sigma_index, length(kernel_stddevs))
        sigma_start = time()
        smoothflavor_seconds = NaN
        roc_seconds = NaN
        pr_seconds = NaN
        write_seconds = NaN
        auc = NaN
        auprc = NaN
        n_called_at_pos_thresh = missing
        tpr_at_pos_thresh = NaN
        fpr_at_pos_thresh = NaN
        precision_at_pos_thresh = NaN
        pos_prior = NaN
        error_message = ""
        status = "completed"
        prefix = joinpath(sim_outdir, "kernel_stddev_$(sigma_key)")

        fatal_stop = false
        try
            if sigma_seed !== nothing
                Random.seed!(sigma_seed)
            end

            sf_timed = @timed CodonMolecularEvolution.smoothFLAVOR_BAME(
                flavorgrid,
                prefix;
                pos_thresh=pos_thresh,
                iters=iters,
                burnin=burnin,
                n_chains=n_chains,
                verbosity=1,
                exports=false,
                sample_allocations=sample_allocations,
                fast_reshaping=fast_reshaping,
                kernel_stddev=σk,
            )
            df, results = sf_timed.value
            smoothflavor_seconds = sf_timed.time
            pos_prior = results.pos_prior

            site_df = copy(df)
            site_df[!, :true_positive] = truth_vec
            site_df[!, :kernel_stddev] .= σk

            roc_timed = @timed roc_curve_from_scores(site_df.posterior_prob_positive, truth_vec)
            roc_df, auc = roc_timed.value
            roc_seconds = roc_timed.time
            roc_df[!, :kernel_stddev] .= σk

            pr_timed = @timed begin
                site_truth_df = combine_site_truth(site_df, truth_df)
                pr_df, auprc_local = pr_curve_from_scores(site_truth_df.posterior_prob_positive, truth_vec)
                (site_truth_df, pr_df, auprc_local)
            end
            site_truth_df, pr_df, auprc = pr_timed.value
            pr_seconds = pr_timed.time
            pr_df[!, :kernel_stddev] .= σk

            th = threshold_summary(site_df.posterior_prob_positive, truth_vec, pos_thresh)
            n_called_at_pos_thresh = th.n_called
            tpr_at_pos_thresh = th.tpr
            fpr_at_pos_thresh = th.fpr
            precision_at_pos_thresh = th.precision

            write_timed = @timed begin
                CSV.write(prefix * "_site_posteriors.csv", site_df)
                CSV.write(prefix * "_roc.csv", roc_df)
                CSV.write(prefix * "_pr.csv", pr_df)
                CSV.write(prefix * "_site_posteriors_with_truth.csv", site_truth_df)
            end
            write_seconds = write_timed.time

            summary_row = (
                kernel_stddev=σk,
                auc=auc,
                auprc=auprc,
                n_sites=n_sites,
                n_true_positive_sites=n_pos,
                n_called_at_pos_thresh=n_called_at_pos_thresh,
                pos_thresh=pos_thresh,
                tpr_at_pos_thresh=tpr_at_pos_thresh,
                fpr_at_pos_thresh=fpr_at_pos_thresh,
                precision_at_pos_thresh=precision_at_pos_thresh,
                posterior_mean_positive_prior_mass=pos_prior,
                simulation_id=sim_id,
                alignment_path=String(row.alignment_path),
                tree_path=String(row.tree_path),
                true_rates_path=String(row.true_rates_path),
                n_sequences=n_sequences,
                truth_prevalence=truth_prevalence,
                flavorgrid_seconds=flavorgrid_seconds,
                smoothflavor_seconds=smoothflavor_seconds,
                roc_seconds=roc_seconds,
                pr_seconds=pr_seconds,
                write_seconds=write_seconds,
                total_sigma_seconds=time() - sigma_start,
            )
            append_csv_row(global_summary_path, summary_row)

            if save_all_site_scores
                site_scores_df = minimal_site_score_df(site_truth_df, sim_id, σk)
                if isfile(global_site_scores_path)
                    CSV.write(global_site_scores_path, site_scores_df; append=true, writeheader=false)
                else
                    CSV.write(global_site_scores_path, site_scores_df)
                end
            end

            completed_count += 1
        catch err
            status = "failed"
            failed_count += 1
            error_message = sprint(showerror, err, catch_backtrace())
            fatal_stop = !continue_on_error
        end

        progress_row = (
            timestamp=read_timestamp(),
            mode=mode,
            simulation_id=sim_id,
            simulation_index=sim_index,
            kernel_stddev=σk,
            kernel_stddev_key=sigma_key,
            seed=sigma_seed === nothing ? missing : sigma_seed,
            status=status,
            n_sequences=n_sequences,
            n_sites=n_sites,
            n_true_positive_sites=n_pos,
            truth_prevalence=truth_prevalence,
            flavorgrid_seconds=flavorgrid_seconds,
            smoothflavor_seconds=smoothflavor_seconds,
            roc_seconds=roc_seconds,
            pr_seconds=pr_seconds,
            write_seconds=write_seconds,
            total_sigma_seconds=time() - sigma_start,
            simulation_elapsed_seconds=time() - sim_start,
            auc=auc,
            auprc=auprc,
            n_called_at_pos_thresh=n_called_at_pos_thresh,
            tpr_at_pos_thresh=tpr_at_pos_thresh,
            fpr_at_pos_thresh=fpr_at_pos_thresh,
            precision_at_pos_thresh=precision_at_pos_thresh,
            pos_thresh=pos_thresh,
            sim_outdir=sim_outdir,
            output_prefix=prefix,
            alignment_path=String(row.alignment_path),
            tree_path=String(row.tree_path),
            true_rates_path=String(row.true_rates_path),
            error_message=error_message,
        )
        append_csv_row(progress_path, progress_row)

        if isfile(global_summary_path)
            sim_summary_df = DataFrame(CSV.File(global_summary_path))
            sim_summary_df = sim_summary_df[String.(sim_summary_df.simulation_id) .== sim_id, :]
            CSV.write(joinpath(sim_outdir, "kernel_stddev_study_summary_enriched.csv"), sim_summary_df)
        end

        fatal_stop && error(error_message)
    end

    simulation_status = failed_count == 0 ? "completed" : (completed_count > 0 ? "completed_with_failures" : "failed")
    total_simulation_seconds = time() - sim_start
    append_csv_row(simulation_runs_path, (
        timestamp=read_timestamp(),
        mode=mode,
        simulation_id=sim_id,
        simulation_index=sim_index,
        status=simulation_status,
        n_kernel_stddev_total=length(kernel_stddevs),
        n_kernel_stddev_completed=completed_count,
        n_kernel_stddev_failed=failed_count,
        flavorgrid_seconds=flavorgrid_seconds,
        total_simulation_seconds=total_simulation_seconds,
        alignment_path=String(row.alignment_path),
        tree_path=String(row.tree_path),
        true_rates_path=String(row.true_rates_path),
        error_message="",
    ))

    if update_aggregate_each_simulation && save_all_site_scores
        recompute_aggregate_outputs!(outdir; save_all_site_scores=save_all_site_scores)
    end

    return (
        simulation_id=sim_id,
        status=simulation_status,
        flavorgrid_seconds=flavorgrid_seconds,
        total_simulation_seconds=total_simulation_seconds,
        n_kernel_stddev_completed=completed_count,
        n_kernel_stddev_failed=failed_count,
    )
end

function run_omnibus_multi_kernel_stddev_sweep(
    rootdir::AbstractString,
    outdir::AbstractString;
    manifest=nothing,
    kernel_stddevs=Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    base_seed::Union{Nothing,Int}=nothing,
    save_all_site_scores::Bool=true,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    skip_completed::Bool=true,
    continue_on_error::Bool=true,
    update_aggregate_each_simulation::Bool=true,
    flavorgrid_kwargs...,
)
    manifest_df = _resolve_manifest(rootdir; manifest=manifest)
    initialize_run_outputs(outdir, manifest_df)

    statuses = NamedTuple[]
    for (sim_index, row) in enumerate(eachrow(manifest_df))
        push!(statuses, run_one_simulation!(
            row,
            sim_index,
            "full",
            outdir;
            kernel_stddevs=kernel_stddevs,
            base_seed=base_seed,
            save_all_site_scores=save_all_site_scores,
            flavorgrid_verbosity=flavorgrid_verbosity,
            optimize_branch_lengths=optimize_branch_lengths,
            pos_thresh=pos_thresh,
            iters=iters,
            burnin=burnin,
            n_chains=n_chains,
            fast_reshaping=fast_reshaping,
            sample_allocations=sample_allocations,
            skip_completed=skip_completed,
            continue_on_error=continue_on_error,
            update_aggregate_each_simulation=update_aggregate_each_simulation,
            flavorgrid_kwargs...,
        ))
    end

    aggregate_summary_df = recompute_aggregate_outputs!(outdir; save_all_site_scores=save_all_site_scores)
    simulation_summary_df = isfile(joinpath(outdir, "omnibus_multi_simulation_summary.csv")) ? DataFrame(CSV.File(joinpath(outdir, "omnibus_multi_simulation_summary.csv"))) : DataFrame()

    return (
        manifest_df=manifest_df,
        simulation_summary_df=simulation_summary_df,
        aggregate_summary_df=aggregate_summary_df,
        status_df=DataFrame(statuses),
    )
end

function run_one_omnibus_multi_kernel_stddev_study(
    rootdir::AbstractString,
    outdir::AbstractString;
    manifest=nothing,
    simulation_id::Union{Nothing,AbstractString}=nothing,
    simulation_index::Int=1,
    random_choice::Bool=false,
    rng_seed::Union{Nothing,Int}=nothing,
    kernel_stddevs=Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    base_seed::Union{Nothing,Int}=nothing,
    save_all_site_scores::Bool=true,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    skip_completed::Bool=true,
    continue_on_error::Bool=true,
    update_aggregate_each_simulation::Bool=true,
    flavorgrid_kwargs...,
)
    selected_row, manifest_df = select_one_omnibus_multi_simulation(
        rootdir;
        manifest=manifest,
        simulation_id=simulation_id,
        simulation_index=simulation_index,
        random_choice=random_choice,
        rng_seed=rng_seed,
    )

    selected_manifest_df = DataFrame(selected_row)
    initialize_run_outputs(outdir, selected_manifest_df)
    CSV.write(joinpath(outdir, "selected_simulation_manifest.csv"), selected_manifest_df)

    status = run_one_simulation!(
        selected_row,
        1,
        "single",
        outdir;
        kernel_stddevs=kernel_stddevs,
        base_seed=base_seed,
        save_all_site_scores=save_all_site_scores,
        flavorgrid_verbosity=flavorgrid_verbosity,
        optimize_branch_lengths=optimize_branch_lengths,
        pos_thresh=pos_thresh,
        iters=iters,
        burnin=burnin,
        n_chains=n_chains,
        fast_reshaping=fast_reshaping,
        sample_allocations=sample_allocations,
        skip_completed=skip_completed,
        continue_on_error=continue_on_error,
        update_aggregate_each_simulation=update_aggregate_each_simulation,
        flavorgrid_kwargs...,
    )

    aggregate_summary_df = recompute_aggregate_outputs!(outdir; save_all_site_scores=save_all_site_scores)
    simulation_summary_df = isfile(joinpath(outdir, "omnibus_multi_simulation_summary.csv")) ? DataFrame(CSV.File(joinpath(outdir, "omnibus_multi_simulation_summary.csv"))) : DataFrame()

    return (
        selected_manifest_row=selected_row,
        selected_manifest_df=selected_manifest_df,
        manifest_df=manifest_df,
        simulation_summary_df=simulation_summary_df,
        aggregate_summary_df=aggregate_summary_df,
        status_df=DataFrame([status]),
    )
end
