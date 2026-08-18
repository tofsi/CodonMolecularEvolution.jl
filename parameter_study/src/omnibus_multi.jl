function discover_omnibus_multi_simulations(rootdir::AbstractString)
    all_files = String[]
    for (dir, _, files) in walkdir(rootdir)
        for f in files
            push!(all_files, joinpath(dir, f))
        end
    end

    isempty(all_files) && error("No files were found under $rootdir")

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

    isempty(true_rates_npz) && error("Could not find omnibus_multi_true_rates.npz under $rootdir")
    length(true_rates_npz) == 1 || error("Found multiple omnibus_multi_true_rates.npz files under $rootdir")

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

    isempty(replicate_rows) && error("No replicate alignment files matching sims.<sim>.settings.replicate.<rep> were found under $rootdir")
    isempty(tree_map) && error("No tree files matching sims.<sim>.nwk were found under $rootdir")

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
            sim=row.sim,
            replicate=row.replicate,
        ))
    end

    manifest_df = DataFrame(rows)
    nrow(manifest_df) > 0 || error("Found files under $rootdir, but no complete simulation rows could be constructed.")

    sort!(manifest_df, [:sim, :replicate])
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

function _select_manifest_replicates(
    manifest::DataFrame,
    selected_simulations,
    replicate_ids,
)
    replicate_ids === nothing && return manifest

    manifest_names = Set(Symbol.(names(manifest)))
    :replicate in manifest_names ||
        throw(ArgumentError("Manifest must contain a :replicate column to select replicate IDs"))

    selected_replicates = Int.(collect(replicate_ids))
    isempty(selected_replicates) && throw(ArgumentError("replicate_ids must not be empty"))
    all(>(0), selected_replicates) ||
        throw(ArgumentError("replicate IDs must be positive; got $selected_replicates"))
    length(unique(selected_replicates)) == length(selected_replicates) ||
        throw(ArgumentError("replicate_ids contains duplicate values: $selected_replicates"))

    available_pairs = Set(zip(Int.(manifest.sim), Int.(manifest.replicate)))
    missing_pairs = [
        (sim=sim, replicate=replicate)
        for sim in Int.(selected_simulations)
        for replicate in selected_replicates
        if (sim, replicate) ∉ available_pairs
    ]
    isempty(missing_pairs) ||
        error("Requested simulation/replicate rows were not found: $(join(missing_pairs, ", "))")

    keep = in(Set(selected_replicates)).(Int.(manifest.replicate))
    return manifest[keep, :]
end

"""
    select_ranked_omnibus_multi_simulations(
        manifest, ranked_simulations, count; replicate_ids=nothing
    )

Select rows belonging to the top `count` base simulations in a
`ranked_simulations.csv` file. By default all replicates are selected; pass
`replicate_ids` to select specific replicates. The returned manifest follows
`suitability_rank` order and includes the ranking columns, so the exact compute
allocation is recorded in the output manifest.
"""
function select_ranked_omnibus_multi_simulations(
    manifest::DataFrame,
    ranked_simulations,
    count::Int,
    ;
    replicate_ids=nothing,
)
    count > 0 || throw(ArgumentError("ranked simulation count must be positive; got $count"))

    ranked_df = if ranked_simulations isa AbstractString
        isfile(ranked_simulations) || error("Ranked simulations file was not found: $ranked_simulations")
        DataFrame(CSV.File(ranked_simulations))
    elseif ranked_simulations isa DataFrame
        copy(ranked_simulations)
    else
        throw(ArgumentError("ranked_simulations must be a CSV path or a DataFrame"))
    end

    manifest_names = Set(Symbol.(names(manifest)))
    :sim in manifest_names || throw(ArgumentError("Manifest must contain a :sim column to use ranked selection"))

    ranked_names = Set(Symbol.(names(ranked_df)))
    required = [:simulation_id, :suitability_rank]
    all(name -> name in ranked_names, required) ||
        throw(ArgumentError("Ranked simulations must contain columns $(required). Found columns: $(names(ranked_df))"))

    ranked_df[!, :simulation_id] = Int.(ranked_df[!, :simulation_id])
    ranked_df[!, :suitability_rank] = Int.(ranked_df[!, :suitability_rank])
    length(unique(ranked_df.simulation_id)) == nrow(ranked_df) ||
        throw(ArgumentError("Ranked simulations contain duplicate simulation_id values"))
    length(unique(ranked_df.suitability_rank)) == nrow(ranked_df) ||
        throw(ArgumentError("Ranked simulations contain duplicate suitability_rank values"))

    sort!(ranked_df, :suitability_rank)
    count <= nrow(ranked_df) ||
        throw(ArgumentError("Requested $count ranked simulations, but only $(nrow(ranked_df)) are available"))
    selected_ranks = first(ranked_df, count)

    missing_sims = setdiff(selected_ranks.simulation_id, unique(Int.(manifest.sim)))
    isempty(missing_sims) ||
        error("Ranked simulations were not found in the discovered manifest: $(join(missing_sims, ", "))")

    # Avoid colliding with the replicate-level manifest's string simulation_id.
    ranking_metadata = select(selected_ranks, Not(:simulation_id))
    insertcols!(ranking_metadata, 1, :sim => selected_ranks.simulation_id)
    selected = innerjoin(manifest, ranking_metadata; on=:sim)

    selected = _select_manifest_replicates(
        selected,
        selected_ranks.simulation_id,
        replicate_ids,
    )

    sort!(selected, [:suitability_rank, :replicate])
    return selected
end

"""
    select_random_omnibus_multi_simulations(
        manifest; count=nothing, replicate_ids=nothing, rng_seed=1
    )

Put base simulations in a deterministic random order and optionally keep only
the first `count`. The full selection order is independent of replicate choice,
so a replicate-1 discovery run and later validation runs can use the same base
simulation order.
"""
function select_random_omnibus_multi_simulations(
    manifest::DataFrame;
    count::Union{Nothing,Int}=nothing,
    replicate_ids=nothing,
    rng_seed::Int=1,
)
    manifest_names = Set(Symbol.(names(manifest)))
    :sim in manifest_names ||
        throw(ArgumentError("Manifest must contain a :sim column to use random selection"))

    simulations = sort(unique(Int.(manifest.sim)))
    isempty(simulations) && error("No base simulations were found in the manifest")

    selected_count = count === nothing ? length(simulations) : count
    selected_count > 0 ||
        throw(ArgumentError("random simulation count must be positive; got $selected_count"))
    selected_count <= length(simulations) || throw(ArgumentError(
        "Requested $selected_count random simulations, but only $(length(simulations)) are available",
    ))

    rng = MersenneTwister(rng_seed)
    random_order = simulations[randperm(rng, length(simulations))]
    selected_simulations = first(random_order, selected_count)
    order_df = DataFrame(
        sim=selected_simulations,
        selection_order=collect(1:selected_count),
        selection_seed=fill(rng_seed, selected_count),
        selection_method=fill("uniform_random_without_replacement", selected_count),
    )

    selected = innerjoin(manifest, order_df; on=:sim)
    selected = _select_manifest_replicates(
        selected,
        selected_simulations,
        replicate_ids,
    )
    sort!(selected, [:selection_order, :replicate])
    return selected
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

    manifest_names = Set(Symbol.(names(manifest_df)))
    required = [:simulation_id, :alignment_path, :tree_path, :true_rates_path]
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

function _simulation_is_complete(
    sim_outdir::AbstractString,
    kernel_stddevs;
    include_original_bame::Bool=true,
    include_meme::Bool=true,
)
    summary_path = joinpath(sim_outdir, "method_sweep_summary.csv")
    isfile(summary_path) || return false

    df = DataFrame(CSV.File(summary_path))
    nrow(df) == 0 && return false
    nms = Set(Symbol.(names(df)))
    (:method in nms && :kernel_stddev in nms) || return false

    methods = String.(df[!, :method])
    if include_original_bame && !any(methods .== "original_BAME")
        return false
    end
    if include_meme && !any(methods .== "MEME")
        return false
    end

    for σ in Float64.(kernel_stddevs)
        rows = (methods .== "smoothFLAVOR_BAME") .& (Float64.(df[!, :kernel_stddev]) .== σ)
        any(rows) || return false
    end

    return true
end

function _read_row_string(row, name::Symbol)
    return String(getproperty(row, name))
end

function _row_has_property(row, name::Symbol)
    return name in propertynames(row)
end

function _load_simulation_inputs(row)
    alignment_path = _read_row_string(row, :alignment_path)
    tree_path = _read_row_string(row, :tree_path)
    true_rates_path = _read_row_string(row, :true_rates_path)

    sim = _row_has_property(row, :sim) ? Int(row.sim) : nothing
    replicate = _row_has_property(row, :replicate) ? Int(row.replicate) : nothing
    meta_path = _row_has_property(row, :true_rates_meta_path) ? String(row.true_rates_meta_path) : nothing

    seqnames, seqs = read_fasta_simple(alignment_path)
    seqnames, seqs, _ = trim_long_sequences_to_modal_length(seqnames, seqs; source=alignment_path)
    validate_alignment_for_flavor(seqnames, seqs; source=alignment_path)

    raw_treestring = read(tree_path, String)
    treestring = strip_tree_group_tags_for_flavor(raw_treestring)

    truth_vec, truth_df = load_omnibus_multi_truth(
        true_rates_path;
        sim=sim,
        replicate=replicate,
        meta_path=meta_path,
    )

    return seqnames, seqs, treestring, truth_vec, truth_df
end

function _aggregate_summaries(outdir::AbstractString; simulation_ids=nothing)
    allowed_ids = simulation_ids === nothing ? nothing : Set(String.(simulation_ids))
    summary_files = String[]
    for (dir, _, files) in walkdir(outdir)
        for f in files
            if f == "method_sweep_summary.csv"
                sim_id = basename(dir)
                if allowed_ids === nothing || sim_id in allowed_ids
                    push!(summary_files, joinpath(dir, f))
                end
            end
        end
    end

    rows = DataFrame[]
    for path in sort(summary_files)
        df = DataFrame(CSV.File(path))
        sim_id = basename(dirname(path))
        df[!, :simulation_id] .= sim_id
        push!(rows, df)
    end

    if isempty(rows)
        agg = DataFrame(simulation_id=String[])
        CSV.write(joinpath(outdir, "aggregate_summary.csv"), agg)
        return agg
    end
    agg = vcat(rows...; cols=:union)
    CSV.write(joinpath(outdir, "aggregate_summary.csv"), agg)
    return agg
end

function _completed_simulation_ids(
    manifest_df::DataFrame,
    outdir::AbstractString,
    kernel_stddevs;
    include_original_bame::Bool=true,
    include_meme::Bool=true,
)
    completed = String[]
    for row in eachrow(manifest_df)
        simulation_id = String(row.simulation_id)
        sim_outdir = joinpath(outdir, simulation_id)
        if _simulation_is_complete(
            sim_outdir,
            kernel_stddevs;
            include_original_bame=include_original_bame,
            include_meme=include_meme,
        )
            push!(completed, simulation_id)
        end
    end
    return completed
end

function _append_progress(outdir::AbstractString, row)
    progress_path = joinpath(outdir, "simulation_progress.csv")
    append_csv_row(progress_path, row)
end

function run_simulation_row!(
    row,
    sim_index::Int,
    outdir::AbstractString;
    kernel_stddevs=Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    include_original_bame::Bool=true,
    include_meme::Bool=true,
    meme_significance::Float64=0.05,
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_adapts::Int=burnin,
    n_chains::Int=4,
    base_seed::Union{Nothing,Int}=nothing,
    save_truth_table::Bool=true,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    save_chain_samples::Bool=true,
    skip_completed::Bool=true,
    continue_on_error::Bool=false,
    flavorgrid_kwargs=NamedTuple(),
)
    simulation_id = String(row.simulation_id)
    sim_outdir = joinpath(outdir, simulation_id)
    mkpath(sim_outdir)

    if skip_completed && _simulation_is_complete(
        sim_outdir,
        kernel_stddevs;
        include_original_bame=include_original_bame,
        include_meme=include_meme,
    )
        @info "Skipping completed simulation" simulation_id=simulation_id
        return DataFrame(CSV.File(joinpath(sim_outdir, "method_sweep_summary.csv")))
    end

    start_time = read_timestamp()

    try
        seqnames, seqs, treestring, truth_vec, truth_df = _load_simulation_inputs(row)

        if save_truth_table
            CSV.write(joinpath(sim_outdir, "truth.csv"), truth_df)
        end

        flavorgrid_elapsed = @elapsed flavorgrid = CodonMolecularEvolution.FLAVORgrid(
            seqnames,
            seqs,
            treestring;
            verbosity=flavorgrid_verbosity,
            optimize_branch_lengths=optimize_branch_lengths,
            flavorgrid_kwargs...,
        )

        summary_df = run_parameter_sweep_on_flavorgrid!(
            flavorgrid,
            truth_vec,
            sim_outdir;
            kernel_stddevs=kernel_stddevs,
            include_original_bame=include_original_bame,
            include_meme=include_meme,
            meme_inputs=(
                seqnames=seqnames,
                seqs=seqs,
                treestring=treestring,
                optimize_branch_lengths=optimize_branch_lengths,
            ),
            meme_significance=meme_significance,
            bame_method=bame_method,
            pos_thresh=pos_thresh,
            iters=iters,
            burnin=burnin,
            n_adapts=n_adapts,
            n_chains=n_chains,
            verbosity=flavorgrid_verbosity,
            fast_reshaping=fast_reshaping,
            sample_allocations=sample_allocations,
            save_chain_samples=save_chain_samples,
            base_seed=base_seed,
            skip_completed=skip_completed,
        )

        _append_progress(outdir, (
            timestamp=read_timestamp(),
            simulation_id=simulation_id,
            sim_index=sim_index,
            status="ok",
            start_time=start_time,
            flavorgrid_elapsed_seconds=flavorgrid_elapsed,
            message="",
        ))

        return summary_df
    catch err
        if err isa InterruptException
            _append_progress(outdir, (
                timestamp=read_timestamp(),
                simulation_id=simulation_id,
                sim_index=sim_index,
                status="interrupted",
                start_time=start_time,
                flavorgrid_elapsed_seconds=NaN,
                message="Interrupted by user; completed method checkpoints were preserved.",
            ))
            rethrow()
        end

        _append_progress(outdir, (
            timestamp=read_timestamp(),
            simulation_id=simulation_id,
            sim_index=sim_index,
            status="error",
            start_time=start_time,
            flavorgrid_elapsed_seconds=NaN,
            message=sprint(showerror, err),
        ))

        if continue_on_error
            @error "Simulation failed; continuing" simulation_id=simulation_id exception=(err, catch_backtrace())
            return DataFrame()
        else
            rethrow()
        end
    end
end

function run_one_omnibus_multi_parameter_sweep(
    rootdir::AbstractString,
    outdir::AbstractString;
    manifest=nothing,
    simulation_id::Union{Nothing,AbstractString}=nothing,
    simulation_index::Int=1,
    random_choice::Bool=false,
    rng_seed::Union{Nothing,Int}=nothing,
    kernel_stddevs=Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    include_original_bame::Bool=true,
    include_meme::Bool=true,
    meme_significance::Float64=0.05,
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_adapts::Int=burnin,
    n_chains::Int=4,
    base_seed::Union{Nothing,Int}=nothing,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    save_chain_samples::Bool=true,
    skip_completed::Bool=true,
    continue_on_error::Bool=false,
    update_aggregate::Bool=true,
    flavorgrid_kwargs=NamedTuple(),
)
    mkpath(outdir)
    row, manifest_df = select_one_omnibus_multi_simulation(
        rootdir;
        manifest=manifest,
        simulation_id=simulation_id,
        simulation_index=simulation_index,
        random_choice=random_choice,
        rng_seed=rng_seed,
    )

    CSV.write(joinpath(outdir, "manifest.csv"), manifest_df)

    row_index = findfirst(manifest_df.simulation_id .== row.simulation_id)
    summary_df = run_simulation_row!(
        row,
        Int(row_index),
        outdir;
        kernel_stddevs=kernel_stddevs,
        include_original_bame=include_original_bame,
        include_meme=include_meme,
        meme_significance=meme_significance,
        bame_method=bame_method,
        pos_thresh=pos_thresh,
        iters=iters,
        burnin=burnin,
        n_adapts=n_adapts,
        n_chains=n_chains,
        base_seed=base_seed,
        flavorgrid_verbosity=flavorgrid_verbosity,
        optimize_branch_lengths=optimize_branch_lengths,
        fast_reshaping=fast_reshaping,
        sample_allocations=sample_allocations,
        save_chain_samples=save_chain_samples,
        skip_completed=skip_completed,
        continue_on_error=continue_on_error,
        flavorgrid_kwargs=flavorgrid_kwargs,
    )

    update_aggregate && _aggregate_summaries(outdir)

    return summary_df
end

function run_all_omnibus_multi_parameter_sweep(
    rootdir::AbstractString,
    outdir::AbstractString;
    manifest=nothing,
    ranked_simulations=nothing,
    ranked_simulation_count::Union{Nothing,Int}=nothing,
    randomize_simulations::Bool=false,
    random_simulation_count::Union{Nothing,Int}=nothing,
    simulation_selection_seed::Int=1,
    replicate_ids=nothing,
    kernel_stddevs=Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    include_original_bame::Bool=true,
    include_meme::Bool=true,
    meme_significance::Float64=0.05,
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_adapts::Int=burnin,
    n_chains::Int=4,
    base_seed::Union{Nothing,Int}=nothing,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    save_chain_samples::Bool=true,
    skip_completed::Bool=true,
    continue_on_error::Bool=true,
    update_aggregate_each_simulation::Bool=true,
    flavorgrid_kwargs=NamedTuple(),
)
    mkpath(outdir)
    manifest_df = _resolve_manifest(rootdir; manifest=manifest)
    use_random_selection = randomize_simulations || random_simulation_count !== nothing
    if ranked_simulation_count !== nothing && use_random_selection
        throw(ArgumentError(
            "Ranked and random simulation selection are mutually exclusive. " *
            "Set either RANKED_SIMULATION_COUNT or RANDOMIZE_SIMULATIONS/RANDOM_SIMULATION_COUNT.",
        ))
    elseif use_random_selection
        manifest_df = select_random_omnibus_multi_simulations(
            manifest_df;
            count=random_simulation_count,
            replicate_ids=replicate_ids,
            rng_seed=simulation_selection_seed,
        )
        @info "Selected simulations in fixed random order" base_simulations=length(unique(manifest_df.sim)) selection_seed=simulation_selection_seed replicate_ids=replicate_ids replicate_runs=nrow(manifest_df)
    elseif ranked_simulation_count !== nothing
        ranked_simulations === nothing &&
            throw(ArgumentError("ranked_simulations is required when ranked_simulation_count is set"))
        manifest_df = select_ranked_omnibus_multi_simulations(
            manifest_df,
            ranked_simulations,
            ranked_simulation_count,
            replicate_ids=replicate_ids,
        )
        @info "Selected ranked simulations" base_simulations=ranked_simulation_count replicate_ids=replicate_ids replicate_runs=nrow(manifest_df)
    end
    CSV.write(joinpath(outdir, "manifest.csv"), manifest_df)

    aggregate_completed_simulations() = _aggregate_summaries(
        outdir;
        simulation_ids=_completed_simulation_ids(
            manifest_df,
            outdir,
            kernel_stddevs;
            include_original_bame=include_original_bame,
            include_meme=include_meme,
        ),
    )

    for (sim_index, row) in enumerate(eachrow(manifest_df))
        run_simulation_row!(
            row,
            sim_index,
            outdir;
            kernel_stddevs=kernel_stddevs,
            include_original_bame=include_original_bame,
            include_meme=include_meme,
            meme_significance=meme_significance,
            bame_method=bame_method,
            pos_thresh=pos_thresh,
            iters=iters,
            burnin=burnin,
            n_adapts=n_adapts,
            n_chains=n_chains,
            base_seed=base_seed,
            flavorgrid_verbosity=flavorgrid_verbosity,
            optimize_branch_lengths=optimize_branch_lengths,
            fast_reshaping=fast_reshaping,
            sample_allocations=sample_allocations,
            save_chain_samples=save_chain_samples,
            skip_completed=skip_completed,
            continue_on_error=continue_on_error,
            flavorgrid_kwargs=flavorgrid_kwargs,
        )

        update_aggregate_each_simulation && aggregate_completed_simulations()
    end

    return aggregate_completed_simulations()
end
