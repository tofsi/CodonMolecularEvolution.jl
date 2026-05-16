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

function _simulation_is_complete(sim_outdir::AbstractString, kernel_stddevs; include_original_bame::Bool=true)
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

function _aggregate_summaries(outdir::AbstractString)
    summary_files = String[]
    for (dir, _, files) in walkdir(outdir)
        for f in files
            if f == "method_sweep_summary.csv"
                push!(summary_files, joinpath(dir, f))
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

    isempty(rows) && return DataFrame()
    agg = vcat(rows...; cols=:union)
    CSV.write(joinpath(outdir, "aggregate_summary.csv"), agg)
    return agg
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
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    base_seed::Union{Nothing,Int}=nothing,
    save_truth_table::Bool=true,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    skip_completed::Bool=true,
    continue_on_error::Bool=false,
    flavorgrid_kwargs=NamedTuple(),
)
    simulation_id = String(row.simulation_id)
    sim_outdir = joinpath(outdir, simulation_id)
    mkpath(sim_outdir)

    if skip_completed && _simulation_is_complete(sim_outdir, kernel_stddevs; include_original_bame=include_original_bame)
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
            bame_method=bame_method,
            pos_thresh=pos_thresh,
            iters=iters,
            burnin=burnin,
            n_chains=n_chains,
            verbosity=flavorgrid_verbosity,
            fast_reshaping=fast_reshaping,
            sample_allocations=sample_allocations,
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
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    base_seed::Union{Nothing,Int}=nothing,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
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
        bame_method=bame_method,
        pos_thresh=pos_thresh,
        iters=iters,
        burnin=burnin,
        n_chains=n_chains,
        base_seed=base_seed,
        flavorgrid_verbosity=flavorgrid_verbosity,
        optimize_branch_lengths=optimize_branch_lengths,
        fast_reshaping=fast_reshaping,
        sample_allocations=sample_allocations,
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
    kernel_stddevs=Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    include_original_bame::Bool=true,
    bame_method=(sampler=:DirichletEM, concentration=0.1, iterations=2500),
    pos_thresh::Float64=0.9,
    iters::Int=1000,
    burnin::Int=div(iters, 4),
    n_chains::Int=4,
    base_seed::Union{Nothing,Int}=nothing,
    flavorgrid_verbosity::Int=1,
    optimize_branch_lengths::Bool=false,
    fast_reshaping::Bool=true,
    sample_allocations::Bool=false,
    skip_completed::Bool=true,
    continue_on_error::Bool=true,
    update_aggregate_each_simulation::Bool=true,
    flavorgrid_kwargs=NamedTuple(),
)
    mkpath(outdir)
    manifest_df = _resolve_manifest(rootdir; manifest=manifest)
    CSV.write(joinpath(outdir, "manifest.csv"), manifest_df)

    for (sim_index, row) in enumerate(eachrow(manifest_df))
        run_simulation_row!(
            row,
            sim_index,
            outdir;
            kernel_stddevs=kernel_stddevs,
            include_original_bame=include_original_bame,
            bame_method=bame_method,
            pos_thresh=pos_thresh,
            iters=iters,
            burnin=burnin,
            n_chains=n_chains,
            base_seed=base_seed,
            flavorgrid_verbosity=flavorgrid_verbosity,
            optimize_branch_lengths=optimize_branch_lengths,
            fast_reshaping=fast_reshaping,
            sample_allocations=sample_allocations,
            skip_completed=skip_completed,
            continue_on_error=continue_on_error,
            flavorgrid_kwargs=flavorgrid_kwargs,
        )

        update_aggregate_each_simulation && _aggregate_summaries(outdir)
    end

    return _aggregate_summaries(outdir)
end
