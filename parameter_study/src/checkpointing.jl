function collect_completed_keys(progress_path::AbstractString)
    if !isfile(progress_path)
        return Set{Tuple{String,String}}()
    end
    df = DataFrame(CSV.File(progress_path))
    if !(:simulation_id in names(df) && :kernel_stddev_key in names(df) && :status in names(df))
        return Set{Tuple{String,String}}()
    end
    completed = df[df.status .== "completed", [:simulation_id, :kernel_stddev_key]]
    return Set((String(r.simulation_id), String(r.kernel_stddev_key)) for r in eachrow(completed))
end

function initialize_run_outputs(outdir::AbstractString, manifest_df::DataFrame)
    mkpath(outdir)
    mkpath(joinpath(outdir, "simulations"))
    mkpath(joinpath(outdir, "aggregate"))
    CSV.write(joinpath(outdir, "omnibus_multi_manifest.csv"), manifest_df)
    return nothing
end
