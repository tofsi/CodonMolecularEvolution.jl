include("common.jl")

opts = common_options()
manifest = discover_omnibus_multi_simulations(opts.rootdir)

println("Discovered $(nrow(manifest)) simulations.")
println(first(manifest, min(10, nrow(manifest))))

mkpath(opts.outdir)
CSV.write(joinpath(opts.outdir, "manifest.csv"), manifest)
println("Wrote manifest to ", joinpath(opts.outdir, "manifest.csv"))
