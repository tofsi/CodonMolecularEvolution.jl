using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CodonMolecularEvolution
include(joinpath(@__DIR__, "..", "src", "SmoothFlavorStudy.jl"))
using .SmoothFlavorStudy

println("Using CodonMolecularEvolution from: ", pathof(CodonMolecularEvolution))
n_chains = max(1, Threads.nthreads() - 1)
result = run_omnibus_multi_kernel_stddev_sweep(
    "/path/to/omnibus_multi",
    "/path/to/output/full_run";
    kernel_stddevs = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    iters = 1000,
    burnin = 250,
    n_chains = n_chains,
    base_seed = 123,
    flavorgrid_verbosity = 1,
    pos_thresh = 0.9,
    skip_completed = true,
    continue_on_error = true,
    save_all_site_scores = true,
)

println(result.status_df)
