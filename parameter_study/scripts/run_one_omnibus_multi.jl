
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CodonMolecularEvolution
include(joinpath(@__DIR__, "..", "src", "SmoothFlavorStudy.jl"))
using .SmoothFlavorStudy

println("Using CodonMolecularEvolution from: ", pathof(CodonMolecularEvolution))
n_chains = max(1, Threads.nthreads() - 1)
result = run_one_omnibus_multi_kernel_stddev_study(
    joinpath(@__DIR__, "..", "data", "omnibus-multi"),
    joinpath(@__DIR__, "..", "output", "one-omnibus-multi");
    simulation_index = 1,
    kernel_stddevs = [0.5, 1.0, 2.0],
    iters = 300,
    burnin = 75,
    n_chains = n_chains,
    base_seed = 123,
    flavorgrid_verbosity = 1,
    pos_thresh = 0.9,
    skip_completed = true,
    continue_on_error = true,
    save_all_site_scores = true,
)

println(result.status_df)
