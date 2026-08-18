include("common.jl")

opts = common_options(
    default_iters=400,
    default_burnin=200,
    default_n_chains=4,
)

simulation_id = getenv_string("SIMULATION_ID", "sim_46_replicate_1")
kernel_stddev = getenv_float("CONVERGENCE_KERNEL_STDDEV", 1.0)
kernel_stddev > 0 || throw(ArgumentError("CONVERGENCE_KERNEL_STDDEV must be positive"))
opts.n_chains >= 2 || throw(ArgumentError("The convergence pilot requires N_CHAINS >= 2 for R-hat"))

kernel_label = replace(string(kernel_stddev), "." => "p")
configuration_name = join(
    (
        simulation_id,
        "kernel_$kernel_label",
        "iters_$(opts.iters)",
        "burnin_$(opts.burnin)",
        "adapts_$(opts.n_adapts)",
        "chains_$(opts.n_chains)",
    ),
    "_",
)
outdir = getenv_string(
    "CONVERGENCE_OUT",
    joinpath(PARAMETER_STUDY_DIR, "results", "convergence-pilot", configuration_name),
)

println("Convergence pilot output: ", outdir)
println(
    "Configuration: simulation=", simulation_id,
    ", kernel_stddev=", kernel_stddev,
    ", iters=", opts.iters,
    ", burnin=", opts.burnin,
    ", n_adapts=", opts.n_adapts,
    ", n_chains=", opts.n_chains,
)

pilot_summary = run_one_omnibus_multi_parameter_sweep(
    opts.rootdir,
    outdir;
    simulation_id=simulation_id,
    kernel_stddevs=[kernel_stddev],
    include_original_bame=false,
    include_meme=false,
    pos_thresh=opts.pos_thresh,
    iters=opts.iters,
    burnin=opts.burnin,
    n_adapts=opts.n_adapts,
    n_chains=opts.n_chains,
    base_seed=opts.base_seed,
    flavorgrid_verbosity=opts.flavorgrid_verbosity,
    optimize_branch_lengths=opts.optimize_branch_lengths,
    fast_reshaping=opts.fast_reshaping,
    sample_allocations=opts.sample_allocations,
    save_chain_samples=true,
    skip_completed=opts.skip_completed,
    continue_on_error=false,
    update_aggregate=true,
)

if getenv_bool("PLOT_CONVERGENCE_CHAINS", true)
    include("plot_convergence_pilot.jl")
    plot_convergence_pilot_from_env(joinpath(outdir, simulation_id))
end

pilot_summary
