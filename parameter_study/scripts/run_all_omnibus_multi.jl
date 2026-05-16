include("common.jl")

opts = common_options()

run_all_omnibus_multi_parameter_sweep(
    opts.rootdir,
    opts.outdir;
    kernel_stddevs=opts.kernel_stddevs,
    include_original_bame=opts.include_original_bame,
    pos_thresh=opts.pos_thresh,
    iters=opts.iters,
    burnin=opts.burnin,
    n_chains=opts.n_chains,
    base_seed=opts.base_seed,
    flavorgrid_verbosity=opts.flavorgrid_verbosity,
    optimize_branch_lengths=opts.optimize_branch_lengths,
    fast_reshaping=opts.fast_reshaping,
    sample_allocations=opts.sample_allocations,
    skip_completed=opts.skip_completed,
    continue_on_error=true,
    update_aggregate_each_simulation=true,
)
