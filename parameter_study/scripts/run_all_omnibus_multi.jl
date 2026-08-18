include("common.jl")

opts = common_options()

run_all_omnibus_multi_parameter_sweep(
    opts.rootdir,
    opts.outdir;
    manifest=opts.manifest,
    ranked_simulations=opts.ranked_simulations,
    ranked_simulation_count=opts.ranked_simulation_count,
    randomize_simulations=opts.randomize_simulations,
    random_simulation_count=opts.random_simulation_count,
    simulation_selection_seed=opts.simulation_selection_seed,
    replicate_ids=opts.replicate_ids,
    kernel_stddevs=opts.kernel_stddevs,
    include_original_bame=opts.include_original_bame,
    include_meme=opts.include_meme,
    meme_significance=opts.meme_significance,
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
    save_chain_samples=opts.save_chain_samples,
    skip_completed=opts.skip_completed,
    continue_on_error=true,
    update_aggregate_each_simulation=true,
)
