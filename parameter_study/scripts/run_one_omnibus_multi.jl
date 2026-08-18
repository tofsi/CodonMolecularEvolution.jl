include("common.jl")

opts = common_options()

simulation_id = getenv_maybe_string("SIMULATION_ID")
simulation_index = getenv_int("SIMULATION_INDEX", 1)
random_choice = getenv_bool("RANDOM_CHOICE", false)
rng_seed = haskey(ENV, "RNG_SEED") ? getenv_int("RNG_SEED", 1) : nothing

run_one_omnibus_multi_parameter_sweep(
    opts.rootdir,
    opts.outdir;
    simulation_id=simulation_id,
    simulation_index=simulation_index,
    random_choice=random_choice,
    rng_seed=rng_seed,
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
    continue_on_error=false,
    update_aggregate=true,
)
