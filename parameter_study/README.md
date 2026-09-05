# smoothFLAVOR parameter study

This layout makes the sweep code single-source-of-truth:

```text
parameter_study/
  src/
    SmoothFlavorStudy.jl       # module entry point
    io_utils.jl                # FASTA/tree/tag utilities
    truth.jl                   # true-rate NPZ/table loaders
    metrics.jl                 # ROC/PR/threshold summaries
    model_study.jl             # one-FLAVORgrid study: BAME + smoothFLAVOR sweep
    omnibus_multi.jl           # omnibus-multi discovery and dataset driver
  scripts/
    run_one_omnibus_multi.jl   # one simulation/replicate
    run_all_omnibus_multi.jl   # full simulation-by-simulation sweep
    run_convergence_pilot.jl   # isolated one-simulation/one-kernel MCMC check
    inspect_omnibus_manifest.jl
  python/
    summarize_true_rate_means.py
```

## Key design

`model_study.jl` is the only place that knows how to compare methods. It uses
the already-built `FLAVORgrid` for BAME methods and the corresponding raw
alignment/tree inputs for MEME.

It runs:

1. `original_BAME` once.
2. `MEME` once, using its site-wise episodic-selection likelihood-ratio test.
3. `smoothFLAVOR_BAME` once for each positive `kernel_stddev`.

`kernel_stddev` is the prior standard deviation of smoothFLAVOR's sampled
bandwidth parameter, not a fixed kernel width. For ambient draw `z`, the
effective bandwidth is `abs(kernel_stddev * z)`.

`omnibus_multi.jl` does not duplicate sweep logic. It only discovers simulations, loads one simulation, builds the `FLAVORgrid`, then calls `run_parameter_sweep_on_flavorgrid!`.

This avoids the earlier problem where changes to one study file did nothing because the omnibus driver had its own copy of the sweep loop.

## Julia package requirements

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The repository project already declares the study dependencies, including CSV,
DataFrames, NPZ, and ZipFile.

## Run one simulation

From the repo root:

```bash
julia --project=. -t 16 parameter_study/scripts/run_one_omnibus_multi.jl
```

Useful environment variables:

```bash
SIMULATION_INDEX=1 \
KERNEL_STDDEVS=0.25,0.5,1,2,4,8 \
ITERS=1000 \
BURNIN=250 \
N_ADAPTS=250 \
N_CHAINS=8 \
julia --project=. -t 16 parameter_study/scripts/run_one_omnibus_multi.jl
```

`N_ADAPTS` controls the number of initial NUTS iterations used for sampler
adaptation. It defaults to `BURNIN`, and every entry point requires
`0 <= N_ADAPTS <= BURNIN < ITERS`. This ensures that no adaptation draws enter
the posterior summary.

MEME is included by default. Configure or disable it with:

```bash
INCLUDE_MEME=true MEME_SIGNIFICANCE=0.05 \
julia --project=. -t 16 parameter_study/scripts/run_one_omnibus_multi.jl
```

For ROC and PR curves, MEME is scored as `1 - p-value`. Its fixed operating
point remains the paper's native `p <= MEME_SIGNIFICANCE`; it does not use the
BAME posterior-probability threshold. Raw MEME p-values and LRTs are saved
without rounding.

Specific simulation id:

```bash
SIMULATION_ID=sim_86_replicate_1 julia --project=. -t 16 parameter_study/scripts/run_one_omnibus_multi.jl
```

## Focused convergence pilot

Before committing the time budget to the broad sweep, run one simulation and
one representative kernel with longer chains. This entry point excludes the
original BAME and MEME baselines and writes to a separate, configuration-specific
directory under `results/convergence-pilot/`:

```bash
ITERS=400 \
BURNIN=200 \
N_ADAPTS=200 \
N_CHAINS=4 \
SIMULATION_ID=sim_46_replicate_1 \
CONVERGENCE_KERNEL_STDDEV=1 \
julia --project=. -t 16 parameter_study/scripts/run_convergence_pilot.jl
```

The pilot defaults are the values above. Override `CONVERGENCE_OUT` if you want
another output root. For a detached SSH run, place the same command after
`nohup env ...` and redirect stdout/stderr to a dedicated log, or run it inside
`tmux`.

The pilot generates PNG and PDF chain plots automatically after sampling. It
plots all kernel parameters plus the six parameters with the highest R-hat.
Control this with:

```bash
TOP_RHAT_PARAMETERS=10 \
PLOT_PARAMETERS=kernel_1,ambient_weight_1344 \
PLOT_FORMATS=png,pdf \
ACF_MAX_LAG=50 \
julia --project=. parameter_study/scripts/plot_convergence_pilot.jl \
  parameter_study/results/convergence-pilot/CONFIGURATION/sim_46_replicate_1
```

Leave `PLOT_PARAMETERS` unset to select parameters automatically. Set
`PLOT_CONVERGENCE_CHAINS=false` to suppress automatic plotting in
`run_convergence_pilot.jl`. Every figure contains the complete chains, with the
burn-in interval shaded and its endpoint marked. A
`kernel_stddev_*_chain_plot_manifest.csv` records the plotted parameters,
R-hat/ESS values, autocorrelation lag limit, and output filenames. Each figure
contains a third panel with per-chain autocorrelation calculated only from the
retained samples; `ACF_MAX_LAG` controls the displayed lag range.

Inspect `kernel_stddev_1_chain_diagnostics_summary.csv` and
`kernel_stddev_1_sampler_diagnostics_summary.csv` first. A practical screen is
`max_rhat <= 1.01`, no parameters above that threshold, adequate bulk/tail ESS
for the parameters that drive the posterior, no retained numerical
errors/divergences, and no repeated maximum-tree-depth hits. The sampler summary
also reports E-BFMI and acceptance rates by chain.

Log posterior behavior is retained explicitly: the sampler trace contains each
iteration's `log_density`, energy, energy error, acceptance rate, step size,
trajectory length, and tree depth. Its summary reports R-hat and bulk/tail ESS
for retained log density and the range of its per-chain means. These checks make
the run auditable, though the run log is still useful for warnings emitted while
adaptation is taking place.

## Run all simulations

```bash
julia --project=. -t 16 parameter_study/scripts/run_all_omnibus_multi.jl
```

This runs simulation-by-simulation, saving continuously after original BAME,
MEME, and each kernel value.

## Run in a fixed random order and stop early

For the primary comparison, randomize all base simulations with a recorded seed
and run replicate 1. Each simulation completes original BAME, MEME, and every
smoothing kernel before the next simulation begins:

```bash
RANDOMIZE_SIMULATIONS=true \
SIMULATION_SELECTION_SEED=20260808 \
REPLICATE_IDS=1 \
julia --project=. -t 16 parameter_study/scripts/run_all_omnibus_multi.jl
```

With no count limit, the script works through all simulations in that fixed
random order. Press `Ctrl-C` to stop. Completed method/kernel checkpoints are
preserved, `aggregate_summary.csv` contains the fully processed simulations,
and rerunning the same command resumes in the same order. Keep the seed fixed.

To impose a maximum in addition to the time limit, add for example:

```bash
RANDOM_SIMULATION_COUNT=20
```

The saved `manifest.csv` records `selection_order`, `selection_seed`, and
`selection_method`. Random selection is uniform without replacement and cannot
be combined with ranked selection.

## Optional ranked exploratory sweep

Set `RANKED_SIMULATION_COUNT` to the number of top-ranked base simulations that
fit the available compute. Ranked sweeps run replicate 1 of each selected
simulation by default, in `suitability_rank` order. Because this enriches the
sample for favorable smoothing conditions, treat it as a secondary exploratory
analysis rather than the primary comparison:

```bash
RANKED_SIMULATION_COUNT=10 \
julia --project=. -t 16 parameter_study/scripts/run_all_omnibus_multi.jl
```

Choose other or additional replicates with a comma-separated list. This is
useful for a second validation stage after the broad replicate-1 sweep:

```bash
RANKED_SIMULATION_COUNT=5 \
REPLICATE_IDS=2,3,4,5 \
julia --project=. -t 16 parameter_study/scripts/run_all_omnibus_multi.jl
```

By default the ranking is read from
`parameter_study/data/omnibus-multi/ranked_simulations.csv`. Override it with
`RANKED_SIMULATIONS_FILE=/path/to/ranked_simulations.csv`. If
`RANKED_SIMULATION_COUNT` is unset, the existing behavior of sweeping every
discovered simulation is preserved.

The output `manifest.csv` contains only the allocated simulation/replicate
runs, along with their suitability rank and ranking metrics. Thus, the default
ranked sweep performs `N` runs for `N` ranked base simulations. Set
`REPLICATE_IDS=1,2,3,4,5` to perform all five replicates (`5N` runs). The
unranked run-all command is unchanged and still runs every discovered row.

## Outputs

For each simulation:

```text
results/.../sim_0_replicate_1/
  original_BAME_SelectionOutput.csv
  original_BAME_site_posteriors.csv
  original_BAME_roc.csv
  original_BAME_pr.csv

  MEME_site_statistics.csv
  MEME_roc.csv
  MEME_pr.csv

  kernel_stddev_0p25_site_posteriors.csv
  kernel_stddev_0p25_roc.csv
  kernel_stddev_0p25_pr.csv
  kernel_stddev_0p25_chain_diagnostics.csv
  kernel_stddev_0p25_chain_diagnostics_summary.csv
  kernel_stddev_0p25_chain_samples.jld2
  kernel_stddev_0p25_sampler_trace.csv
  kernel_stddev_0p25_sampler_diagnostics.csv
  kernel_stddev_0p25_sampler_diagnostics_summary.csv
  ...

  method_sweep_summary.csv
```

At the output root:

```text
manifest.csv
aggregate_summary.csv
simulation_progress.csv
```

## Reproducible result figures

Generate publication-oriented figures and auditable CSV tables directly from
completed method checkpoints:

```bash
RESULT_KERNEL_STDDEVS=0.25,1,2,4 \
PLOT_FORMATS=png,pdf,svg \
julia --project=. parameter_study/scripts/plot_omnibus_results.jl \
  parameter_study/results/omnibus-multi-big-800retained-16chains-random8
```

By default, the script discovers simulations that have Original BAME and every
requested smoothFLAVOR kernel. Incomplete simulations are recorded in
`analysis_manifest.csv` but excluded from the paired comparison. Freeze an
exact analysis set with a comma-separated list:

```bash
RESULT_SIMULATIONS=sim_19_replicate_1,sim_77_replicate_1,sim_90_replicate_1
```

The default output directory is `publication_results/` under the result root.
Override it with `RESULT_ANALYSIS_DIR`. Outputs include:

```text
aggregate_roc_curves.{png,pdf,svg}
aggregate_pr_curves.{png,pdf,svg}
aggregate_auc.{png,pdf,svg}
aggregate_runtime_mcmc_diagnostics.{png,pdf,svg}
analysis_manifest.csv
per_simulation_auc.csv
aggregate_auc_summary.csv
aggregate_roc_curves.csv
aggregate_pr_curves.csv
aggregate_runtime_mcmc_summary.csv
```

The ROC and precision-recall figures are macro-averages: every included
simulation is interpolated onto a common axis and contributes equal weight to
the mean curve. Their light ribbons and matching dashed boundaries are
pointwise confidence intervals; solid lines show the means. Method colors use
the Okabe-Ito palette for common forms of color-vision deficiency. The AUC
figure reports mean ROC AUC and PR AUC with confidence-interval error bars and
faint points for the contributing simulations. By default, all intervals are
two-sided 95% bias-corrected and accelerated (BCa) bootstrap intervals across
simulations, so the simulation rather than each site is the resampling unit.
The same seeded resamples are used for every method to preserve pairing. Set
`RESULT_CONFIDENCE_LEVEL` to change the level, `RESULT_BOOTSTRAP_SAMPLES` to
change the default 10,000 resamples, or `RESULT_BOOTSTRAP_SEED` to change the
default seed `20260904`.

Only simulations containing Original BAME, every requested smoothFLAVOR
kernel, and all corresponding ROC and PR files enter the aggregates. This
keeps every method comparison paired on the same set of complete simulations.
The CSV summaries also report the between-simulation standard deviation and
standard error used to construct the confidence intervals.

The runtime/MCMC figure shows every simulation as a faint point and the median
and interquartile range as the larger point and whiskers. Runtime is displayed
on a logarithmic scale and includes FLAVOR for direct comparison. The MCMC
panels contain smoothFLAVOR only because FLAVOR does not use the MCMC sampler.
Dashed reference lines mark R-hat 1.01 and ESS 100. The associated CSV includes
additional sampler summaries that are not all shown in the figure.

The JLD2 file stores all per-chain ambient samples, including the adaptation
and burn-in portion, together with `burnin`, `n_adapts`, and parameter names.
The CSV diagnostics are calculated only from retained iterations. Set
`SAVE_CHAIN_SAMPLES=false` to omit the potentially large JLD2 files; R-hat and
ESS diagnostics are still written.

The most useful convergence fields are also copied onto each smoothFLAVOR row
in `method_sweep_summary.csv` and therefore into `aggregate_summary.csv`. This
includes the MCMC configuration, maximum R-hat, minimum bulk/tail ESS, sampler
errors, maximum-tree-depth hits, E-BFMI, and log-density R-hat/ESS. That keeps
the method-comparison table self-contained for reporting and filtering.

## Resume behavior

`skip_completed=true` by default.

A method/kernel is considered complete if its summary row is already present in `method_sweep_summary.csv`.

If the run stops halfway through a simulation, rerunning skips `original_BAME` and any completed `kernel_stddev` values, then continues from the first missing one.

## Notes

This study intentionally does not include the no-smoothing case. All
`kernel_stddevs` must be strictly positive.
