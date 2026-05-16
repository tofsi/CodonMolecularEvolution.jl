# smoothFLAVOR parameter study refactor

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
    inspect_omnibus_manifest.jl
  python/
    summarize_true_rate_means.py
```

## Key design

`model_study.jl` is the only place that knows how to compare methods on an already-built `FLAVORgrid`.

It runs:

1. `original_BAME` once.
2. `smoothFLAVOR_BAME` once for each positive `kernel_stddev`.

`omnibus_multi.jl` does not duplicate sweep logic. It only discovers simulations, loads one simulation, builds the `FLAVORgrid`, then calls `run_parameter_sweep_on_flavorgrid!`.

This avoids the earlier problem where changes to one study file did nothing because the omnibus driver had its own copy of the sweep loop.

## Install/copy

Copy the `src`, `scripts`, and `python` folders into:

```text
CodonMolecularEvolution.jl/parameter_study/
```

The scripts assume they live inside `parameter_study/scripts`.

## Julia package requirements

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.add(["CSV","DataFrames","NPZ","ZipFile"])'
```

`CodonMolecularEvolution` should be the local package in the repo root project.

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
N_CHAINS=8 \
julia --project=. -t 16 parameter_study/scripts/run_one_omnibus_multi.jl
```

Specific simulation id:

```bash
SIMULATION_ID=sim_86_replicate_1 julia --project=. -t 16 parameter_study/scripts/run_one_omnibus_multi.jl
```

## Run all simulations

```bash
julia --project=. -t 16 parameter_study/scripts/run_all_omnibus_multi.jl
```

This runs simulation-by-simulation, saving continuously after BAME and after each kernel value.

## Outputs

For each simulation:

```text
results/.../sim_0_replicate_1/
  original_BAME_SelectionOutput.csv
  original_BAME_site_posteriors.csv
  original_BAME_roc.csv
  original_BAME_pr.csv

  kernel_stddev_0p25_site_posteriors.csv
  kernel_stddev_0p25_roc.csv
  kernel_stddev_0p25_pr.csv
  ...

  method_sweep_summary.csv
```

At the output root:

```text
manifest.csv
aggregate_summary.csv
simulation_progress.csv
```

## Resume behavior

`skip_completed=true` by default.

A method/kernel is considered complete if its summary row is already present in `method_sweep_summary.csv`.

If the run stops halfway through a simulation, rerunning skips `original_BAME` and any completed `kernel_stddev` values, then continues from the first missing one.

## Notes

This refactor intentionally does not include the no-smoothing case. All `kernel_stddevs` must be strictly positive.
