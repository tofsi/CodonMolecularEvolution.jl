# SmoothFlavorStudy

## File layout

- `src/SmoothFlavorStudy.jl`  
  Module entrypoint. Loads all source files and exports the public API.

- `src/metrics.jl`  
  Generic binary-classification metrics. Builds ROC and PR curves, computes AUC and AUPRC, and computes single-threshold summaries.

- `src/truth.jl`  
  Ground-truth parsing and normalization. Converts truth labels to boolean vectors and parses `omnibus_multi_true_rates` tables into sitewise truth.

- `src/io_utils.jl`  
  File discovery, FASTA parsing, path normalization, and CSV/table helpers.

- `src/checkpointing.jl`  
  Checkpoint helpers for long runs. Appends progress rows, tracks completed `(simulation_id, kernel_stddev)` pairs, and initializes output folders.

- `src/kernel_study.jl`  
  Core single-dataset parameter study. Takes one alignment/tree/truth set, sweeps over `kernel_stddev`, writes site-level outputs plus ROC/PR outputs, and returns summaries.

- `src/omnibus_multi.jl`  
  Omnibus-multi orchestration. Discovers simulation triples `(alignment, tree, true_rates)`, runs one selected simulation or the full sweep, writes continuous checkpoints, and maintains aggregate summaries.

- `scripts/run_one_omnibus_multi.jl`  
  Minimal smoke-test entrypoint for one selected simulation.

- `scripts/run_full_omnibus_multi.jl`  
  Minimal full omnibus-multi sweep entrypoint.

## Public API

After including `src/SmoothFlavorStudy.jl`:

```julia
using .SmoothFlavorStudy
```

the main entrypoints are:

- `run_smoothFLAVOR_kernel_stddev_study(...)`
- `run_one_omnibus_multi_kernel_stddev_study(...)`
- `run_omnibus_multi_kernel_stddev_sweep(...)`

## Why this split

The goal is to keep the code maintainable:

- **metrics** does not know anything about FLAVOR or omnibus file layout.
- **truth** only knows how to convert biological truth into boolean labels.
- **kernel_study** only knows how to run one sweep on one dataset.
- **omnibus_multi** only knows how to discover datasets and orchestrate many sweeps.
- **checkpointing** only knows how to keep long jobs safe and resumable.

## Continuous-save behavior

The omnibus drivers checkpoint continuously:

- `progress.csv` gets one row per completed or failed `kernel_stddev`
- `simulation_runs.csv` gets one row per completed simulation
- `omnibus_multi_simulation_summary.csv` is appended incrementally
- `omnibus_multi_all_site_scores.csv` is appended incrementally if `save_all_site_scores=true`
- aggregate ROC/PR summaries are recomputed after each simulation by default

