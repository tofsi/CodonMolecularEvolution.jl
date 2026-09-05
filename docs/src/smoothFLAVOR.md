# smoothFLAVOR

smoothFLAVOR estimates FLAVOR mixture-category weights with a smooth
logistic-normal prior and NUTS. It reuses the conditional-likelihood grid from
[`FLAVORgrid`](@ref), correlating neighboring category logits along the gamma
mean (`mu`) and synonymous-rate (`alpha`) axes. Gamma `shape` and the
uncapped/capped axis are not smoothed.

The bandwidth is inferred rather than fixed. If its standard-normal ambient
parameter is ``z``, a draw has bandwidth
``|\mathtt{kernel\_stddev}\,z|``. Thus `kernel_stddev` controls the prior scale
of smoothing, while the posterior can still concentrate near no smoothing.

## Basic use

Construct the same likelihood grid used for an ordinary FLAVOR analysis, then
pass it to `smoothFLAVOR_BAME` with an output-file prefix:

```julia
using CodonMolecularEvolution

flavorgrid = FLAVORgrid(seqnames, seqs, treestring)
df, results = CodonMolecularEvolution.smoothFLAVOR_BAME(
    flavorgrid,
    "output/example";
    iters=1_000,
    burnin=250,
    n_adapts=250,
    n_chains=4,
    kernel_stddev=1.0,
    save_chain_diagnostics=true,
)
```

`iters` is the total number of draws per chain, including adaptation and
burn-in. Always use `0 <= n_adapts <= burnin < iters`, so adaptation draws do
not enter the posterior summary. Multiple chains can run concurrently when
Julia is started with multiple threads, for example `julia --threads=auto`.

The returned table contains the posterior probability of a
positive-selection-capable category, its BAME-style Bayes factor, and the
`pos_thresh` call for each codon site. The second return value contains the raw
chains, NUTS statistics, fitted category summaries, and constructed model.

The default `iters=10` is intended for smoke tests, not inference. For a real
analysis, inspect R-hat, effective sample size, numerical errors, tree-depth
hits, E-BFMI, and retained log-density diagnostics before interpreting site
results.

## Output files

With `exports=true`, the site table is written as
`PREFIX_smoothFLAVOR_BAME.csv`. Enabling `save_chain_diagnostics` adds
parameter-chain and NUTS diagnostic CSV files. `save_chain_samples=true` also
writes all draws—including adaptation and burn-in—to
`PREFIX_chain_samples.jld2`; it can be large. The output directory must exist
before the analysis starts.

## API

```@docs; canonical=false
CodonMolecularEvolution.smoothFLAVOR_BAME
CodonMolecularEvolution.SKBDIModel_from_FLAVOR
CodonMolecularEvolution.GeneralizedFUBARModel_from_FLAVOR
CodonMolecularEvolution.summarize_smoothFLAVOR_BAME
CodonMolecularEvolution.save_smoothFLAVOR_chain_artifacts!
CodonMolecularEvolution.save_NUTS_sampler_artifacts!
CodonMolecularEvolution.flavor_con_lik_matrix
CodonMolecularEvolution.flavor_parameter_metadata
```
