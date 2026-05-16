module SmoothFlavorStudy

using CSV
using DataFrames
using Dates
using Random
using Statistics
using NPZ
using ZipFile

import CodonMolecularEvolution
import MolecularEvolution

include("io_utils.jl")
include("truth.jl")
include("metrics.jl")
include("model_study.jl")
include("omnibus_multi.jl")

export
    read_fasta_simple,
    validate_alignment_for_flavor,
    discover_omnibus_multi_simulations,
    select_one_omnibus_multi_simulation,
    run_one_omnibus_multi_parameter_sweep,
    run_all_omnibus_multi_parameter_sweep,
    run_parameter_sweep_on_flavorgrid!,
    load_omnibus_multi_truth,
    roc_curve_from_scores,
    pr_curve_from_scores,
    threshold_summary

end
