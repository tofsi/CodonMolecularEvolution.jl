using Pkg

const PARAMETER_STUDY_DIR = normpath(joinpath(@__DIR__, ".."))
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

Pkg.activate(REPO_ROOT)

include(joinpath(PARAMETER_STUDY_DIR, "src", "SmoothFlavorStudy.jl"))
using .SmoothFlavorStudy
using CodonMolecularEvolution
using CSV
using DataFrames

println("Using CodonMolecularEvolution from: ", pathof(CodonMolecularEvolution))

function getenv_string(name::AbstractString, default::AbstractString)
    return get(ENV, name, default)
end

function getenv_int(name::AbstractString, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function getenv_float(name::AbstractString, default::Float64)
    return parse(Float64, get(ENV, name, string(default)))
end

function getenv_bool(name::AbstractString, default::Bool)
    v = lowercase(get(ENV, name, string(default)))
    return v in ("1", "true", "yes", "y")
end

function getenv_maybe_string(name::AbstractString)
    v = get(ENV, name, "")
    return isempty(v) ? nothing : v
end

function parse_float_list(s::AbstractString)
    vals = Float64[]
    for part in split(s, ',')
        p = strip(part)
        isempty(p) && continue
        push!(vals, parse(Float64, p))
    end
    return vals
end

function parse_int_list(s::AbstractString)
    vals = Int[]
    for part in split(s, ',')
        p = strip(part)
        isempty(p) && continue
        push!(vals, parse(Int, p))
    end
    return vals
end

function common_options(;
    default_iters::Int=100,
    default_burnin::Int=50,
    default_n_chains::Int=15,
)
    rootdir = getenv_string("OMNIBUS_ROOT", joinpath(PARAMETER_STUDY_DIR, "data", "omnibus-multi"))
    outdir = getenv_string("OMNIBUS_OUT", joinpath(PARAMETER_STUDY_DIR, "results", "omnibus-multi"))

    kernel_stddevs = parse_float_list(getenv_string("KERNEL_STDDEVS", "0.25,0.5,1,2,4,8"))
    random_simulation_count = haskey(ENV, "RANDOM_SIMULATION_COUNT") ?
        getenv_int("RANDOM_SIMULATION_COUNT", 1) : nothing
    iters = getenv_int("ITERS", default_iters)
    burnin = getenv_int("BURNIN", default_burnin)
    n_adapts = getenv_int("N_ADAPTS", burnin)
    n_chains = getenv_int("N_CHAINS", default_n_chains)

    0 <= n_adapts <= burnin < iters || throw(ArgumentError(
        "Require 0 <= N_ADAPTS <= BURNIN < ITERS; got " *
        "N_ADAPTS=$n_adapts, BURNIN=$burnin, ITERS=$iters",
    ))
    n_chains > 0 || throw(ArgumentError("N_CHAINS must be positive; got $n_chains"))

    return (
        rootdir=rootdir,
        outdir=outdir,
        manifest=getenv_maybe_string("OMNIBUS_MANIFEST"),
        kernel_stddevs=kernel_stddevs,
        include_original_bame=getenv_bool("INCLUDE_ORIGINAL_BAME", true),
        include_meme=getenv_bool("INCLUDE_MEME", true),
        meme_significance=getenv_float("MEME_SIGNIFICANCE", 0.05),
        pos_thresh=getenv_float("POS_THRESH", 0.9),
        iters=iters,
        burnin=burnin,
        n_adapts=n_adapts,
        n_chains=n_chains,
        base_seed=haskey(ENV, "BASE_SEED") ? getenv_int("BASE_SEED", 1) : nothing,
        flavorgrid_verbosity=getenv_int("FLAVORGRID_VERBOSITY", 1),
        optimize_branch_lengths=getenv_bool("OPTIMIZE_BRANCH_LENGTHS", false),
        fast_reshaping=getenv_bool("FAST_RESHAPING", true),
        sample_allocations=getenv_bool("SAMPLE_ALLOCATIONS", false),
        save_chain_samples=getenv_bool("SAVE_CHAIN_SAMPLES", true),
        skip_completed=getenv_bool("SKIP_COMPLETED", true),
        ranked_simulations=getenv_string(
            "RANKED_SIMULATIONS_FILE",
            joinpath(rootdir, "ranked_simulations.csv"),
        ),
        ranked_simulation_count=haskey(ENV, "RANKED_SIMULATION_COUNT") ?
            getenv_int("RANKED_SIMULATION_COUNT", 1) : nothing,
        randomize_simulations=getenv_bool(
            "RANDOMIZE_SIMULATIONS",
            random_simulation_count !== nothing,
        ),
        random_simulation_count=random_simulation_count,
        simulation_selection_seed=getenv_int("SIMULATION_SELECTION_SEED", 20260808),
        replicate_ids=parse_int_list(getenv_string("REPLICATE_IDS", "1")),
    )
end
