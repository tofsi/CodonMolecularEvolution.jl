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

function common_options()
    rootdir = getenv_string("OMNIBUS_ROOT", joinpath(PARAMETER_STUDY_DIR, "data", "omnibus-multi"))
    outdir = getenv_string("OMNIBUS_OUT", joinpath(PARAMETER_STUDY_DIR, "results", "omnibus-multi"))

    kernel_stddevs = parse_float_list(getenv_string("KERNEL_STDDEVS", "0.25,0.5,1,2,4,8"))

    return (
        rootdir=rootdir,
        outdir=outdir,
        kernel_stddevs=kernel_stddevs,
        include_original_bame=getenv_bool("INCLUDE_ORIGINAL_BAME", true),
        pos_thresh=getenv_float("POS_THRESH", 0.9),
        iters=getenv_int("ITERS", 1000),
        burnin=getenv_int("BURNIN", max(1, div(getenv_int("ITERS", 1000), 4))),
        n_chains=getenv_int("N_CHAINS", 4),
        base_seed=haskey(ENV, "BASE_SEED") ? getenv_int("BASE_SEED", 1) : nothing,
        flavorgrid_verbosity=getenv_int("FLAVORGRID_VERBOSITY", 1),
        optimize_branch_lengths=getenv_bool("OPTIMIZE_BRANCH_LENGTHS", false),
        fast_reshaping=getenv_bool("FAST_RESHAPING", true),
        sample_allocations=getenv_bool("SAMPLE_ALLOCATIONS", false),
        skip_completed=getenv_bool("SKIP_COMPLETED", true),
    )
end
