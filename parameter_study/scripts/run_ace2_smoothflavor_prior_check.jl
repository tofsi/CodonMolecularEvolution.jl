include("common.jl")

using Random
using StatsBase
using DataFrames
using CSV
using Statistics
using LinearAlgebra
using MolecularEvolution

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
const ANALYSIS_NAME = "../results/prior_predictive_checks"
const FASTA_PATH = "../../test/data/Ace2_with_bat/Ace2_with_bat.fasta"
const TREE_PATH = "../../test/data/Ace2_with_bat/Ace2_with_bat.tre"

const N_DRAWS = 5_000
const N_SAVED_WEIGHT_DRAWS = 20
const KERNEL_STDDEV = 4.0
const RNG_SEED = 1

"""
Draw from the ambient prior z ~ N(0, I), transform each draw to category
weights θ, and simulate latent site categories C_s ~ Categorical(θ).

The sequence likelihood is never evaluated here. The FLAVORgrid is used only
to define the category grid, category metadata, and number of codon sites.
"""
function sample_prior_check(flavorgrid; n_draws, kernel_stddev, seed)
    sk_model = CodonMolecularEvolution.SKBDIModel_from_FLAVOR(
        flavorgrid;
        kernel_stddev=kernel_stddev,
        suppress=false,
    )
    model = CodonMolecularEvolution.GeneralizedFUBARModel(sk_model)

    parameters = permutedims(reduce(hcat, sk_model.codon_param_vec))
    mu, shape, alpha, capped = eachcol(parameters)
    positive = Bool.(CodonMolecularEvolution.get_pos_sel_mask(flavorgrid))

    n_categories = model.n_categories
    n_sites = size(flavorgrid.prob_matrix, 2)
    rng = MersenneTwister(seed)

    theta_sum = zeros(n_categories)
    theta_sumsq = zeros(n_categories)
    draw_rows = NamedTuple[]
    saved_weights = NamedTuple[]

    for draw in 1:n_draws
        z = rand(rng, model.prior)
        theta = model.to_probability_vector(z)

        # A finite prior-predictive alignment of latent category allocations.
        allocations = sample(rng, 1:n_categories, Weights(theta), n_sites)

        theta_sum .+= theta
        theta_sumsq .+= theta .^ 2

        push!(draw_rows, (
            draw=draw,
            kernel_width=abs(kernel_stddev * z[1]),
            positive_mass=sum(theta[positive]),
            capped_mass=dot(theta, capped),
            maximum_weight=maximum(theta),
            effective_categories=1 / sum(abs2, theta),
            expected_mu=dot(theta, mu),
            expected_shape=dot(theta, shape),
            expected_alpha=dot(theta, alpha),
            predictive_positive_fraction=mean(positive[allocations]),
            predictive_capped_fraction=mean(capped[allocations]),
        ))

        if draw <= N_SAVED_WEIGHT_DRAWS
            append!(saved_weights, [
                (draw=draw, category=k, weight=theta[k])
                for k in 1:n_categories
            ])
        end
    end

    theta_mean = theta_sum ./ n_draws
    theta_sd = sqrt.(max.(theta_sumsq ./ n_draws .- theta_mean .^ 2, 0.0))

    category_summary = DataFrame(
        category=1:n_categories,
        mu=mu,
        shape=shape,
        alpha=alpha,
        capped=Bool.(capped),
        positive_selection_capable=positive,
        prior_mean_weight=theta_mean,
        prior_sd_weight=theta_sd,
    )

    return (
        draws=DataFrame(draw_rows),
        categories=category_summary,
        saved_weights=DataFrame(saved_weights),
    )
end

function main()
    # This is the same simple setup used for an ordinary FLAVOR analysis.
    seqnames, seqs = read_fasta(FASTA_PATH)
    treestring = readlines(TREE_PATH)[1]
    flavorgrid = FLAVORgrid(seqnames, seqs, treestring)

    result = sample_prior_check(
        flavorgrid;
        n_draws=N_DRAWS,
        kernel_stddev=KERNEL_STDDEV,
        seed=RNG_SEED,
    )

    outdir = ANALYSIS_NAME * "_prior_check"
    mkpath(outdir)

    CSV.write(joinpath(outdir, "prior_draws.csv"), result.draws)
    CSV.write(joinpath(outdir, "prior_draw_summary.csv"), describe(result.draws))
    CSV.write(joinpath(outdir, "prior_category_summary.csv"), result.categories)
    CSV.write(joinpath(outdir, "saved_weight_draws.csv"), result.saved_weights)

    println(describe(result.draws))
    println("\nResults written to: $outdir")
end

main()
