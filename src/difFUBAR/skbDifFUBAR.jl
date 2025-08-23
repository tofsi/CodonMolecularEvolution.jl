using Statistics, Combinatorics, Distributions, MCMCChains, LinearAlgebra, Phylo, FASTX, MolecularEvolution, CodonMolecularEvolution, Plots, EllipticalSliceSampling, AbstractMCMC, NNlib, StatsBase, ProgressMeter, JLD2, AdvancedHMC, Zygote, ADTypes, DifferentiationInterface, LogDensityProblems, Mooncake, LogDensityProblemsAD
#import Zygote
#import LogDensityProblemsAD
include("convolution.jl")

"""
# GeneralizedFUBARModel
A general framework for the class of models to which FUBAR and skbdi belong.
## Fields
- n_parameters: Number of parameters in the model
- n_categories: Number of categories in the model
- log_likelihood: log likelihood for a probability vector for the categories
- to_probability_vector: returns the probability vector for a given set of parameters
- prior: the prior distribution in parameter space.
- con_lik_matrix: the matrix of conditional distributions for the sites.
"""
struct GeneralizedFUBARModel
    n_parameters::Int64
    n_categories::Int64
    log_likelihood::Function
    to_probability_vector::Function
    prior::Distribution
    con_lik_matrix::Matrix{Float64}
end

function log_posterior(model::GeneralizedFUBARModel, parameters::AbstractVector{<:Real})
    # log_posterior = log_prior + log_likelihood up to an additive constant
    return model.log_likelihood(parameters) + logpdf(model.prior, parameters)
end

"""
FUBARLogDensity
This struct implements the LogDensityProblems interface for the FUBAR model.
"""
struct FUBARLogDensity
    model::GeneralizedFUBARModel
end

function LogDensityProblems.logdensity(p::FUBARLogDensity, parameters::AbstractVector{<:Real})
    return log_posterior(p.model, parameters)
end

function LogDensityProblems.dimension(p::FUBARLogDensity)
    return p.model.n_parameters
end

function LogDensityProblems.capabilities(::Type{FUBARLogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end


"""
# sample_NUTS()
Samples from the model using NUTS
## Arguments:
- model::GeneralizedFUBARModel: The model to sample from.
- iters::Int64: The number of iterations to sample.
- n_chains::Int64: The number of chains to use for sampling.
- progress::Bool: Whether to show progress bars.
## Returns:
- ambient_samples::Vector{Any}: The sampled parameter values. Indexed as 
- stats::Any: The sampling statistics.
"""
function sample_NUTS(model::GeneralizedFUBARModel, iters::Int64, n_chains::Int64; progress=false)

    initial_parameters = rand(model.prior)
    target = FUBARLogDensity(model)
    metric = DiagEuclideanMetric(model.n_parameters)
    hamiltonian = Hamiltonian(metric, target, AutoMooncake(; config=nothing)) # We use mooncake because it is good for the case R^n → R
    n_adapts = div(iters, 10) # Number of adaptation steps, Default is min(1000, div(iters, 10))
    print("finding good epsilon...\n")
    epsilon = find_good_stepsize(hamiltonian, initial_parameters)
    integrator = Leapfrog(epsilon)
    print("Epsilon used: ", epsilon, "\n")
    ambient_samples = Vector{Any}(undef, n_chains)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
    stats = nothing
    if n_chains > 1
        Threads.@sync for i in 1:n_chains
            Threads.@spawn begin
                local_initial = rand(model.prior)
                local_samples, _ = sample(hamiltonian, kernel, local_initial, iters, adaptor, n_adapts; verbose=false)
                ambient_samples[i] = local_samples
            end
        end
    else
        samples, stats = sample(hamiltonian, kernel, initial_parameters, iters, adaptor, n_adapts; verbose=false, progress=false)
        ambient_samples[1] = samples
    end

    return ambient_samples, stats
end

"""
# SKBDIModel: 
holds the model parameters for
a general skbdi model.
## Fields:
- parameter_grids: A vector of vectors, where each inner vector contains 
                 the parameter values for a specific grid.
- parameter_names: A vector of strings, where each string is the name 
                 of a parameter corresponding to the parameter values in parameter_grids.
- hypothesis_masks: A matrix of booleans with mask[hypothesis, category] = true 
       if the hypothesis is true for the category.
- transition_functions: A vector of transition functions F:R -> [0, 1] for the skbdi model
- log_con_lik_matrix: A matrix of log likelihoods for each category.
- con_lik_matrix: A matrix of likelihoods for each category.
- codon_param_vec: A vector of vectors indexed category, parameter containing
                 parameter values for each category
- codon_param_index_vec: A vector of integers indexed category, parameter containing the indices 
                       in the parameter grids for each category.
- ambient_to_parameter_transform: A function that transforms an ambient sample into the parameter space.
- kernel_dim: The number of kernel parameters.
- grid_sizes: A tuple of integers representing the sizes of the grids for each parameter.
- masks: The disjoint masks used by the model when applying suppression parameters.
- suppression_dim: The number of suppression parameters.
- unsuppressed_dim: The number of unsuppressed parameters.
- total_dim: The total number of parameters in the model.
- n_codon_parameters: The number of codon parameters (e.g. alpha, beta, omega_1) in the model.
"""
struct SKBDIModel
    # Model Parameters:
    parameter_grids::Vector{Vector{Float64}}
    parameter_names::Vector{String}
    hypothesis_masks::Matrix{Bool}
    transition_function::Function
    log_con_lik_matrix::Matrix{Float64}
    con_lik_matrix::Matrix{Float64}
    codon_param_vec::Vector{Vector{Float64}}
    codon_param_index_vec::Vector{Vector{Int64}}
    ambient_to_parameter_transform::Function
    kernel_dim::Int64
    grid_sizes::Tuple
    ## Derived fields:
    masks::Matrix{Bool}
    mask_subset_indicators::Matrix{Bool}
    suppression_dim::Int64
    unsuppressed_dim::Int64
    total_dim::Int64
    n_codon_parameters::Int64
end


"""
# GeneralizedFUBARModel(model::SKBDIModel)
Constructor for creating a GeneralizedFUBARModel from a SKBDIModel.
"""
function GeneralizedFUBARModel(model::SKBDIModel)
    return GeneralizedFUBARModel(
        model.total_dim,
        model.unsuppressed_dim,
        a -> log_likelihood(model, a),
        a -> to_probability_vector(model, a),
        MvNormal(zeros(model.total_dim), I),
        model.con_lik_matrix)
end

"""
# gen_disjoint_masks()
generates additional suppression masks for each non-empty combination of hypotheses
## Arguments:
hypothesis_masks::Matrix{Bool}: The original hypothesis masks. Indexed mask[hypothesis, category] = true if the hypothesis is true for the category.
## Returns:
disjoint_masks::Array{Bool}: The generated disjoint masks, indexed [mask, category].
mask_subset_indicators::Matrix{Bool}: mask_subset_indicators[mask, hypothesis] == true if the hypothesis is true for this mask.
"""
function gen_disjoint_masks(hypothesis_masks::Matrix{Bool})
    disjoint_masks = Vector{Bool}[]
    mask_subset_indicators = Vector{Bool}[]
    # Go through subsets WARNING: grows super fast with number of hypotheses
    for i = 1:(2^(size(hypothesis_masks, 1))-1)
        b = [(i >> (j - 1) & 1) == 1 for j = 1:size(hypothesis_masks, 1)]
        # u (⋃_{j:b_j == 1} masks[j])∖(⋂_{j:b_j==0} masks[j])
        u = prod(hypothesis_masks[b, :], dims=1) .& .!(sum(hypothesis_masks[.!b, :], dims=1) .> 0)
        if any(u)
            push!(disjoint_masks, Matrix(u)[1, :]) # Ugh annoying typecasting bitmatrix → vector{bool}
            push!(mask_subset_indicators, b)
        end
    end
    return collect(hcat(disjoint_masks...)'), collect(hcat(mask_subset_indicators...)')
end

"""
SKBDIModel constructor
Creates a SKBDIModel given a set of parameters (upto "derived fields" in the struct docstring).
See above docstring for field descriptions.
"""
function SKBDIModel(parameter_grids::Vector{Vector{Float64}},
    parameter_names::Vector{String},
    hypothesis_masks::Matrix{Bool},
    transition_function::Function,
    log_con_lik_matrix::Matrix{Float64},
    con_lik_matrix::Matrix{Float64},
    codon_param_vec::Vector{Vector{Float64}},
    codon_param_index_vec::Vector{Vector{Int64}},
    ambient_to_parameter_transform::Function,
    kernel_dim::Int64,
    grid_sizes::Tuple)
    masks, mask_subset_indicators = gen_disjoint_masks(hypothesis_masks)
    suppression_dim = size(hypothesis_masks)[1]
    unsuppressed_dim = size(log_con_lik_matrix)[1]
    total_dim = kernel_dim + suppression_dim + unsuppressed_dim
    n_codon_parameters = length(grid_sizes)

    return SKBDIModel(parameter_grids,
        parameter_names,
        hypothesis_masks,
        transition_function,
        log_con_lik_matrix,
        con_lik_matrix,
        codon_param_vec,
        codon_param_index_vec,
        ambient_to_parameter_transform,
        kernel_dim,
        grid_sizes,
        masks,
        mask_subset_indicators,
        suppression_dim,
        unsuppressed_dim,
        total_dim,
        n_codon_parameters)
end

"""
# reshape_probability_vector(grid_sizes::Tuple, codon_param_index_vec::Vector{Vector{Int64}}, probability_vector::AbstractVector{<:Real})
Takes a vector with index according to codon_param_index_vec 
and returns the corresponding multidimensional array with shape according to grid sizes.
"""
function reshape_probability_vector(grid_sizes::Tuple, codon_param_index_vec::Vector{Vector{Int64}}, probability_vector::AbstractVector{<:Real})
    probability_array = zeros(Float64, grid_sizes)
    for (i, grid_point) in enumerate(codon_param_index_vec)
        probability_array[grid_point...] = probability_vector[i]
    end
    return probability_array
end

"""
# unreshape_probability_vector(codon_param_index_vec::Vector{Vector{Int64}}, probability_array::AbstractArray{<:Real})
Takes a multidimensional array and returns the corresponding vector with index according to codon_param_index_vec.
"""
function unreshape_probability_vector(codon_param_index_vec::Vector{Vector{Int64}}, probability_array::AbstractArray{<:Real})
    probability_vector = zeros(Float64, length(codon_param_index_vec))
    for (i, grid_point) in enumerate(codon_param_index_vec)
        probability_vector[i] = probability_array[grid_point...]
    end
    return probability_vector
end

"""
# reshape_probability_vector(grid_sizes::Tuple, probability_vector::AbstractVector{<:Real})
Takes a vector with index according to codon_param_index_vec and reshapes it into a multidimensional array based on the grid sizes.
NOTE: Only works when the index matches con_lik_mat from difFUBAR_grid(), but is fast and AD safe
"""
function reshape_probability_vector(grid_sizes::Tuple, probability_vector::AbstractVector{<:Real})
    return permutedims(reshape(probability_vector, reverse(grid_sizes)...),
        reverse(1:length(grid_sizes)))
end

"""
# unreshape_probability_vector()
Takes a multidimensional array and returns the corresponding vector with index according to codon_param_index_vec.
NOTE: Only works when the index matches con_lik_mat from difFUBAR_grid, but is fast and AD safe
"""
function unreshape_probability_vector(grid_sizes::Tuple, probability_array::AbstractArray{<:Real})
    return vec(permutedims(probability_array, reverse(1:length(grid_sizes))))
end

"""
# split_parameters(model::SKBDIModel, parameters::AbstractVector{<:Real})
Splits the parameters into kernel, suppression, and unsuppressed parameters according to the model.
The parameters are assumed to be in the order: kernel, suppression, unsuppressed with lengths
model.kernel_dim, model.suppression_dim, model.unsuppressed_dim respectively.
"""
function split_parameters(model::SKBDIModel, parameters::AbstractVector{<:Real})

    begin
        kernel_parameters = parameters[1:model.kernel_dim]
        suppression_parameters = parameters[model.kernel_dim+1:model.kernel_dim+model.suppression_dim]
        unsuppressed_parameters = parameters[model.kernel_dim+model.suppression_dim+1:end]
    end
    return kernel_parameters, suppression_parameters, unsuppressed_parameters
end

"""
# to_probability_vector(model::SKBDIModel, ambient_sample::Vector{Float64})
Computes the probability vector for the model given the parameters (ambient_sample).
Parameters are assumed to be in the order: kernel, suppression, unsuppressed with lengths matching
model.kernel_dim, model.suppression_dim, model.unsuppressed_dim respectively.
"""
function to_probability_vector(model::SKBDIModel, ambient_sample::Vector{Float64})

    parameters = model.ambient_to_parameter_transform(ambient_sample)
    _, suppression_parameters, unsuppressed_parameters = split_parameters(model, parameters)
    probability_vector = softmax(unsuppressed_parameters)
    # Apply the transition functions for each suppression parameter
    transition_function_values = model.transition_function.(suppression_parameters)
    for i in 1:size(model.masks, 1)
        # In the case of hypothesis overlap, we take the minimum of the transition functions
        mask = model.masks[i, :]
        suppression_factor = minimum(
            transition_function_values[model.mask_subset_indicators[i, :]])
        # we are multiplying the vector by something like [s, 1, ..., s, 1, s, s, 1, 1, 1, ..., 1]
        # We do it this way because it makes AD happy :)
        probability_vector = probability_vector .* (suppression_factor .* mask .+ Float64.(.!mask))
    end
    return any(probability_vector .> 0.0) ? probability_vector ./ sum(probability_vector) : probability_vector
end

"""
# log_likelihood(model::SKBDIModel, ambient_sample::AbstractVector{<:Real})
Computes the log-likelihood of the model given the parameters (ambient_sample).
The parameters are assumed to be in the order: kernel, suppression, unsuppressed with lengths matching
model.kernel_dim, model.suppression_dim, model.unsuppressed_dim respectively.
"""
function log_likelihood(model::SKBDIModel, ambient_sample::AbstractVector{<:Real})
    probability_vector = to_probability_vector(model, ambient_sample)
    return sum(log.(model.con_lik_matrix' * probability_vector))
end

"""
# fubar_ambient_to_parameter_transform(ambient_sample::AbstractVector{<:Real}, grid_sizes::Tuple, codon_param_index_vec::Vector{Vector{Int64}}, kernel_dim::Int64, suppression_dim::Int64, kernel_stddev::Float64, suppression_stddev::Float64)
Transforms an ambient sample (~N(0, I)) into the parameter space (~N(0, Sigma)).
NOTE: This is identical to the below grid_based_ambient_to_parameter_transform, but uses the fubar_apply_smoothing function.
TODO: make this and the below function into only one function using multiple dispatch.
The main difference is the reshaping operation.
## Parameters:
ambient_sample::AbstractVector{<:Real}: The ambient sample to transform.
grid_sizes::Tuple: The sizes of the parameter grids.
codon_param_index_vec::Vector{Vector{Int64}}: The indices of the codon parameters.
kernel_dim::Int64: The dimensionality of the kernel parameters.
suppression_dim::Int64: The dimensionality of the suppression parameters.
kernel_stddev::Float64: The standard deviation for the kernel parameters.
suppression_stddev::Float64: The standard deviation for the suppression parameters.
## Returns:
AbstractVector{<:Real}: The transformed parameters with covariance structure matching the model, of the same shape as the ambient sample 
(kernel_parameters, suppression_parameters, unsuppressed_parameters).
"""
function fubar_ambient_to_parameter_transform(ambient_sample::AbstractVector{<:Real}, grid_sizes::Tuple, codon_param_index_vec::Vector{Vector{Int64}}, kernel_dim::Int64, suppression_dim::Int64, kernel_stddev::Float64, suppression_stddev::Float64)
    kernel_parameters = ambient_sample[1:kernel_dim]
    suppression_parameters = ambient_sample[kernel_dim+1:kernel_dim+suppression_dim]
    ambient_unsuppressed_parameters = ambient_sample[kernel_dim+suppression_dim+1:end]
    kernel_parameters = kernel_stddev * kernel_parameters
    suppression_parameters = suppression_stddev * suppression_parameters
    return vcat(kernel_parameters, suppression_parameters, fubar_apply_smoothing(grid_sizes, codon_param_index_vec, ambient_unsuppressed_parameters, kernel_parameters))
end

"""
# ambient_to_parameter_transform(ambient_sample::AbstractVector{<:Real}, grid_sizes::Tuple, codon_param_index_vec::Vector{Vector{Int64}}, kernel_dim::Int64, suppression_dim::Int64, kernel_stddev::Float64, suppression_stddev::Float64)
Transforms an ambient sample (~N(0, I)) into the parameter space (~N(0, Sigma)).
## Parameters:
ambient_sample::AbstractVector{<:Real}: The ambient sample to transform.
grid_sizes::Tuple: The sizes of the parameter grids.
codon_param_index_vec::Vector{Vector{Int64}}: The indices of the codon parameters.
kernel_dim::Int64: The dimensionality of the kernel parameters.
suppression_dim::Int64: The dimensionality of the suppression parameters.
kernel_stddev::Float64: The standard deviation for the kernel parameters.
suppression_stddev::Float64: The standard deviation for the suppression parameters.
## Returns:
AbstractVector{<:Real}: The transformed parameters with covariance structure matching the model, of the same shape as the ambient sample 
(kernel_parameters, suppression_parameters, unsuppressed_parameters).
"""
function grid_based_ambient_to_parameter_transform(ambient_sample::AbstractVector{<:Real}, grid_sizes::Tuple, kernel_dim::Int64, suppression_dim::Int64, kernel_stddev::Float64, suppression_stddev::Float64)

    kernel_parameters = ambient_sample[1:kernel_dim]
    suppression_parameters = ambient_sample[kernel_dim+1:kernel_dim+suppression_dim]
    ambient_unsuppressed_parameters = ambient_sample[kernel_dim+suppression_dim+1:end]
    kernel_parameters = kernel_stddev * kernel_parameters
    suppression_parameters = suppression_stddev * suppression_parameters
    return vcat(kernel_parameters, suppression_parameters, apply_smoothing(grid_sizes, ambient_unsuppressed_parameters, kernel_parameters))
    #return vcat(kernel_parameters, suppression_parameters, ambient_unsuppressed_parameters)
end

"""
# calculate_alloc_grid_and_theta(model::GeneralizedFUBARModel,ambient_samples::Vector{Vector{Float64}}, burnin::Int; progress=false)
Samples the allocation grid used for site-wise inference from the ambient samples 
(see the other diffubar files for details about how alloc_grid is used.)
TODO: Find a way to make this faster!
## Parameters:
- model::GeneralizedFUBARModel: The model to use for the transformation.
- ambient_samples::Vector{Vector{Float64}}: The ambient samples to transform.
- burnin::Int: The burn-in period for the MCMC sampler.
- progress::Bool: Whether to show progress updates.
## Returns:
- alloc_grid::Matrix{Int64}: The allocation grid for site-wise inference
- theta::Vector{Float64}: The estimated posterior probability vector for the model.
"""
function calculate_alloc_grid_and_theta(model::GeneralizedFUBARModel,
    ambient_samples::Vector{Vector{Float64}},
    burnin::Int; progress=false)

    n_samples = size(ambient_samples, 1)
    n_sites = size(model.con_lik_matrix, 2)
    alloc_grid = zeros(Int64, size(model.con_lik_matrix))
    theta = zeros(Float64, model.n_categories)
    p = progress ? Progress(n_samples - burnin; desc="Sampling allocations") : nothing
    for i = burnin+1:n_samples
        probability_vector = model.to_probability_vector(ambient_samples[i])
        theta .+= probability_vector # Accumulate the probability vector
        v = zeros(Float64, model.n_categories)
        for i = 1:n_sites
            for j = 1:model.n_categories
                v[j] = probability_vector[j] * model.con_lik_matrix[j, i]
            end
            samp = sample(1:model.n_categories, Weights(v))
            alloc_grid[samp, i] += 1 # Increment the allocation for the sampled category
        end
        progress ? next!(p) : nothing
    end
    return alloc_grid, theta ./ (n_samples - burnin)
end



"""
    skbdifFUBAR(seqnames, seqs, treestring, tags, outpath; <keyword arguments>)

Takes a tagged phylogeny and an alignment as input and performs SKBDI + difFUBAR analysis.
Returns `df, results_tuple, plots_named_tuple` where `df` is a DataFrame of the detected sites, `results_tuple` is a tuple of the partial calculations needed to re-run `difFUBAR_tabulate_and_plot`, and `plots_named_tuple` is a named tuple of plots.
Consistent with the docs of [`difFUBAR_tabulate_and_plot`](@ref), `results_tuple` stores `(alloc_grid, codon_param_vec, alphagrid, omegagrid, tag_colors)`.

# Arguments
- `seqnames`: vector of untagged sequence names.
- `seqs`: vector of aligned sequences, corresponding to `seqnames`.
- `treestring`: a tagged newick tree string.
- `tags`: vector of tag signatures.
- `outpath`: export directory.
- `sampler::String`: the sampler to use, either "ess" or "nuts". "ess" uses the ESS sampler, while "nuts" uses the NUTS sampler.
- `chains=1`: number of chains to use. Only "nuts" supports multiple chains
- `tag_colors=DIFFUBAR_TAG_COLORS[sortperm(tags)]`: vector of tag colors (hex format). The default option is consistent with the difFUBAR paper (Foreground 1: red, Foreground 2: blue).
- `pos_thresh=0.95`: threshold of significance for the posteriors.
- `iters=2500`: iterations used in the Gibbs sampler.
- `burnin=div(iters, 5)`: burnin used in the Gibbs sampler.
- `concentration=0.1`: concentration parameter used for the Dirichlet prior.
- `binarize=false`: if true, the tree is binarized before the analysis.
- `verbosity=1`: as verbosity increases, prints are added accumulatively. 
    - 0 - no prints
    - 1 - show current step and where output files are exported
    - 2 - show the chosen `difFUBAR_grid` version and amount of parallel threads. Show sampler progress bar.
- `exports=true`: if true, output files are exported.
- `exports2json=false`: if true, the results are exported to a JSON file (HyPhy format).
- `code=MolecularEvolution.universal_code`: genetic code used for the analysis.
- `optimize_branch_lengths=false`: if true, the branch lengths of the phylogenetic tree are optimized.
- `version::Union{difFUBARGrid, Nothing}=nothing`: explicitly choose the version of `difFUBAR_grid` to use. If `nothing`, the version is heuristically chosen based on the available RAM and Julia threads.
- `t=0`: explicitly choose the amount of Julia threads to use. If `0`, the degree of parallelization is heuristically chosen based on the available RAM and Julia threads.

!!! note
    Julia starts up with a single thread of execution, by default. See [Starting Julia with multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads).
"""
function skbdifFUBAR(seqnames, seqs, treestring, tags, outpath, sampler::String; n_chains=1,
    tag_colors=CodonMolecularEvolution.DIFFUBAR_TAG_COLORS[sortperm(tags)], pos_thresh=0.95, iters=5000,
    burnin::Int=div(iters, 4), concentration=0.1, binarize=false, verbosity=1,
    exports=true, exports2json=false, code=MolecularEvolution.universal_code,
    optimize_branch_lengths=false, version=nothing, t=0)
    # TODO: Split this up into multiple fcts because the setup is mostly shared with difFUBAR

    if n_chains > 1 && sampler != "nuts"
        error("Only NUTS supports multiple chains at the moment. Use sampler='nuts' to use multiple chains.")
    end

    total_time = @elapsed begin
        analysis_name = outpath
        leaf_name_transform = CodonMolecularEvolution.generate_tag_stripper(tags)
        plot_collection = NamedTuple[]
        tree, tags, tag_colors, analysis_name = CodonMolecularEvolution.difFUBAR_init(
            analysis_name, treestring, tags, tag_colors=tag_colors, exports=exports,
            verbosity=verbosity, disable_binarize=!binarize, plot_collection=plot_collection)
        ((tree, LL, alpha, beta, GTRmat, F3x4_freqs, eq_freqs), fit_time) =
            @timed CodonMolecularEvolution.difFUBAR_global_fit_2steps(
                seqnames, seqs, tree, leaf_name_transform, code, verbosity=verbosity,
                optimize_branch_lengths=optimize_branch_lengths)
        ((con_lik_matrix, log_con_lik_matrix, codon_param_vec, alphagrid, omegagrid,
                param_kinds, shallow_tree, background_omega_grid, codon_param_index_vec), grid_time) =
            @timed CodonMolecularEvolution.difFUBAR_grid(
                tree, tags, GTRmat, F3x4_freqs, code, verbosity=verbosity,
                foreground_grid=6, background_grid=4, version=version, t=t)
        # We should figure out a good prior dist for F(s)
        transition_function = s -> CodonMolecularEvolution.quintic_smooth_transition(s, -1.0, 1.0)
        # Define the masks for the suppression parameters
        # TODO: Make this an argument to this function
        hypothesis_masks = ones(Bool, (4, size(con_lik_matrix)[1]))

        hypothesis_masks[1, :] = [c[2] > 1 for c in codon_param_vec] # omega_1 > 1
        hypothesis_masks[2, :] = [c[3] > 1 for c in codon_param_vec] # omega_2 > 1s
        hypothesis_masks[3, :] = [c[2] > c[3] for c in codon_param_vec] # omega_1 > omega_2
        hypothesis_masks[4, :] = [c[3] > c[2] for c in codon_param_vec] # omega_2 > omega_1

        grid_sizes = zeros(Int64, length(codon_param_index_vec[1]))
        for i = 1:length(codon_param_index_vec[1])
            grid_sizes[i] = maximum([index[i] for index in codon_param_index_vec])
        end
        grid_sizes = tuple(grid_sizes...)
        #square_distance_matrix = generate_square_l2_distance_matrix(codon_param_index_vec)
        suppression_stddev = 1.0
        kernel_stddev = 4.0

        model = SKBDIModel(
            [alphagrid, omegagrid, background_omega_grid],
            ["alpha", "omega_1", "omega_2", "background_omega"],
            hypothesis_masks,
            transition_function,
            log_con_lik_matrix,
            con_lik_matrix,
            codon_param_vec,
            codon_param_index_vec,
            ambient_sample -> grid_based_ambient_to_parameter_transform(
                ambient_sample,
                grid_sizes,
                1,
                size(hypothesis_masks, 1),
                kernel_stddev,
                suppression_stddev),
            1,
            grid_sizes
        )

        # Checking the reshaping operations
        probability_vector = rand(length(codon_param_index_vec))
        probability_array = reshape_probability_vector(grid_sizes, probability_vector)
        new_probability_vector = unreshape_probability_vector(grid_sizes, probability_array)

        # Check if reshaping procedure works!
        for (i, index) in enumerate(codon_param_index_vec)
            if probability_vector[i] != probability_array[index...]
                print("ERROR MISMATCH AT ")
                print(index)
                print(i)
            end
        end
        if probability_vector != new_probability_vector
            println("BAD BAD BADDDD")
        end
        # One further step of abstraction skbdi -> generalized fubar
        fubar_model = GeneralizedFUBARModel(model)

        if verbosity > 0
            println("Step 4: Sampling from the model.")
        end
        ambient_samples = Vector{Any}(undef, n_chains)


        sample_time = @elapsed begin
            if sampler == "ess"
                ess_model = ESSModel(fubar_model.prior, fubar_model.log_likelihood)
                ambient_samples[1] = AbstractMCMC.sample(ess_model, ESS(), iters, progress=true)
                print(size(ambient_samples), " samples drawn from the target distribution.\n")
            elseif sampler == "nuts"
                ambient_samples, _ = sample_NUTS(fubar_model, iters, n_chains; progress=verbosity > 0)
                #println(stats)
                #ambient_samples = [ambient_samples[i] for i in 1:size(ambient_samples, 1)]
            else
                error("Unknown sampler: $sampler. Use 'ess' or 'nuts'.")
            end
            alloc_grid = zeros(Int64, size(fubar_model.con_lik_matrix))
            theta = zeros(Float64, fubar_model.n_categories)

            for i in 1:n_chains
                alloc_grid_i, theta_i = calculate_alloc_grid_and_theta(fubar_model, ambient_samples[i], burnin, progress=verbosity > 0)
                alloc_grid .+= alloc_grid_i
                theta .+= theta_i
            end
            theta ./= n_chains # Average the theta values across chains
        end
    end
    # Now we should have the same stuff that diffubar generates.
    df, plots_named_tuple = CodonMolecularEvolution.difFUBAR_tabulate_and_plot(
        analysis_name, pos_thresh, alloc_grid, codon_param_vec, alphagrid, omegagrid, tag_colors, verbosity=verbosity, exports=exports)
    json = CodonMolecularEvolution.dNdS2JSON(CodonMolecularEvolution.difFUBAR2JSON(),
        (outpath=analysis_name, df=df, θ=theta, posterior_mat=alloc_grid ./ sum(alloc_grid, dims=1),
            categories=reduce(hcat, codon_param_vec)', tags=tags, tree=shallow_tree, LL=LL,
            timers=(total_time, fit_time, grid_time, sample_time),
            treestring=treestring, seqnames=seqnames, seqs=seqs,
            leaf_name_transform=leaf_name_transform, pos_thresh=pos_thresh,
            iters=iters, burnin=burnin, concentration=concentration, binarize=binarize, exports=exports2json))
    # Return df, (tuple of partial calculations needed to re-run tablulate), plots 
    push!(plot_collection, plots_named_tuple)
    return df, (alloc_grid, ambient_samples, model, tag_colors), merge(plot_collection...)
end

struct GRIDFUBAR <: CodonMolecularEvolution.BayesianFUBARMethod end

"""
    FUBAR_analysis_grid_based(method::GRIDFUBAR, grid::FUBARGrid{T};
    analysis_name="grid_skbdi_fubar_analysis",
    volume_scaling=1.0,
    exports=true,
    verbosity=1,
    posterior_threshold=0.95,
    m=10,
    ϵ=1e-6,
    iters=1000,
    burnin=div(iters, 4)) where {T}

Perform a Fast Unconstrained Bayesian AppRoximation (FUBAR) analysis using the SKBDI (Smooth Kernel Bayesian Density Inference) approach.

# Arguments
- `method::SKBDIFUBAR`: Empty struct used for dispatch
- `grid::FUBARGrid{T}`: Grid to perform inference on

# Keywords
- `analysis_name::String="grid_skbdi_fubar_analysis"`: Name for the analysis output files and directory
- `volume_scaling::Float64=1.0`: Controls the scaling of the marginal parameter violin plots
- `exports::Bool=true`: Whether to export results to files
- `verbosity::Int=1`: Control level of output messages (0=none, higher values=more details)
- `posterior_threshold::Float64=0.95`: Posterior probability threshold for classification
- `iters::Int=1000`: Number of MCMC samples to generate
- `burnin=div(iters, 4)`: Number of initial samples to neglect when computing posterior summaries

# Returns
- A tuple containing:
    - `analysis`: DataFrame with FUBAR analysis results
    - `ambient_samples`: Posterior samples in ambient space
    - `theta`: Posterior samples of the probability vector
    - `theta_array`: Posterior samples of the reshaped probability vector as an array

# Description
This function increments upon the method found in the gaussian_fubar.jl file in src/FUBAR using grid based smoothing 
with Gaussian blur instead of Krylov subspace methods.
"""
function FUBAR_analysis_grid_based(method::GRIDFUBAR, grid::FUBARGrid{T};
    analysis_name="grid_skbdi_fubar_analysis",
    volume_scaling=1.0,
    exports=true,
    verbosity=1,
    posterior_threshold=0.95,
    m=10,
    ϵ=1e-6,
    iters=1000,
    burnin=div(iters, 4)) where {T}

    codon_param_index_vec = [[grid.alpha_ind_vec[i], grid.beta_ind_vec[i]] for i = 1:length(grid.alpha_ind_vec)]
    codon_param_vec = [[grid.alpha_vec[i], grid.beta_vec[i]] for i in 1:length(grid.alpha_vec)]
    grid_sizes = (maximum(grid.alpha_ind_vec), maximum(grid.beta_ind_vec))
    alpha_grid = sort(unique(grid.alpha_vec))
    beta_grid = sort(unique(grid.beta_vec))
    parameter_grids = [alpha_grid, beta_grid]
    hypothesis_masks = ones(Bool, (1, size(grid.cond_lik_matrix)[1]))
    hypothesis_masks[1, :] = [c[2] > c[1] for c in codon_param_vec] # beta > alpha
    transition_function = s -> CodonMolecularEvolution.quintic_smooth_transition(s, 0.0, 1.0)
    suppression_stddev = 2.0
    kernel_stddev = 4.0
    grid_dimension = length(grid.grid_values)
    model = SKBDIModel(parameter_grids,
        ["alpha", "beta"],
        hypothesis_masks,
        transition_function,
        log.(grid.cond_lik_matrix),
        grid.cond_lik_matrix,
        codon_param_vec,
        codon_param_index_vec,
        s -> fubar_ambient_to_parameter_transform(s, grid_sizes, codon_param_index_vec, 1, 1, kernel_stddev, suppression_stddev),
        1,
        grid_sizes
    )
    fubar_model = GeneralizedFUBARModel(model)
    ess_model = ESSModel(fubar_model.prior, fubar_model.log_likelihood)
    ambient_samples = AbstractMCMC.sample(ess_model, ESS(), iters, progress=true)
    theta = [fubar_model.to_probability_vector(s) for s in ambient_samples]
    theta_array = [reshape_probability_vector(grid_sizes, codon_param_index_vec, t) for t in theta]
    kernel_samples = [[s[1]] for s in ambient_samples]
    analysis = CodonMolecularEvolution.FUBAR_tabulate_from_θ(CodonMolecularEvolution.SKBDIFUBAR(), theta[burnin+1:end], kernel_samples[burnin+1:end], grid, analysis_name, posterior_threshold=posterior_threshold, volume_scaling=volume_scaling, verbosity=verbosity, exports=exports)
    return (analysis, ambient_samples, theta, theta_array)
end


"""
# hypothesis_posterior_probabilities(alloc_grid::Matrix{Int64}, codon_param_vec::Vector{Vector{Float64}})
Computes the posterior probabilities of the hypotheses given the allocation grid and codon parameter vector, for the diffubar method.
"""
function hypothesis_posterior_probabilities(alloc_grid::Matrix{Int64}, codon_param_vec::Vector{Vector{Float64}})
    ω1 = [c[2] for c in codon_param_vec]
    ω2 = [c[3] for c in codon_param_vec]
    alpha_vec = [c[1] for c in codon_param_vec]
    ω1_greater_filt = ω1 .> ω2
    ω2_greater_filt = ω2 .> ω1
    ω1_pos_filt = ω1 .> 1.0
    ω2_pos_filt = ω2 .> 1.0
    posterior_probabilities = zeros(size(alloc_grid, 2), 4)
    for site in 1:size(alloc_grid, 2)
        posterior_probabilities[site, 1] = sum(alloc_grid[ω1_pos_filt, site]) / sum(alloc_grid[:, site])
        posterior_probabilities[site, 2] = sum(alloc_grid[ω2_pos_filt, site]) / sum(alloc_grid[:, site])
        posterior_probabilities[site, 3] = sum(alloc_grid[ω1_greater_filt, site]) / sum(alloc_grid[:, site])
        posterior_probabilities[site, 4] = sum(alloc_grid[ω2_greater_filt, site]) / sum(alloc_grid[:, site])
    end
    return posterior_probabilities
end

"""
# diffubar_example()
Runs the skbdifFUBAR analysis on the tiny Ace2 dataset.
"""
function diffubar_example()
    analysis_name = "output/Ace2"
    seqnames, seqs = read_fasta("../../test/data/Ace2_tiny/Ace2_tiny_tagged.fasta")
    treestring = readlines(open("../../test/data/Ace2_tiny/tiny_tagged_no_bg.tre"))[1]
    #treestring, tags, colors = import_colored_figtree_nexus_as_tagged_tree("test/data/Ace2_tiny/Ace2_tiny_tagged_no_bg_tags_flipped.nex")
    tags = ["{G1}", "{G2}"]
    df, results = skbdifFUBAR(seqnames, seqs, treestring, tags, analysis_name, "nuts"; iters=100, n_chains=8)
    alloc_grid, ambient_samples, model, _ = results

    @save "output/alloc_grid.jld2" alloc_grid
    @save "output/ambient_samples.jld2" ambient_samples
    model_dict = Dict( # We don't save the functions, so we can load this later
        :parameter_grids => model.parameter_grids,
        :parameter_names => model.parameter_names,
        :hypothesis_masks => model.hypothesis_masks,
        :log_con_lik_matrix => model.log_con_lik_matrix,
        :con_lik_matrix => model.con_lik_matrix,
        :codon_param_vec => model.codon_param_vec,
        :codon_param_index_vec => model.codon_param_index_vec,
        :kernel_dim => model.kernel_dim,
        :masks => model.masks,
        :mask_subset_indicators => model.mask_subset_indicators,
        :suppression_dim => model.suppression_dim,
        :unsuppressed_dim => model.unsuppressed_dim,
        :total_dim => model.total_dim
    )

    @save "output/model.jld2" model_dict
end

"""
# fubar_example()
Performs FUBAR analysis using grid based smoothing on the flu dataset.
"""
function fubar_example()
    outdir = "fubar"
    seqnames, seqs = read_fasta("../../test/data/flu/flu.fasta")
    treestring = readlines("../../test/data/flu/flu.tre")[1]
    fgrid = alphabetagrid(seqnames, seqs, treestring)
    analysis, ambient_samples_fubar, theta_fubar, theta_array_fubar = FUBAR_analysis_grid_based(GRIDFUBAR(), fgrid, analysis_name=outdir * "/flu_grid_based", progress=true, iters=100000)
    alpha_values = sort(unique(fgrid.alpha_vec))
    beta_values = sort(unique(fgrid.beta_vec))
    @save "output/fubar/ambient_samples.jld2" ambient_samples_fubar
    @save "output/fubar/theta.jld2" theta_fubar
    @save "output/fubar/theta_array.jld2" theta_array_fubar
    @save "output/fubar/alpha_values.jld2" alpha_values
    @save "output/fubar/beta_values.jld2" beta_values
end

# diffubar_example()