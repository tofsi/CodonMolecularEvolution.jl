ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using CSV
using DataFrames
using JLD2
using LinearAlgebra
using Plots
using Statistics

function _plot_env_list(name::AbstractString)
    value = strip(get(ENV, name, ""))
    isempty(value) && return nothing
    return filter(item -> !isempty(item), strip.(split(value, ',')))
end

function _plot_file_slug(value::AbstractString)
    return replace(lowercase(value), r"[^a-z0-9]+" => "_")
end

function _discover_chain_prefix(result_dir::AbstractString)
    requested_prefix = strip(get(ENV, "CHAIN_FILE_PREFIX", ""))
    if !isempty(requested_prefix)
        sample_path = joinpath(result_dir, requested_prefix * "_chain_samples.jld2")
        isfile(sample_path) || error("Chain sample file was not found: $sample_path")
        return requested_prefix
    end

    suffix = "_chain_samples.jld2"
    sample_files = filter(name -> endswith(name, suffix), readdir(result_dir))
    length(sample_files) == 1 || error(
        "Expected exactly one *$suffix file in $result_dir, found $(length(sample_files)). " *
        "Set CHAIN_FILE_PREFIX when the directory contains multiple kernels.",
    )
    return first(sample_files)[1:end-length(suffix)]
end

function _load_convergence_plot_inputs(result_dir::AbstractString, prefix::AbstractString)
    sample_path = joinpath(result_dir, prefix * "_chain_samples.jld2")
    diagnostic_path = joinpath(result_dir, prefix * "_chain_diagnostics.csv")
    trace_path = joinpath(result_dir, prefix * "_sampler_trace.csv")

    for path in (sample_path, diagnostic_path, trace_path)
        isfile(path) || error("Required convergence artifact was not found: $path")
    end

    saved = JLD2.load(sample_path)
    samples = Float64.(saved["samples"])
    parameter_names = String.(saved["parameter_names"])
    burnin = Int(saved["burnin"])
    n_adapts = Int(saved["n_adapts"])
    diagnostics = DataFrame(CSV.File(diagnostic_path))
    sampler_trace = DataFrame(CSV.File(trace_path))

    n_iterations, n_parameters, n_chains = size(samples)
    length(parameter_names) == n_parameters ||
        throw(DimensionMismatch("Parameter-name count does not match the saved sample array"))
    0 <= n_adapts <= burnin < n_iterations || error(
        "Invalid saved adaptation/burn-in configuration: " *
        "n_adapts=$n_adapts, burnin=$burnin, iterations=$n_iterations",
    )

    return (
        samples=samples,
        parameter_names=parameter_names,
        burnin=burnin,
        n_adapts=n_adapts,
        n_iterations=n_iterations,
        n_chains=n_chains,
        diagnostics=diagnostics,
        sampler_trace=sampler_trace,
    )
end

function _select_convergence_parameters(inputs; top_n::Int=6, parameters=nothing)
    available = Set(inputs.parameter_names)
    if parameters !== nothing
        selected = String.(parameters)
        missing_parameters = filter(name -> name ∉ available, selected)
        isempty(missing_parameters) || error(
            "Requested parameters were not found in the saved chains: " *
            join(missing_parameters, ", "),
        )
        return unique(selected)
    end

    top_n >= 0 || throw(ArgumentError("top_n must be non-negative"))
    diagnostics = sort(inputs.diagnostics, :rhat; rev=true)
    n_worst = min(top_n, nrow(diagnostics))
    worst = String.(first(diagnostics, n_worst).parameters)

    diagnostic_names = Set(Symbol.(names(diagnostics)))
    kernel_parameters = if :parameter_group in diagnostic_names
        String.(diagnostics[diagnostics.parameter_group .== "kernel", :parameters])
    else
        filter(name -> startswith(name, "kernel_"), inputs.parameter_names)
    end

    return unique(vcat(kernel_parameters, worst))
end

function _burnin_trace_plot!(plot_object, burnin::Int)
    vspan!(plot_object, [1, burnin + 0.5]; color=:gray, alpha=0.10, label="")
    vline!(
        plot_object,
        [burnin + 0.5];
        color=:black,
        linestyle=:dash,
        linewidth=1.2,
        label="burn-in ends",
    )
    return plot_object
end

function _log_density_plot(inputs, colors)
    plot_object = plot(
        title="Log posterior by chain",
        xlabel="Iteration",
        ylabel="Log posterior density",
        legend=:outertopright,
    )
    _burnin_trace_plot!(plot_object, inputs.burnin)

    for chain in 1:inputs.n_chains
        rows = inputs.sampler_trace[inputs.sampler_trace.chain .== chain, :]
        sort!(rows, :iteration)
        plot!(
            plot_object,
            Int.(rows.iteration),
            Float64.(rows.log_density);
            color=colors[chain],
            linewidth=1.0,
            label="Chain $chain",
        )
    end
    return plot_object
end

function _parameter_trace_plot(inputs, parameter::AbstractString, colors)
    parameter_index = findfirst(==(parameter), inputs.parameter_names)
    parameter_index === nothing && error("Parameter was not found: $parameter")
    diagnostic_row = only(eachrow(inputs.diagnostics[inputs.diagnostics.parameters .== parameter, :]))
    title = string(
        replace(parameter, "_" => " "),
        " — R̂ ", round(Float64(diagnostic_row.rhat); digits=4),
        ", bulk ESS ", round(Float64(diagnostic_row.ess_bulk); digits=1),
        ", tail ESS ", round(Float64(diagnostic_row.ess_tail); digits=1),
    )

    plot_object = plot(
        title=title,
        xlabel="Iteration",
        ylabel=replace(parameter, "_" => " "),
        legend=false,
    )
    _burnin_trace_plot!(plot_object, inputs.burnin)
    iterations = collect(1:inputs.n_iterations)
    for chain in 1:inputs.n_chains
        plot!(
            plot_object,
            iterations,
            inputs.samples[:, parameter_index, chain];
            color=colors[chain],
            linewidth=1.0,
            label="Chain $chain",
        )
    end
    return plot_object, diagnostic_row
end

function _autocorrelation(values::AbstractVector{<:Real}, max_lag::Int)
    n = length(values)
    n > 1 || throw(ArgumentError("At least two retained samples are required for autocorrelation"))
    0 <= max_lag < n || throw(ArgumentError("Require 0 <= max_lag < retained sample count"))

    centered = Float64.(values) .- mean(values)
    denominator = sum(abs2, centered)
    denominator > 0 || return vcat(1.0, zeros(max_lag))
    return [
        lag == 0 ? 1.0 :
        dot(view(centered, 1:n-lag), view(centered, lag+1:n)) / denominator
        for lag in 0:max_lag
    ]
end

function _parameter_autocorrelation_plot(inputs, parameter::AbstractString, colors, max_lag::Int)
    parameter_index = findfirst(==(parameter), inputs.parameter_names)
    parameter_index === nothing && error("Parameter was not found: $parameter")
    retained_count = inputs.n_iterations - inputs.burnin
    retained_count > 1 || error("At least two retained iterations are required")
    effective_max_lag = min(max_lag, retained_count - 1)
    effective_max_lag >= 0 || throw(ArgumentError("max_lag must be non-negative"))
    lags = collect(0:effective_max_lag)
    reference = 1.96 / sqrt(retained_count)

    plot_object = plot(
        title="Retained-sample autocorrelation",
        xlabel="Lag (iterations)",
        ylabel="Autocorrelation",
        legend=false,
        ylims=(-1.0, 1.05),
    )
    hspan!(plot_object, [-reference, reference]; color=:gray, alpha=0.10, label="")
    hline!(plot_object, [0.0]; color=:black, linewidth=0.8, label="")

    for chain in 1:inputs.n_chains
        retained = inputs.samples[inputs.burnin+1:end, parameter_index, chain]
        plot!(
            plot_object,
            lags,
            _autocorrelation(retained, effective_max_lag);
            color=colors[chain],
            linewidth=1.5,
            label="Chain $chain",
        )
    end
    return plot_object, effective_max_lag
end

function plot_convergence_pilot(
    result_dir::AbstractString;
    top_n::Int=6,
    parameters=nothing,
    formats=String["png", "pdf"],
    max_lag::Int=50,
)
    result_dir = abspath(result_dir)
    isdir(result_dir) || error("Convergence result directory was not found: $result_dir")
    prefix = _discover_chain_prefix(result_dir)
    inputs = _load_convergence_plot_inputs(result_dir, prefix)
    selected_parameters = _select_convergence_parameters(
        inputs;
        top_n=top_n,
        parameters=parameters,
    )
    isempty(selected_parameters) && error("No parameters were selected for plotting")

    normalized_formats = lowercase.(String.(formats))
    all(format -> format in ("png", "pdf", "svg"), normalized_formats) ||
        throw(ArgumentError("PLOT_FORMATS supports png, pdf, and svg"))

    colors = palette(:tab10, inputs.n_chains)
    log_density_plot = _log_density_plot(inputs, colors)
    sorted_diagnostics = sort(inputs.diagnostics, :rhat; rev=true)
    primary_parameter = String(first(sorted_diagnostics.parameters))
    manifest_rows = NamedTuple[]

    for parameter in selected_parameters
        parameter_plot, diagnostic_row = _parameter_trace_plot(inputs, parameter, colors)
        autocorrelation_plot, effective_max_lag = _parameter_autocorrelation_plot(
            inputs,
            parameter,
            colors,
            max_lag,
        )
        combined_plot = plot(
            log_density_plot,
            parameter_plot,
            autocorrelation_plot;
            layout=(3, 1),
            size=(1200, 1200),
            dpi=180,
            plot_title="Convergence pilot chains",
        )

        parameter_slug = _plot_file_slug(parameter)
        output_base = joinpath(result_dir, prefix * "_chain_traces_" * parameter_slug)
        output_paths = Dict{String,String}()
        for format in normalized_formats
            output_path = output_base * "." * format
            savefig(combined_plot, output_path)
            output_paths[format] = output_path
            println("Wrote ", output_path)
        end

        if parameter == primary_parameter
            primary_base = joinpath(result_dir, prefix * "_chain_traces")
            for format in normalized_formats
                primary_path = primary_base * "." * format
                savefig(combined_plot, primary_path)
                println("Wrote ", primary_path)
            end
        end

        push!(manifest_rows, (
            parameter=parameter,
            is_primary=parameter == primary_parameter,
            rhat=Float64(diagnostic_row.rhat),
            ess_bulk=Float64(diagnostic_row.ess_bulk),
            ess_tail=Float64(diagnostic_row.ess_tail),
            acf_max_lag=effective_max_lag,
            png=get(output_paths, "png", ""),
            pdf=get(output_paths, "pdf", ""),
            svg=get(output_paths, "svg", ""),
        ))
    end

    manifest = DataFrame(manifest_rows)
    manifest_path = joinpath(result_dir, prefix * "_chain_plot_manifest.csv")
    CSV.write(manifest_path, manifest)
    println("Wrote ", manifest_path)
    return manifest
end

function plot_convergence_pilot_from_env(result_dir::AbstractString)
    top_n = parse(Int, get(ENV, "TOP_RHAT_PARAMETERS", "6"))
    max_lag = parse(Int, get(ENV, "ACF_MAX_LAG", "50"))
    parameters = _plot_env_list("PLOT_PARAMETERS")
    formats = something(_plot_env_list("PLOT_FORMATS"), String["png", "pdf"])
    return plot_convergence_pilot(
        result_dir;
        top_n=top_n,
        parameters=parameters,
        formats=formats,
        max_lag=max_lag,
    )
end

function _plot_convergence_usage()
    return "Usage: julia --project=. parameter_study/scripts/plot_convergence_pilot.jl " *
        "/path/to/simulation-result-directory"
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error(_plot_convergence_usage())
    plot_convergence_pilot_from_env(ARGS[1])
end
