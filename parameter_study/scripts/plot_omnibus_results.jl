ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using CSV
using DataFrames
using Plots
using Printf
using Statistics

const DEFAULT_RESULT_KERNELS = Float64[0.25, 1.0, 2.0, 4.0]
const DEFAULT_RESULT_FORMATS = String["png", "pdf", "svg"]

function _result_list(name::AbstractString)
    value = strip(get(ENV, name, ""))
    isempty(value) && return nothing
    return filter(!isempty, strip.(split(value, ',')))
end

function _result_float_list(name::AbstractString, default)
    values = _result_list(name)
    return values === nothing ? Float64.(default) : parse.(Float64, values)
end

function _result_formats()
    formats = lowercase.(something(_result_list("PLOT_FORMATS"), DEFAULT_RESULT_FORMATS))
    all(format -> format in ("png", "pdf", "svg"), formats) ||
        throw(ArgumentError("PLOT_FORMATS supports png, pdf, and svg"))
    return unique(formats)
end

function _kernel_slug(kernel::Real)
    return replace(string(Float64(kernel)), "-" => "m", "." => "p")
end

function _simulation_number(simulation_id::AbstractString)
    matched = match(r"^sim_(\d+)_", simulation_id)
    return matched === nothing ? typemax(Int) : parse(Int, matched.captures[1])
end

function _simulation_label(simulation_id::AbstractString)
    number = _simulation_number(simulation_id)
    return number == typemax(Int) ? replace(simulation_id, '_' => ' ') : "Simulation $number"
end

function _simulation_dirs(result_root::AbstractString)
    dirs = filter(
        path -> isdir(path) && isfile(joinpath(path, "method_sweep_summary.csv")),
        joinpath.(result_root, readdir(result_root)),
    )
    sort!(dirs; by=path -> (_simulation_number(basename(path)), basename(path)))
    return dirs
end

function _method_mask(df::DataFrame, method::AbstractString; kernel=nothing)
    mask = String.(df.method) .== method
    if kernel !== nothing
        mask .&= .!ismissing.(df.kernel_stddev)
        numeric_kernel = [ismissing(value) ? NaN : Float64(value) for value in df.kernel_stddev]
        mask .&= numeric_kernel .== Float64(kernel)
    end
    return mask
end

function _single_method_row(df::DataFrame, method::AbstractString; kernel=nothing)
    rows = df[_method_mask(df, method; kernel=kernel), :]
    nrow(rows) == 1 || return nothing
    return only(eachrow(rows))
end

function _row_value(row, name::Symbol, default=missing)
    name in propertynames(row) || return default
    value = getproperty(row, name)
    return ismissing(value) ? default : value
end

function _read_optional_row(path::AbstractString)
    isfile(path) || return nothing
    df = DataFrame(CSV.File(path))
    nrow(df) == 1 || error("Expected one summary row in $path; found $(nrow(df))")
    return only(eachrow(df))
end

function _diagnostics(simulation_dir::AbstractString, kernel::Real, summary_row)
    prefix = joinpath(simulation_dir, "kernel_stddev_$(_kernel_slug(kernel))")
    chain_row = _read_optional_row(prefix * "_chain_diagnostics_summary.csv")
    sampler_row = _read_optional_row(prefix * "_sampler_diagnostics_summary.csv")

    first_available(name, fallback=missing) = begin
        value = _row_value(summary_row, name, missing)
        !ismissing(value) && return value
        value = chain_row === nothing ? missing : _row_value(chain_row, name, missing)
        !ismissing(value) && return value
        value = sampler_row === nothing ? missing : _row_value(sampler_row, name, missing)
        return ismissing(value) ? fallback : value
    end

    return (
        mcmc_iterations=first_available(:mcmc_iterations, first_available(:iterations_per_chain)),
        mcmc_burnin=first_available(:mcmc_burnin, first_available(:burnin)),
        mcmc_n_adapts=first_available(:mcmc_n_adapts, first_available(:n_adapts)),
        mcmc_n_chains=first_available(:mcmc_n_chains, first_available(:n_chains)),
        mcmc_retained_per_chain=first_available(:mcmc_retained_per_chain, first_available(:retained_per_chain)),
        mcmc_max_rhat=first_available(:mcmc_max_rhat, first_available(:max_rhat)),
        mcmc_n_rhat_above_1p01=first_available(:mcmc_n_rhat_above_1p01, first_available(:n_rhat_above_1p01)),
        mcmc_min_ess_bulk=first_available(:mcmc_min_ess_bulk, first_available(:min_ess_bulk)),
        mcmc_min_ess_tail=first_available(:mcmc_min_ess_tail, first_available(:min_ess_tail)),
        mcmc_retained_numerical_errors=first_available(
            :mcmc_retained_numerical_errors,
            first_available(:total_retained_numerical_errors),
        ),
        mcmc_retained_max_tree_depth_hits=first_available(
            :mcmc_retained_max_tree_depth_hits,
            first_available(:total_retained_max_tree_depth_hits),
        ),
        mcmc_min_ebfmi=first_available(:mcmc_min_ebfmi, first_available(:min_retained_ebfmi)),
        mcmc_log_density_rhat=first_available(:mcmc_log_density_rhat, first_available(:log_density_rhat)),
        mcmc_log_density_ess_bulk=first_available(
            :mcmc_log_density_ess_bulk,
            first_available(:log_density_ess_bulk),
        ),
    )
end

function collect_paired_results(
    result_root::AbstractString;
    kernels=DEFAULT_RESULT_KERNELS,
    requested_simulations=nothing,
)
    requested = requested_simulations === nothing ? nothing : Set(String.(requested_simulations))
    rows = NamedTuple[]
    manifest_rows = NamedTuple[]
    seen = Set{String}()

    for simulation_dir in _simulation_dirs(result_root)
        simulation_id = basename(simulation_dir)
        requested !== nothing && simulation_id ∉ requested && continue
        push!(seen, simulation_id)

        summary_path = joinpath(simulation_dir, "method_sweep_summary.csv")
        summary = DataFrame(CSV.File(summary_path))
        required_columns = Set([:method, :kernel_stddev, :auc, :auprc])
        required_columns ⊆ Set(Symbol.(names(summary))) ||
            error("Missing required columns in $summary_path")

        baseline = _single_method_row(summary, "original_BAME")
        smooth_rows = [_single_method_row(summary, "smoothFLAVOR_BAME"; kernel=kernel) for kernel in kernels]
        missing_kernels = Float64[kernel for (kernel, row) in zip(kernels, smooth_rows) if row === nothing]
        included = baseline !== nothing && isempty(missing_kernels)
        reason = baseline === nothing ? "missing original_BAME" :
            isempty(missing_kernels) ? "complete" : "missing kernels: $(join(missing_kernels, ","))"

        push!(manifest_rows, (
            simulation_id=simulation_id,
            included=included,
            reason=reason,
            requested_kernels=join(Float64.(kernels), ","),
            summary_path=summary_path,
        ))

        if requested !== nothing && !included
            error("Requested simulation $simulation_id is incomplete: $reason")
        end
        included || continue

        for (kernel, smooth) in zip(kernels, smooth_rows)
            diagnostics = _diagnostics(simulation_dir, kernel, smooth)
            push!(rows, merge((
                simulation_id=simulation_id,
                simulation_number=_simulation_number(simulation_id),
                kernel_stddev=Float64(kernel),
                n_sites=Int(_row_value(smooth, :n_sites)),
                n_true_positive_sites=Int(_row_value(smooth, :n_true_positive_sites)),
                original_auc=Float64(baseline.auc),
                smooth_auc=Float64(smooth.auc),
                delta_auc=Float64(smooth.auc - baseline.auc),
                original_auprc=Float64(baseline.auprc),
                smooth_auprc=Float64(smooth.auprc),
                delta_auprc=Float64(smooth.auprc - baseline.auprc),
                pos_thresh=Float64(_row_value(smooth, :pos_thresh)),
                tpr_at_pos_thresh=Float64(_row_value(smooth, :tpr_at_pos_thresh)),
                fpr_at_pos_thresh=Float64(_row_value(smooth, :fpr_at_pos_thresh)),
                precision_at_pos_thresh=Float64(_row_value(smooth, :precision_at_pos_thresh)),
                n_called_at_pos_thresh=Int(_row_value(smooth, :n_called_at_pos_thresh)),
                elapsed_seconds=Float64(_row_value(smooth, :elapsed_seconds)),
            ), diagnostics))
        end
    end

    if requested !== nothing
        missing_simulations = setdiff(requested, seen)
        isempty(missing_simulations) || error(
            "Requested simulation directories were not found: $(join(sort(collect(missing_simulations)), ", "))",
        )
    end

    isempty(rows) && error("No simulations were complete for kernels $(join(kernels, ", "))")
    paired = DataFrame(rows)
    sort!(paired, [:simulation_number, :kernel_stddev])
    manifest = DataFrame(manifest_rows)
    sort!(manifest, :simulation_id)
    return paired, manifest
end

function summarize_kernel_effects(paired::DataFrame)
    summaries = NamedTuple[]
    for group in groupby(paired, :kernel_stddev)
        delta_auc = Float64.(group.delta_auc)
        delta_auprc = Float64.(group.delta_auprc)
        push!(summaries, (
            kernel_stddev=only(unique(group.kernel_stddev)),
            n_simulations=nrow(group),
            median_delta_auc=median(delta_auc),
            min_delta_auc=minimum(delta_auc),
            max_delta_auc=maximum(delta_auc),
            n_auc_improved=count(>(0), delta_auc),
            median_delta_auprc=median(delta_auprc),
            min_delta_auprc=minimum(delta_auprc),
            max_delta_auprc=maximum(delta_auprc),
            n_auprc_improved=count(>(0), delta_auprc),
        ))
    end
    summary = DataFrame(summaries)
    sort!(summary, :kernel_stddev)
    return summary
end

function _plot_colors(n::Int)
    return palette(:tab10, max(n, 3))[1:n]
end

function _kernel_ticks(kernels)
    values = sort(unique(Float64.(kernels)))
    return values, string.(values)
end

function _save_plot(plot_object, output_base::AbstractString, formats)
    paths = String[]
    for format in formats
        path = output_base * "." * format
        savefig(plot_object, path)
        push!(paths, path)
        println("Wrote ", path)
    end
    return paths
end

function plot_paired_performance(paired::DataFrame, analysis_dir::AbstractString, formats)
    simulation_ids = unique(String.(paired.simulation_id))
    colors = _plot_colors(length(simulation_ids))
    ticks = _kernel_ticks(paired.kernel_stddev)

    function metric_plot(column::Symbol, ylabel::AbstractString)
        plot_object = plot(
            xlabel="σ",
            ylabel=ylabel,
            xscale=:log2,
            xticks=ticks,
            legend=:outertopright,
            grid=true,
            framestyle=:box,
            left_margin=12Plots.mm,
            bottom_margin=10Plots.mm,
        )
        hline!(plot_object, [0.0]; color=:black, linewidth=1.0, label="")
        for (simulation_id, color) in zip(simulation_ids, colors)
            rows = sort(paired[paired.simulation_id .== simulation_id, :], :kernel_stddev)
            plot!(
                plot_object,
                rows.kernel_stddev,
                rows[!, column];
                label=_simulation_label(simulation_id),
                color=color,
                marker=:circle,
                markersize=5,
                linewidth=2,
            )
        end
        return plot_object
    end

    auc_plot = metric_plot(:delta_auc, "ΔROC AUC")
    auprc_plot = metric_plot(:delta_auprc, "ΔPR AUC")
    combined = plot(
        auc_plot,
        auprc_plot;
        layout=(2, 1),
        size=(1000, 1150),
        dpi=200,
    )
    return _save_plot(combined, joinpath(analysis_dir, "paired_performance_differences"), formats)
end

function _curve_path(simulation_dir::AbstractString, kind::AbstractString; kernel=nothing)
    if kernel === nothing
        return joinpath(simulation_dir, "original_BAME_$(kind).csv")
    end
    return joinpath(simulation_dir, "kernel_stddev_$(_kernel_slug(kernel))_$(kind).csv")
end

function plot_curves_by_simulation(
    paired::DataFrame,
    result_root::AbstractString,
    analysis_dir::AbstractString,
    kind::AbstractString,
    formats,
)
    kind in ("roc", "pr") || throw(ArgumentError("kind must be roc or pr"))
    simulation_ids = unique(String.(paired.simulation_id))
    kernels = sort(unique(Float64.(paired.kernel_stddev)))
    colors = _plot_colors(length(kernels))
    panels = Plots.Plot[]

    for simulation_id in simulation_ids
        simulation_dir = joinpath(result_root, simulation_id)
        panel = plot(
            title=_simulation_label(simulation_id),
            xlims=(0, 1),
            ylims=(0, 1),
            framestyle=:box,
            grid=true,
            legend=:bottomright,
            legendfontsize=7,
            xlabel=kind == "roc" ? "False-positive rate" : "Recall",
            ylabel=kind == "roc" ? "True-positive rate" : "Precision",
        )

        baseline_path = _curve_path(simulation_dir, kind)
        isfile(baseline_path) || error("Missing curve file: $baseline_path")
        baseline = DataFrame(CSV.File(baseline_path))
        x_name, y_name = kind == "roc" ? (:fpr, :tpr) : (:recall, :precision)
        baseline = sort(baseline, x_name)
        plot!(
            panel,
            Float64.(baseline[!, x_name]),
            Float64.(baseline[!, y_name]);
            label="FLAVOR",
            color=:black,
            linestyle=:solid,
            linewidth=2,
        )

        for (kernel, color) in zip(kernels, colors)
            path = _curve_path(simulation_dir, kind; kernel=kernel)
            isfile(path) || error("Missing curve file: $path")
            curve = sort(DataFrame(CSV.File(path)), x_name)
            plot!(
                panel,
                Float64.(curve[!, x_name]),
                Float64.(curve[!, y_name]);
                label="σ=$(kernel)",
                color=color,
                linewidth=1.8,
            )
        end

        if kind == "roc"
            plot!(panel, [0.0, 1.0], [0.0, 1.0]; color=:gray, linestyle=:dot, label="")
        else
            prevalence = only(unique(paired[paired.simulation_id .== simulation_id, :].n_true_positive_sites ./ paired[paired.simulation_id .== simulation_id, :].n_sites))
            hline!(panel, [prevalence]; color=:gray, linestyle=:dot, label="")
        end
        push!(panels, panel)
    end

    combined = plot(
        panels...;
        layout=(length(panels), 1),
        size=(900, 500 * length(panels)),
        dpi=200,
    )
    filename = kind == "roc" ? "roc_curves_by_simulation" : "pr_curves_by_simulation"
    return _save_plot(combined, joinpath(analysis_dir, filename), formats)
end

function plot_mcmc_diagnostics(paired::DataFrame, analysis_dir::AbstractString, formats)
    complete = dropmissing(
        paired,
        [:mcmc_max_rhat, :mcmc_min_ess_bulk, :mcmc_n_rhat_above_1p01],
    )
    nrow(complete) == nrow(paired) || @warn "Some MCMC diagnostics were unavailable; diagnostic plots use complete rows only"
    nrow(complete) > 0 || return String[]

    simulation_ids = unique(String.(complete.simulation_id))
    colors = _plot_colors(length(simulation_ids))
    ticks = _kernel_ticks(complete.kernel_stddev)

    rhat_plot = plot(
        xlabel="σ",
        ylabel="Maximum R̂",
        xscale=:log2,
        xticks=ticks,
        legend=:outertopright,
        framestyle=:box,
        grid=true,
        bottom_margin=7Plots.mm,
    )
    hline!(rhat_plot, [1.01]; color=:black, linestyle=:dash, label="")

    ess_plot = plot(
        xlabel="σ",
        ylabel="Minimum bulk ESS",
        xscale=:log2,
        yscale=:log10,
        xticks=ticks,
        legend=:outertopright,
        framestyle=:box,
        grid=true,
        bottom_margin=7Plots.mm,
    )
    hline!(ess_plot, [100.0]; color=:black, linestyle=:dash, label="")

    for (simulation_id, color) in zip(simulation_ids, colors)
        rows = sort(complete[complete.simulation_id .== simulation_id, :], :kernel_stddev)
        label = _simulation_label(simulation_id)
        plot!(rhat_plot, rows.kernel_stddev, Float64.(rows.mcmc_max_rhat); label=label, color=color, marker=:circle, linewidth=2)
        plot!(ess_plot, rows.kernel_stddev, Float64.(rows.mcmc_min_ess_bulk); label=label, color=color, marker=:circle, linewidth=2)
    end

    combined = plot(
        rhat_plot,
        ess_plot;
        layout=(2, 1),
        size=(900, 1100),
        dpi=200,
    )
    return _save_plot(combined, joinpath(analysis_dir, "mcmc_diagnostics"), formats)
end

function generate_omnibus_results(
    result_root::AbstractString;
    analysis_dir::AbstractString=joinpath(result_root, "publication_results"),
    kernels=DEFAULT_RESULT_KERNELS,
    requested_simulations=nothing,
    formats=DEFAULT_RESULT_FORMATS,
)
    result_root = abspath(result_root)
    analysis_dir = abspath(analysis_dir)
    isdir(result_root) || error("Result root was not found: $result_root")
    mkpath(analysis_dir)

    paired, manifest = collect_paired_results(
        result_root;
        kernels=Float64.(kernels),
        requested_simulations=requested_simulations,
    )
    kernel_summary = summarize_kernel_effects(paired)

    CSV.write(joinpath(analysis_dir, "analysis_manifest.csv"), manifest)
    CSV.write(joinpath(analysis_dir, "paired_performance.csv"), paired)
    CSV.write(joinpath(analysis_dir, "kernel_effect_summary.csv"), kernel_summary)

    plot_paired_performance(paired, analysis_dir, formats)
    plot_curves_by_simulation(paired, result_root, analysis_dir, "roc", formats)
    plot_curves_by_simulation(paired, result_root, analysis_dir, "pr", formats)
    plot_mcmc_diagnostics(paired, analysis_dir, formats)

    println("Included simulations: ", join(unique(paired.simulation_id), ", "))
    println("Kernels: ", join(sort(unique(paired.kernel_stddev)), ", "))
    println("Wrote reproducible results to: ", analysis_dir)
    return (paired=paired, manifest=manifest, kernel_summary=kernel_summary)
end

function generate_omnibus_results_from_env(result_root::AbstractString)
    kernels = _result_float_list("RESULT_KERNEL_STDDEVS", DEFAULT_RESULT_KERNELS)
    simulations = _result_list("RESULT_SIMULATIONS")
    formats = _result_formats()
    analysis_dir = get(ENV, "RESULT_ANALYSIS_DIR", joinpath(result_root, "publication_results"))
    return generate_omnibus_results(
        result_root;
        analysis_dir=analysis_dir,
        kernels=kernels,
        requested_simulations=simulations,
        formats=formats,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error(
        "Usage: julia --project=. parameter_study/scripts/plot_omnibus_results.jl /path/to/result-root",
    )
    generate_omnibus_results_from_env(ARGS[1])
end
