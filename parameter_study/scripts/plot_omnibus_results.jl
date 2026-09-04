ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using CSV
using DataFrames
using Distributions
using Plots
using Printf
using Random
using Statistics

const DEFAULT_RESULT_KERNELS = Float64[0.25, 1.0, 2.0, 4.0]
const DEFAULT_RESULT_FORMATS = String["png", "pdf", "svg"]
const DEFAULT_CONFIDENCE_LEVEL = 0.95
const DEFAULT_BOOTSTRAP_SAMPLES = 10_000
const DEFAULT_BOOTSTRAP_SEED = 20260904
const AGGREGATE_CURVE_GRID = collect(range(0.0, 1.0; length=501))
const RESULT_TICK_FONTSIZE = 11
const RESULT_GUIDE_FONTSIZE = 14
const RESULT_TITLE_FONTSIZE = 16
const RESULT_LEGEND_FONTSIZE = 12
const OKABE_ITO_METHOD_COLORS = String[
    "#000000", # black
    "#0072B2", # blue
    "#D55E00", # vermillion
    "#009E73", # bluish green
    "#E69F00", # orange
    "#CC79A7", # reddish purple
    "#56B4E9", # sky blue
    "#F0E442", # yellow
]

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

function _result_confidence_level()
    value = parse(Float64, get(ENV, "RESULT_CONFIDENCE_LEVEL", string(DEFAULT_CONFIDENCE_LEVEL)))
    0.0 < value < 1.0 || throw(ArgumentError("RESULT_CONFIDENCE_LEVEL must be between 0 and 1"))
    return value
end

function _result_integer(name::AbstractString, default::Integer; minimum::Integer)
    value = parse(Int, get(ENV, name, string(default)))
    value >= minimum || throw(ArgumentError("$name must be at least $minimum"))
    return value
end

function _kernel_slug(kernel::Real)
    return replace(string(Float64(kernel)), "-" => "m", "." => "p")
end

function _kernel_label(kernel::Real)
    return @sprintf("%g", Float64(kernel))
end

function _simulation_number(simulation_id::AbstractString)
    matched = match(r"^sim_(\d+)_", simulation_id)
    return matched === nothing ? typemax(Int) : parse(Int, matched.captures[1])
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

function _row_value(row, name::Symbol)
    name in propertynames(row) || error("Summary row is missing $name")
    value = getproperty(row, name)
    ismissing(value) && error("Summary row has a missing $name")
    return value
end

function _optional_float(row, name::Symbol)
    name in propertynames(row) || return missing
    value = getproperty(row, name)
    return ismissing(value) ? missing : Float64(value)
end

function _curve_path(
    simulation_dir::AbstractString,
    kind::AbstractString;
    method::AbstractString="original_BAME",
    kernel=nothing,
)
    kind in ("roc", "pr") || throw(ArgumentError("kind must be roc or pr"))
    if method == "original_BAME"
        return joinpath(simulation_dir, "original_BAME_$(kind).csv")
    elseif method == "smoothFLAVOR_BAME" && kernel !== nothing
        return joinpath(simulation_dir, "kernel_stddev_$(_kernel_slug(kernel))_$(kind).csv")
    end
    throw(ArgumentError("Unsupported method/kernel combination: $method, $kernel"))
end

function _required_curve_paths(simulation_dir::AbstractString, kernels)
    paths = String[
        _curve_path(simulation_dir, "roc"),
        _curve_path(simulation_dir, "pr"),
    ]
    for kernel in kernels, kind in ("roc", "pr")
        push!(paths, _curve_path(
            simulation_dir,
            kind;
            method="smoothFLAVOR_BAME",
            kernel=kernel,
        ))
    end
    return paths
end

"""
Collect one method-level row per complete simulation and configuration.

A simulation is complete only when its summary contains Original BAME and all
requested smoothing kernels and all corresponding ROC/PR files exist. This
ensures every aggregate compares methods on exactly the same simulations.
"""
function collect_complete_results(
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
        required_columns = Set([
            :method,
            :kernel_stddev,
            :auc,
            :auprc,
            :n_sites,
            :n_true_positive_sites,
            :elapsed_seconds,
        ])
        required_columns ⊆ Set(Symbol.(names(summary))) ||
            error("Missing required columns in $summary_path")

        baseline = _single_method_row(summary, "original_BAME")
        smooth_rows = [_single_method_row(summary, "smoothFLAVOR_BAME"; kernel=kernel) for kernel in kernels]
        missing_kernels = Float64[kernel for (kernel, row) in zip(kernels, smooth_rows) if row === nothing]
        missing_curve_paths = filter(!isfile, _required_curve_paths(simulation_dir, kernels))

        included = baseline !== nothing && isempty(missing_kernels) && isempty(missing_curve_paths)
        reason = if baseline === nothing
            "missing original_BAME"
        elseif !isempty(missing_kernels)
            "missing kernels: $(join(missing_kernels, ","))"
        elseif !isempty(missing_curve_paths)
            "missing curves: $(join(basename.(missing_curve_paths), ","))"
        else
            "complete"
        end

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

        baseline_runtime = Float64(_row_value(baseline, :elapsed_seconds))

        push!(rows, (
            simulation_id=simulation_id,
            simulation_number=_simulation_number(simulation_id),
            method="original_BAME",
            kernel_stddev=missing,
            method_order=0,
            label="FLAVOR",
            n_sites=Int(_row_value(baseline, :n_sites)),
            n_true_positive_sites=Int(_row_value(baseline, :n_true_positive_sites)),
            roc_auc=Float64(_row_value(baseline, :auc)),
            pr_auc=Float64(_row_value(baseline, :auprc)),
            elapsed_seconds=baseline_runtime,
            runtime_relative_to_flavor=1.0,
            mcmc_max_rhat=missing,
            mcmc_n_rhat_above_1p01=missing,
            mcmc_min_ess_bulk=missing,
            mcmc_min_ess_tail=missing,
            mcmc_retained_numerical_errors=missing,
            mcmc_retained_max_tree_depth_hits=missing,
            mcmc_min_ebfmi=missing,
            mcmc_log_density_rhat=missing,
            mcmc_log_density_ess_bulk=missing,
            mcmc_log_density_ess_tail=missing,
        ))

        for (method_order, (kernel, smooth)) in enumerate(zip(kernels, smooth_rows))
            smooth_runtime = Float64(_row_value(smooth, :elapsed_seconds))
            push!(rows, (
                simulation_id=simulation_id,
                simulation_number=_simulation_number(simulation_id),
                method="smoothFLAVOR_BAME",
                kernel_stddev=Float64(kernel),
                method_order=method_order,
                label="smoothFLAVOR σ=$(_kernel_label(kernel))",
                n_sites=Int(_row_value(smooth, :n_sites)),
                n_true_positive_sites=Int(_row_value(smooth, :n_true_positive_sites)),
                roc_auc=Float64(_row_value(smooth, :auc)),
                pr_auc=Float64(_row_value(smooth, :auprc)),
                elapsed_seconds=smooth_runtime,
                runtime_relative_to_flavor=smooth_runtime / baseline_runtime,
                mcmc_max_rhat=_optional_float(smooth, :mcmc_max_rhat),
                mcmc_n_rhat_above_1p01=_optional_float(smooth, :mcmc_n_rhat_above_1p01),
                mcmc_min_ess_bulk=_optional_float(smooth, :mcmc_min_ess_bulk),
                mcmc_min_ess_tail=_optional_float(smooth, :mcmc_min_ess_tail),
                mcmc_retained_numerical_errors=_optional_float(smooth, :mcmc_retained_numerical_errors),
                mcmc_retained_max_tree_depth_hits=_optional_float(smooth, :mcmc_retained_max_tree_depth_hits),
                mcmc_min_ebfmi=_optional_float(smooth, :mcmc_min_ebfmi),
                mcmc_log_density_rhat=_optional_float(smooth, :mcmc_log_density_rhat),
                mcmc_log_density_ess_bulk=_optional_float(smooth, :mcmc_log_density_ess_bulk),
                mcmc_log_density_ess_tail=_optional_float(smooth, :mcmc_log_density_ess_tail),
            ))
        end
    end

    if requested !== nothing
        missing_simulations = setdiff(requested, seen)
        isempty(missing_simulations) || error(
            "Requested simulation directories were not found: $(join(sort(collect(missing_simulations)), ", "))",
        )
    end

    isempty(rows) && error("No simulations were complete for kernels $(join(kernels, ", "))")
    results = DataFrame(rows)
    sort!(results, [:simulation_number, :method_order])
    manifest = DataFrame(manifest_rows)
    sort!(manifest, :simulation_id)
    return results, manifest
end

function _bootstrap_weights(n_simulations::Int, n_bootstrap::Int, seed::Integer)
    n_simulations >= 2 || throw(ArgumentError("BCa intervals require at least two simulations"))
    rng = MersenneTwister(seed)
    weights = zeros(Float64, n_bootstrap, n_simulations)
    for bootstrap_index in 1:n_bootstrap, draw in 1:n_simulations
        simulation_index = rand(rng, 1:n_simulations)
        weights[bootstrap_index, simulation_index] += 1.0 / n_simulations
    end
    return weights
end

function _bca_bounds(
    observed::Real,
    bootstrap_statistics,
    jackknife_statistics,
    confidence_level::Real,
)
    bootstrap = Float64.(bootstrap_statistics)
    jackknife = Float64.(jackknife_statistics)
    n_bootstrap = length(bootstrap)

    n_less = count(<(observed), bootstrap)
    n_equal = count(==(observed), bootstrap)
    probability_less = (n_less + 0.5 * n_equal) / n_bootstrap
    probability_floor = 0.5 / n_bootstrap
    bias_correction = quantile(
        Normal(),
        clamp(probability_less, probability_floor, 1.0 - probability_floor),
    )

    jackknife_mean = mean(jackknife)
    differences = jackknife_mean .- jackknife
    squared_sum = sum(abs2, differences)
    acceleration = squared_sum == 0.0 ? 0.0 :
        sum(differences .^ 3) / (6.0 * squared_sum^(3 / 2))

    alpha = 1.0 - confidence_level
    tail_probabilities = (alpha / 2.0, 1.0 - alpha / 2.0)
    adjusted_probabilities = map(tail_probabilities) do probability
        normal_quantile = quantile(Normal(), probability)
        numerator = bias_correction + normal_quantile
        denominator = 1.0 - acceleration * numerator
        adjusted = cdf(Normal(), bias_correction + numerator / denominator)
        return clamp(adjusted, 0.0, 1.0)
    end
    lower, upper = quantile(bootstrap, collect(adjusted_probabilities))
    return (ci_lower=lower, ci_upper=upper)
end

function _mean_bca_summary(values, bootstrap_weights, confidence_level::Real)
    numeric = Float64.(values)
    isempty(numeric) && throw(ArgumentError("Cannot summarize an empty sample"))
    all(isfinite, numeric) || throw(ArgumentError("Cannot summarize non-finite values"))

    n = length(numeric)
    size(bootstrap_weights, 2) == n || throw(DimensionMismatch(
        "Bootstrap weights have $(size(bootstrap_weights, 2)) simulations; values have $n",
    ))
    sample_mean = mean(numeric)
    sample_sd = std(numeric; corrected=true)
    sample_se = sample_sd / sqrt(n)
    bootstrap_means = bootstrap_weights * numeric
    jackknife_means = (n * sample_mean .- numeric) ./ (n - 1)
    interval = _bca_bounds(sample_mean, bootstrap_means, jackknife_means, confidence_level)
    return merge((
        mean=sample_mean,
        standard_deviation=sample_sd,
        standard_error=sample_se,
    ), interval)
end

function summarize_aggregate_auc(
    results::DataFrame,
    bootstrap_weights;
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
)
    rows = NamedTuple[]
    for method_group in groupby(results, :method_order),
        (metric_order, (metric, metric_label)) in enumerate((
            (:roc_auc, "ROC AUC"),
            (:pr_auc, "PR AUC"),
        ))
        summary = _mean_bca_summary(method_group[!, metric], bootstrap_weights, confidence_level)
        push!(rows, merge((
            method=first(method_group.method),
            kernel_stddev=first(method_group.kernel_stddev),
            method_order=first(method_group.method_order),
            label=first(method_group.label),
            metric=String(metric),
            metric_label=metric_label,
            metric_order=metric_order,
            n_simulations=nrow(method_group),
            confidence_level=Float64(confidence_level),
            confidence_interval="two-sided BCa bootstrap interval for the mean",
            bootstrap_samples=size(bootstrap_weights, 1),
            bootstrap_seed=Int(bootstrap_seed),
        ), summary))
    end
    aggregate = DataFrame(rows)
    sort!(aggregate, [:metric_order, :method_order])
    return aggregate
end

function _clean_curve(path::AbstractString, kind::AbstractString)
    curve = DataFrame(CSV.File(path))
    x_name, y_name = kind == "roc" ? (:fpr, :tpr) : (:recall, :precision)
    required = Set([x_name, y_name])
    required ⊆ Set(Symbol.(names(curve))) || error("Missing $(join(required, ", ")) in $path")

    x = Float64.(curve[!, x_name])
    y = Float64.(curve[!, y_name])
    valid = isfinite.(x) .& isfinite.(y) .&
        (x .>= 0.0) .& (x .<= 1.0) .& (y .>= 0.0) .& (y .<= 1.0)
    any(valid) || error("No usable $kind points in $path")
    cleaned = DataFrame(x=x[valid], y=y[valid])

    # Curves can contain vertical segments. A single-valued macro-average needs
    # one ordinate per abscissa, so retain the upper point at duplicate x values.
    cleaned = combine(groupby(cleaned, :x), :y => maximum => :y)
    sort!(cleaned, :x)

    if first(cleaned.x) > 0.0
        start_y = kind == "roc" ? 0.0 : first(cleaned.y)
        pushfirst!(cleaned, (x=0.0, y=start_y))
    end
    if last(cleaned.x) < 1.0
        end_y = kind == "roc" ? 1.0 : last(cleaned.y)
        push!(cleaned, (x=1.0, y=end_y))
    end
    nrow(cleaned) >= 2 || error("At least two distinct $kind points are required in $path")
    return cleaned
end

function _interpolate_curve(curve::DataFrame, grid)
    x = curve.x
    y = curve.y
    interpolated = Vector{Float64}(undef, length(grid))
    for (grid_index, target) in enumerate(grid)
        left = searchsortedlast(x, target)
        if left == 0
            interpolated[grid_index] = first(y)
        elseif left == length(x)
            interpolated[grid_index] = last(y)
        else
            fraction = (target - x[left]) / (x[left + 1] - x[left])
            interpolated[grid_index] = y[left] + fraction * (y[left + 1] - y[left])
        end
    end
    return interpolated
end

function aggregate_curves(
    results::DataFrame,
    result_root::AbstractString,
    kind::AbstractString;
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    grid=AGGREGATE_CURVE_GRID,
    bootstrap_weights,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
)
    kind in ("roc", "pr") || throw(ArgumentError("kind must be roc or pr"))
    rows = NamedTuple[]

    for method_group in groupby(results, :method_order)
        n_simulations = nrow(method_group)
        curve_matrix = Matrix{Float64}(undef, n_simulations, length(grid))
        for (simulation_index, method_row) in enumerate(eachrow(method_group))
            path = _curve_path(
                joinpath(result_root, method_row.simulation_id),
                kind;
                method=method_row.method,
                kernel=method_row.kernel_stddev,
            )
            curve_matrix[simulation_index, :] = _interpolate_curve(_clean_curve(path, kind), grid)
        end

        size(bootstrap_weights, 2) == n_simulations || throw(DimensionMismatch(
            "Bootstrap weights have $(size(bootstrap_weights, 2)) simulations; curves have $n_simulations",
        ))
        mean_curve = vec(mean(curve_matrix; dims=1))
        sd_curve = vec(std(curve_matrix; dims=1, corrected=true))
        se_curve = sd_curve ./ sqrt(n_simulations)
        bootstrap_curves = bootstrap_weights * curve_matrix
        jackknife_curves = (n_simulations .* permutedims(mean_curve) .- curve_matrix) ./
            (n_simulations - 1)

        for (grid_index, x) in enumerate(grid)
            interval = _bca_bounds(
                mean_curve[grid_index],
                view(bootstrap_curves, :, grid_index),
                view(jackknife_curves, :, grid_index),
                confidence_level,
            )
            common = (
                method=first(method_group.method),
                kernel_stddev=first(method_group.kernel_stddev),
                method_order=first(method_group.method_order),
                label=first(method_group.label),
                n_simulations=n_simulations,
                confidence_level=Float64(confidence_level),
                bootstrap_samples=size(bootstrap_weights, 1),
                bootstrap_seed=Int(bootstrap_seed),
            )
            if kind == "roc"
                push!(rows, merge(common, (
                    fpr=Float64(x),
                    mean_tpr=mean_curve[grid_index],
                    standard_deviation_tpr=sd_curve[grid_index],
                    standard_error_tpr=se_curve[grid_index],
                    ci_lower_tpr=interval.ci_lower,
                    ci_upper_tpr=interval.ci_upper,
                )))
            else
                push!(rows, merge(common, (
                    recall=Float64(x),
                    mean_precision=mean_curve[grid_index],
                    standard_deviation_precision=sd_curve[grid_index],
                    standard_error_precision=se_curve[grid_index],
                    ci_lower_precision=interval.ci_lower,
                    ci_upper_precision=interval.ci_upper,
                )))
            end
        end
    end

    aggregate = DataFrame(rows)
    sort!(aggregate, [:method_order, kind == "roc" ? :fpr : :recall])
    return aggregate
end

function _plot_colors(n::Int)
    0 <= n <= length(OKABE_ITO_METHOD_COLORS) || throw(ArgumentError(
        "The color-blind-safe method palette supports at most $(length(OKABE_ITO_METHOD_COLORS)) methods",
    ))
    return OKABE_ITO_METHOD_COLORS[1:n]
end

function _point_offsets(n::Int)
    n >= 1 || throw(ArgumentError("At least one point is required"))
    n == 1 && return [0.0]
    n_left = cld(n, 2)
    n_right = fld(n, 2)
    left = collect(range(-0.24, -0.08; length=n_left))
    right = collect(range(0.08, 0.24; length=n_right))
    return vcat(left, right)
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

function _curve_panel(
    title::AbstractString,
    x_label::AbstractString,
    y_label::AbstractString;
    legend=false,
    titlefontsize=RESULT_TITLE_FONTSIZE,
)
    ticks = collect(0.0:0.2:1.0)
    return plot(
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        xlims=(0, 1),
        ylims=(0, 1),
        xticks=ticks,
        yticks=ticks,
        framestyle=:box,
        grid=true,
        legend=legend,
        tickfontsize=RESULT_TICK_FONTSIZE,
        guidefontsize=RESULT_GUIDE_FONTSIZE,
        titlefontsize=titlefontsize,
        legendfontsize=RESULT_LEGEND_FONTSIZE,
        left_margin=10Plots.mm,
        right_margin=4Plots.mm,
        top_margin=6Plots.mm,
        bottom_margin=8Plots.mm,
    )
end

function plot_aggregate_curves(
    aggregate::DataFrame,
    results::DataFrame,
    analysis_dir::AbstractString,
    kind::AbstractString,
    formats,
)
    kind in ("roc", "pr") || throw(ArgumentError("kind must be roc or pr"))
    method_orders = sort(unique(Int.(aggregate.method_order)))
    n_methods = length(method_orders)
    colors = _plot_colors(n_methods)
    x_name, mean_name, lower_name, upper_name = kind == "roc" ?
        (:fpr, :mean_tpr, :ci_lower_tpr, :ci_upper_tpr) :
        (:recall, :mean_precision, :ci_lower_precision, :ci_upper_precision)
    curve_title = kind == "roc" ? "Aggregate ROC curves" : "Aggregate precision-recall curves"
    plot_object = _curve_panel(
        curve_title,
        kind == "roc" ? "False-positive rate" : "Recall",
        kind == "roc" ? "True-positive rate" : "Precision";
        legend=kind == "roc" ? :bottomright : :topright,
        titlefontsize=14,
    )
    plot!(plot_object; size=(1000, 820), dpi=200)

    for (method_order, color) in zip(method_orders, colors)
        curve = aggregate[aggregate.method_order .== method_order, :]
        x = Float64.(curve[!, x_name])
        y = Float64.(curve[!, mean_name])
        lower = Float64.(curve[!, lower_name])
        upper = Float64.(curve[!, upper_name])
        plot!(
            plot_object,
            x,
            y;
            ribbon=(y .- lower, upper .- y),
            label=first(curve.label),
            color=color,
            fillcolor=color,
            fillalpha=0.08,
            linestyle=:solid,
            linewidth=2.4,
        )
        plot!(plot_object, x, lower; color=color, linestyle=:dash, linewidth=1.0, alpha=0.85, label="")
        plot!(plot_object, x, upper; color=color, linestyle=:dash, linewidth=1.0, alpha=0.85, label="")
    end

    if kind == "roc"
        plot!(plot_object, [0.0, 1.0], [0.0, 1.0]; color=:gray, linestyle=:dot, linewidth=1.5, label="Random")
    else
        baseline_rows = results[results.method_order .== 0, :]
        macro_prevalence = mean(baseline_rows.n_true_positive_sites ./ baseline_rows.n_sites)
        plot!(
            plot_object,
            [0.0, 1.0],
            [macro_prevalence, macro_prevalence];
            color=:gray,
            linestyle=:dot,
            linewidth=1.5,
            label="Mean positive-site prevalence",
        )
    end
    filename = kind == "roc" ? "aggregate_roc_curves" : "aggregate_pr_curves"
    return _save_plot(plot_object, joinpath(analysis_dir, filename), formats)
end

function plot_aggregate_auc(
    aggregate_auc::DataFrame,
    results::DataFrame,
    analysis_dir::AbstractString,
    formats,
)
    method_orders = sort(unique(Int.(aggregate_auc.method_order)))
    colors = _plot_colors(length(method_orders))
    panels = Plots.Plot[]

    for (metric_index, metric) in enumerate(("roc_auc", "pr_auc"))
        metric_rows = sort(aggregate_auc[aggregate_auc.metric .== metric, :], :method_order)
        x = collect(1:nrow(metric_rows))
        labels = [
            row.method == "original_BAME" ? "FLAVOR" : "σ=$(_kernel_label(row.kernel_stddev))"
            for row in eachrow(metric_rows)
        ]
        show_xaxis = metric_index == 2
        panel = plot(
            title=first(metric_rows.metric_label),
            xlabel=show_xaxis ? "Method / smoothFLAVOR kernel" : "",
            ylabel=first(metric_rows.metric_label),
            xlims=(0.5, length(x) + 0.5),
            ylims=(0, 1),
            xticks=(x, show_xaxis ? labels : fill("", length(labels))),
            yticks=collect(0.0:0.2:1.0),
            legend=false,
            framestyle=:box,
            grid=true,
            tickfontsize=RESULT_TICK_FONTSIZE,
            guidefontsize=RESULT_GUIDE_FONTSIZE,
            titlefontsize=RESULT_TITLE_FONTSIZE,
            left_margin=12Plots.mm,
            right_margin=5Plots.mm,
            top_margin=7Plots.mm,
            bottom_margin=show_xaxis ? 12Plots.mm : 2Plots.mm,
        )

        for (row_index, row) in enumerate(eachrow(metric_rows))
            individual = sort(
                results[results.method_order .== row.method_order, :],
                :simulation_number,
            )
            offsets = _point_offsets(nrow(individual))
            scatter!(
                panel,
                row_index .+ offsets,
                Float64.(individual[!, Symbol(metric)]);
                color=colors[row_index],
                alpha=0.35,
                marker=:circle,
                markersize=3.5,
                markerstrokewidth=0,
                label="",
            )

            lower_error = row.mean - row.ci_lower
            upper_error = row.ci_upper - row.mean
            scatter!(
                panel,
                [row_index],
                [row.mean];
                yerror=([lower_error], [upper_error]),
                color=colors[row_index],
                marker=:circle,
                markersize=7,
                markerstrokewidth=1,
                linewidth=2,
                label="",
            )
        end
        push!(panels, panel)
    end

    combined = plot(
        panels...;
        layout=(2, 1),
        link=:x,
        size=(1100, 1000),
        dpi=200,
    )
    return _save_plot(combined, joinpath(analysis_dir, "aggregate_auc"), formats)
end

function summarize_runtime_mcmc(results::DataFrame)
    metrics = (
        (:elapsed_seconds, "Runtime (seconds)"),
        (:runtime_relative_to_flavor, "Runtime relative to FLAVOR"),
        (:mcmc_max_rhat, "Maximum R-hat"),
        (:mcmc_n_rhat_above_1p01, "Parameters with R-hat > 1.01"),
        (:mcmc_min_ess_bulk, "Minimum bulk ESS"),
        (:mcmc_min_ess_tail, "Minimum tail ESS"),
        (:mcmc_retained_numerical_errors, "Retained numerical errors"),
        (:mcmc_retained_max_tree_depth_hits, "Retained maximum-tree-depth hits"),
        (:mcmc_min_ebfmi, "Minimum E-BFMI"),
        (:mcmc_log_density_rhat, "Log-density R-hat"),
        (:mcmc_log_density_ess_bulk, "Log-density bulk ESS"),
        (:mcmc_log_density_ess_tail, "Log-density tail ESS"),
    )
    rows = NamedTuple[]
    for (metric, metric_label) in metrics
        available = dropmissing(results, metric)
        for method_group in groupby(available, :method_order)
            values = Float64.(method_group[!, metric])
            all(isfinite, values) || continue
            quartiles = quantile(values, [0.25, 0.5, 0.75])
            push!(rows, (
                metric=String(metric),
                metric_label=metric_label,
                method=first(method_group.method),
                kernel_stddev=first(method_group.kernel_stddev),
                method_order=first(method_group.method_order),
                label=first(method_group.label),
                n_simulations=length(values),
                mean=mean(values),
                standard_deviation=length(values) == 1 ? 0.0 : std(values; corrected=true),
                minimum=minimum(values),
                first_quartile=quartiles[1],
                median=quartiles[2],
                third_quartile=quartiles[3],
                maximum=maximum(values),
            ))
        end
    end
    summary = DataFrame(rows)
    sort!(summary, [:metric, :method_order])
    return summary
end

function _diagnostic_method_label(row)
    return row.method == "original_BAME" ? "FLAVOR" : "σ=$(_kernel_label(row.kernel_stddev))"
end

function plot_runtime_mcmc_diagnostics(
    diagnostic_summary::DataFrame,
    results::DataFrame,
    analysis_dir::AbstractString,
    formats,
)
    colors = _plot_colors(maximum(Int.(results.method_order)) + 1)
    metric_specs = (
        (
            metric=:elapsed_seconds,
            title="Runtime",
            ylabel="Seconds",
            xlabel="Method / smoothFLAVOR kernel",
            yscale=:log10,
            transform=identity,
            threshold=nothing,
            ylims=:auto,
            yticks=:auto,
        ),
        (
            metric=:mcmc_max_rhat,
            title="Maximum R̂",
            ylabel="R̂",
            xlabel="smoothFLAVOR kernel",
            yscale=:identity,
            transform=identity,
            threshold=1.01,
            ylims=(0.98, 2.0),
            yticks=:auto,
        ),
        (
            metric=:mcmc_n_rhat_above_1p01,
            title="Parameters with R̂ > 1.01",
            ylabel="Count",
            xlabel="smoothFLAVOR kernel",
            yscale=:identity,
            transform=identity,
            threshold=nothing,
            ylims=:auto,
            yticks=:auto,
        ),
        (
            metric=:mcmc_min_ess_bulk,
            title="Minimum bulk ESS",
            ylabel="ESS",
            xlabel="smoothFLAVOR kernel",
            yscale=:log10,
            transform=identity,
            threshold=100.0,
            ylims=:auto,
            yticks=:auto,
        ),
        (
            metric=:mcmc_min_ess_tail,
            title="Minimum tail ESS",
            ylabel="ESS",
            xlabel="smoothFLAVOR kernel",
            yscale=:log10,
            transform=identity,
            threshold=100.0,
            ylims=:auto,
            yticks=:auto,
        ),
        (
            metric=:mcmc_retained_max_tree_depth_hits,
            title="Maximum-tree-depth hits",
            ylabel="Count",
            xlabel="smoothFLAVOR kernel",
            yscale=:log10,
            transform=value -> value + 1.0,
            threshold=nothing,
            ylims=(0.8, 2_000.0),
            yticks=([1.0, 2.0, 11.0, 101.0, 1_001.0], ["0", "1", "10", "100", "1000"]),
        ),
    )
    panels = Plots.Plot[]

    for spec in metric_specs
        metric_name = String(spec.metric)
        metric_summary = sort(
            diagnostic_summary[diagnostic_summary.metric .== metric_name, :],
            :method_order,
        )
        nrow(metric_summary) > 0 || continue
        x = collect(1:nrow(metric_summary))
        labels = [_diagnostic_method_label(row) for row in eachrow(metric_summary)]
        panel = plot(
            title=spec.title,
            xlabel=spec.xlabel,
            ylabel=spec.ylabel,
            xlims=(0.5, length(x) + 0.5),
            xticks=(x, labels),
            yscale=spec.yscale,
            ylims=spec.ylims,
            yticks=spec.yticks,
            legend=false,
            framestyle=:box,
            grid=true,
            tickfontsize=9,
            guidefontsize=11,
            titlefontsize=13,
            left_margin=9Plots.mm,
            right_margin=3Plots.mm,
            top_margin=5Plots.mm,
            bottom_margin=8Plots.mm,
        )

        for (x_index, row) in enumerate(eachrow(metric_summary))
            individual = dropmissing(
                results[results.method_order .== row.method_order, :],
                spec.metric,
            )
            values = spec.transform.(Float64.(individual[!, spec.metric]))
            offsets = _point_offsets(length(values))
            color = colors[Int(row.method_order) + 1]
            scatter!(
                panel,
                x_index .+ offsets,
                values;
                color=color,
                alpha=0.35,
                markersize=3.4,
                markerstrokewidth=0,
                label="",
            )

            first_quartile = spec.transform(row.first_quartile)
            median_value = spec.transform(row.median)
            third_quartile = spec.transform(row.third_quartile)
            scatter!(
                panel,
                [x_index],
                [median_value];
                yerror=(
                    [median_value - first_quartile],
                    [third_quartile - median_value],
                ),
                color=color,
                marker=:circle,
                markersize=6.5,
                markerstrokewidth=1,
                linewidth=2,
                label="",
            )
        end

        if spec.threshold !== nothing
            hline!(panel, [spec.transform(spec.threshold)]; color=:gray, linestyle=:dash, linewidth=1.2, label="")
        end
        push!(panels, panel)
    end

    combined = plot(
        panels...;
        layout=(2, 3),
        size=(1550, 1000),
        dpi=200,
    )
    return _save_plot(
        combined,
        joinpath(analysis_dir, "aggregate_runtime_mcmc_diagnostics"),
        formats,
    )
end

function generate_omnibus_results(
    result_root::AbstractString;
    analysis_dir::AbstractString=joinpath(result_root, "publication_results"),
    kernels=DEFAULT_RESULT_KERNELS,
    requested_simulations=nothing,
    formats=DEFAULT_RESULT_FORMATS,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
)
    result_root = abspath(result_root)
    analysis_dir = abspath(analysis_dir)
    isdir(result_root) || error("Result root was not found: $result_root")
    0.0 < confidence_level < 1.0 || throw(ArgumentError("confidence_level must be between 0 and 1"))
    mkpath(analysis_dir)

    results, manifest = collect_complete_results(
        result_root;
        kernels=Float64.(kernels),
        requested_simulations=requested_simulations,
    )
    n_simulations = length(unique(results.simulation_id))
    n_simulations >= 2 || error("Aggregate confidence intervals require at least two complete simulations")
    bootstrap_samples >= 1_000 || throw(ArgumentError("bootstrap_samples must be at least 1000"))
    bootstrap_weights = _bootstrap_weights(n_simulations, bootstrap_samples, bootstrap_seed)
    aggregate_auc = summarize_aggregate_auc(
        results,
        bootstrap_weights;
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    aggregate_roc = aggregate_curves(
        results,
        result_root,
        "roc";
        confidence_level=confidence_level,
        bootstrap_weights=bootstrap_weights,
        bootstrap_seed=bootstrap_seed,
    )
    aggregate_pr = aggregate_curves(
        results,
        result_root,
        "pr";
        confidence_level=confidence_level,
        bootstrap_weights=bootstrap_weights,
        bootstrap_seed=bootstrap_seed,
    )
    runtime_mcmc_summary = summarize_runtime_mcmc(results)

    CSV.write(joinpath(analysis_dir, "analysis_manifest.csv"), manifest)
    CSV.write(joinpath(analysis_dir, "per_simulation_auc.csv"), results)
    CSV.write(joinpath(analysis_dir, "aggregate_auc_summary.csv"), aggregate_auc)
    CSV.write(joinpath(analysis_dir, "aggregate_roc_curves.csv"), aggregate_roc)
    CSV.write(joinpath(analysis_dir, "aggregate_pr_curves.csv"), aggregate_pr)
    CSV.write(joinpath(analysis_dir, "aggregate_runtime_mcmc_summary.csv"), runtime_mcmc_summary)

    plot_aggregate_curves(aggregate_roc, results, analysis_dir, "roc", formats)
    plot_aggregate_curves(aggregate_pr, results, analysis_dir, "pr", formats)
    plot_aggregate_auc(aggregate_auc, results, analysis_dir, formats)
    plot_runtime_mcmc_diagnostics(runtime_mcmc_summary, results, analysis_dir, formats)

    included_simulations = unique(String.(results.simulation_id))
    println("Included ", length(included_simulations), " simulations: ", join(included_simulations, ", "))
    println("Kernels: ", join(sort(unique(skipmissing(results.kernel_stddev))), ", "))
    println(@sprintf(
        "Uncertainty: %.0f%% two-sided BCa bootstrap confidence intervals across simulations (%d resamples; seed %d)",
        100 * confidence_level,
        bootstrap_samples,
        bootstrap_seed,
    ))
    println("Wrote aggregate results to: ", analysis_dir)
    return (
        per_simulation=results,
        manifest=manifest,
        aggregate_auc=aggregate_auc,
        aggregate_roc=aggregate_roc,
        aggregate_pr=aggregate_pr,
        runtime_mcmc_summary=runtime_mcmc_summary,
    )
end

function generate_omnibus_results_from_env(result_root::AbstractString)
    kernels = _result_float_list("RESULT_KERNEL_STDDEVS", DEFAULT_RESULT_KERNELS)
    simulations = _result_list("RESULT_SIMULATIONS")
    formats = _result_formats()
    confidence_level = _result_confidence_level()
    bootstrap_samples = _result_integer(
        "RESULT_BOOTSTRAP_SAMPLES",
        DEFAULT_BOOTSTRAP_SAMPLES;
        minimum=1_000,
    )
    bootstrap_seed = _result_integer(
        "RESULT_BOOTSTRAP_SEED",
        DEFAULT_BOOTSTRAP_SEED;
        minimum=0,
    )
    analysis_dir = get(ENV, "RESULT_ANALYSIS_DIR", joinpath(result_root, "publication_results"))
    return generate_omnibus_results(
        result_root;
        analysis_dir=analysis_dir,
        kernels=kernels,
        requested_simulations=simulations,
        formats=formats,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error(
        "Usage: julia --project=. parameter_study/scripts/plot_omnibus_results.jl /path/to/result-root",
    )
    generate_omnibus_results_from_env(ARGS[1])
end
