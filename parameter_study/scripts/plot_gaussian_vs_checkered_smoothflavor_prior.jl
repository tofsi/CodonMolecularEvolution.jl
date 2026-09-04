using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

Pkg.activate(REPO_ROOT)
using CodonMolecularEvolution
using LinearAlgebra
using Plots
using Statistics

const OUTDIR = joinpath(@__DIR__, "..", "results", "smoothflavor_prior_gaussian_slice")
const REPORT_SIGMA = 1.0
const SIGMA_VALUES = collect(0.0:0.05:4.0)
const GAUSSIAN_GRID_STDDEV = 2.0
const CHECKERBOARD_LOW_MULTIPLIER = 0.05
const DENSITY_AXIS_SCALE = 1_000.0
const COVARIANCE_JITTER = 1e-6
const COVARIANCE_REPORT_SIGMA = 2.0

function flavor_grids()
    tr(x) = max(0.0, 10^x - 0.05)
    trinv(x) = log10(x + 0.05)

    mugrid = CodonMolecularEvolution.gridsetup(0.01, 16.0, 8, trinv, tr)
    kappagrid = CodonMolecularEvolution.gridsetup(0.05, 20.0, 6, trinv, tr)
    alphagrid = CodonMolecularEvolution.gridsetup(0.01, 10.0, 8, trinv, tr)
    return mugrid, kappagrid, alphagrid
end

function gaussian_surface(grid_sizes; checkered=false)
    n_mu, _, n_alpha, _ = grid_sizes
    center_mu = (n_mu + 1) / 2
    center_alpha = (n_alpha + 1) / 2
    surface = Matrix{Float64}(undef, n_mu, n_alpha)

    for i in axes(surface, 1), k in axes(surface, 2)
        squared_radius = ((i - center_mu) / GAUSSIAN_GRID_STDDEV)^2 +
                         ((k - center_alpha) / GAUSSIAN_GRID_STDDEV)^2
        checker_multiplier = !checkered || iseven(i + k) ? 1.0 : CHECKERBOARD_LOW_MULTIPLIER
        surface[i, k] = exp(-0.5 * squared_radius) * checker_multiplier
    end
    return surface
end

function probability_vector_from_surface(surface, grid_sizes)
    weights = Array{Float64}(undef, grid_sizes)
    for i in axes(weights, 1), j in axes(weights, 2), k in axes(weights, 3), c in axes(weights, 4)
        weights[i, j, k, c] = surface[i, k]
    end
    return vec(permutedims(weights, (3, 2, 1, 4))) ./ sum(weights)
end

function conditional_mu_alpha_slice(theta, grid_sizes, kappa_index, capped_index)
    theta_grid = reshape(theta, grid_sizes[3], grid_sizes[2], grid_sizes[1], grid_sizes[4])
    slice = permutedims(theta_grid[:, kappa_index, :, capped_index], (2, 1))
    return slice ./ sum(slice)
end

function edge_to_peak_ratio(surface)
    edge_values = vcat(surface[1, :], surface[end, :], surface[:, 1], surface[:, end])
    return maximum(edge_values) / maximum(surface)
end

function covariance_smoothing_factor(n, sigma)
    return CodonMolecularEvolution.gaussian_covariance_factor(
        n,
        sigma^2;
        jitter=COVARIANCE_JITTER,
    )
end

function legacy_smoothing_factor(n, sigma)
    kernel = CodonMolecularEvolution.gaussian_kernel(5, sigma^2)
    basis = Matrix{Float64}(I, n, n)
    return hcat([
        CodonMolecularEvolution.same_conv(view(basis, :, column), kernel)
        for column in axes(basis, 2)
    ]...)
end

function invert_smoothing(parameters, grid_sizes, k_mu, k_alpha)
    scheme = CodonMolecularEvolution.FLAVORReshapingScheme(grid_sizes)
    parameter_array = CodonMolecularEvolution.reshape_probability_vector(scheme, parameters)
    ambient_array = similar(parameter_array)

    for shape_index in axes(parameter_array, 2), capped_index in axes(parameter_array, 4)
        block = parameter_array[:, shape_index, :, capped_index]
        ambient_array[:, shape_index, :, capped_index] .= k_mu \ block / transpose(k_alpha)
    end

    return CodonMolecularEvolution.unreshape_probability_vector(scheme, ambient_array)
end

function apply_smoothing_factors(parameters, grid_sizes, mu_factor, alpha_factor)
    scheme = CodonMolecularEvolution.FLAVORReshapingScheme(grid_sizes)
    parameter_array = CodonMolecularEvolution.reshape_probability_vector(scheme, parameters)
    smoothed_array = similar(parameter_array)

    for shape_index in axes(parameter_array, 2), capped_index in axes(parameter_array, 4)
        block = parameter_array[:, shape_index, :, capped_index]
        smoothed_array[:, shape_index, :, capped_index] .=
            mu_factor * block * transpose(alpha_factor)
    end

    return CodonMolecularEvolution.unreshape_probability_vector(scheme, smoothed_array)
end

function smoothing_geometry(grid_sizes, sigma, factor_function)
    mu_factor = factor_function(grid_sizes[1], sigma)
    alpha_factor = factor_function(grid_sizes[3], sigma)
    ambient_ones = invert_smoothing(
        ones(prod(grid_sizes)),
        grid_sizes,
        mu_factor,
        alpha_factor,
    )
    ones_precision_norm = dot(ambient_ones, ambient_ones)

    logdet_mu, sign_mu = logabsdet(mu_factor)
    logdet_alpha, sign_alpha = logabsdet(alpha_factor)
    iszero(sign_mu) && error("Singular mu smoothing matrix at sigma=$sigma")
    iszero(sign_alpha) && error("Singular alpha smoothing matrix at sigma=$sigma")

    n_blocks = grid_sizes[2] * grid_sizes[4]
    logabsdet_smoothing = n_blocks * (
        grid_sizes[3] * logdet_mu + grid_sizes[1] * logdet_alpha
    )
    # For ALR contrast matrix B, det(B * Sigma * B') = det(Sigma) * (1' * inv(Sigma) * 1).
    logdet_alr_covariance = 2 * logabsdet_smoothing + log(ones_precision_norm)

    return (
        mu_factor=mu_factor,
        alpha_factor=alpha_factor,
        ambient_ones=ambient_ones,
        ones_precision_norm=ones_precision_norm,
        logdet_alr_covariance=logdet_alr_covariance,
        covariance_condition_number=cond(mu_factor * transpose(mu_factor)),
        minimum_marginal_variance=minimum(diag(mu_factor * transpose(mu_factor))),
    )
end

function logistic_normal_log_density(theta, grid_sizes, geometry)
    log_theta = log.(theta)
    centered_logits = log_theta .- mean(log_theta)
    ambient_logits = invert_smoothing(
        centered_logits,
        grid_sizes,
        geometry.mu_factor,
        geometry.alpha_factor,
    )

    unprojected_quadratic = dot(ambient_logits, ambient_logits)
    additive_cross_term = dot(geometry.ambient_ones, ambient_logits)
    # Minimize the Gaussian quadratic over the additive logit direction removed by softmax.
    projected_quadratic = unprojected_quadratic -
        additive_cross_term^2 / geometry.ones_precision_norm

    n_categories = length(theta)
    log_density = -0.5 * projected_quadratic -
        0.5 * geometry.logdet_alr_covariance -
        0.5 * (n_categories - 1) * log(2 * pi) -
        sum(log_theta)

    reconstructed = apply_smoothing_factors(
        ambient_logits,
        grid_sizes,
        geometry.mu_factor,
        geometry.alpha_factor,
    )
    residual = norm(reconstructed - centered_logits)
    return log_density, ambient_logits, residual, projected_quadratic
end

density_axis(log_density) = asinh(log_density / DENSITY_AXIS_SCALE)

function density_metrics(theta, geometries, grid_sizes)
    log_densities = Float64[]
    residuals = Float64[]
    ambient_norms = Float64[]
    projected_quadratics = Float64[]

    for geometry in geometries
        log_density, ambient_logits, residual, projected_quadratic =
            logistic_normal_log_density(theta, grid_sizes, geometry)
        push!(log_densities, log_density)
        push!(residuals, residual)
        push!(ambient_norms, norm(ambient_logits))
        push!(projected_quadratics, projected_quadratic)
    end

    return (
        unsmoothed_log_density=log_densities[1],
        log_densities=log_densities,
        residuals=residuals,
        ambient_norms=ambient_norms,
        projected_quadratics=projected_quadratics,
    )
end

function prior_density_plot(
    covariance_metrics,
    legacy_metrics;
    show_ylabel=true,
)
    density_plot = plot(
        SIGMA_VALUES,
        density_axis.(legacy_metrics.log_densities);
        xlabel="fixed kernel standard deviation, σ",
        xlims=extrema(SIGMA_VALUES),
        ylabel=show_ylabel ? "asinh(log density / 1000)" : "",
        label="Zero-padded convolution",
        color=:darkorange,
        linewidth=2.5,
        linestyle=:dash,
        tickfontsize=11,
        titlefontsize=17,
        guidefontsize=16,
        legendfontsize=12,
        legend=:topright,
        margin=8Plots.mm,
    )
    plot!(
        density_plot,
        SIGMA_VALUES,
        density_axis.(covariance_metrics.log_densities);
        label="Directional covariance",
        color=:steelblue,
        linewidth=3,
    )
    hline!(density_plot, [density_axis(covariance_metrics.unsmoothed_log_density)];
        color=:grey40, linestyle=:dot, linewidth=2, label="Logit-normal")
    return density_plot
end

function original_kernel_prior_density_plot(
    gaussian_heat,
    checkered_heat,
    shared_colorbar,
    gaussian_metrics,
    checkered_metrics,
)
    function density_panel(metrics; show_ylabel=true)
        panel = plot(
            SIGMA_VALUES,
            density_axis.(metrics.log_densities);
            xlabel="fixed kernel standard deviation, σ",
            xlims=extrema(SIGMA_VALUES),
            ylabel=show_ylabel ? "asinh(log density / 1000)" : "",
            label="Original five-point kernel",
            color=:darkorange,
            linewidth=3,
            marker=:circle,
            markersize=2.5,
            markerstrokewidth=0,
            title="",
            legend=:bottomright,
            tickfontsize=11,
            titlefontsize=15,
            guidefontsize=15,
            legendfontsize=12,
            margin=8Plots.mm,
        )
        hline!(
            panel,
            [density_axis(metrics.unsmoothed_log_density)];
            color=:grey35,
            linestyle=:dot,
            linewidth=2,
            label="Unsmoothed logit-normal",
        )
        return panel
    end

    gaussian_panel = density_panel(
        gaussian_metrics,
        show_ylabel=true,
    )
    checkered_panel = density_panel(
        checkered_metrics,
        show_ylabel=false,
    )
    return plot(
        gaussian_heat,
        checkered_heat,
        shared_colorbar,
        gaussian_panel,
        checkered_panel;
        layout=@layout([a b c{0.12w}; d e _]),
        size=(1500, 1300),
        dpi=300,
    )
end

function kernel_stability_plots(covariance_geometries, legacy_geometries)
    covariance_condition_numbers = getproperty.(
        covariance_geometries,
        :covariance_condition_number,
    )
    legacy_condition_numbers = getproperty.(
        legacy_geometries,
        :covariance_condition_number,
    )
    covariance_minimum_variances = getproperty.(
        covariance_geometries,
        :minimum_marginal_variance,
    )
    legacy_minimum_variances = getproperty.(
        legacy_geometries,
        :minimum_marginal_variance,
    )

    condition_plot = plot(
        SIGMA_VALUES,
        log10.(legacy_condition_numbers);
        xlabel="fixed kernel standard deviation, σ",
        ylabel="log₁₀ condition number",
        label="Zero-padded convolution",
        color=:darkorange,
        linestyle=:dash,
        linewidth=2.5,
        xlims=extrema(SIGMA_VALUES),
        legend=:topleft,
        tickfontsize=11,
        titlefontsize=17,
        guidefontsize=16,
        legendfontsize=12,
        margin=8Plots.mm,
    )
    plot!(
        condition_plot,
        SIGMA_VALUES,
        log10.(covariance_condition_numbers);
        label="Directional covariance",
        color=:steelblue,
        linewidth=3,
    )

    variance_plot = plot(
        SIGMA_VALUES,
        legacy_minimum_variances;
        xlabel="fixed kernel standard deviation, σ",
        ylabel="minimum marginal variance",
        label="Zero-padded convolution",
        color=:darkorange,
        linestyle=:dash,
        linewidth=2.5,
        xlims=extrema(SIGMA_VALUES),
        legend=:bottomleft,
        tickfontsize=11,
        titlefontsize=17,
        guidefontsize=16,
        legendfontsize=12,
        margin=8Plots.mm,
    )
    plot!(
        variance_plot,
        SIGMA_VALUES,
        covariance_minimum_variances;
        label="Directional covariance",
        color=:steelblue,
        linewidth=3,
    )
    return condition_plot, variance_plot
end

function covariance_matrix_plot(n_mu)
    covariance = CodonMolecularEvolution.gaussian_covariance_matrix(
        n_mu,
        COVARIANCE_REPORT_SIGMA^2;
        jitter=COVARIANCE_JITTER,
    )
    ticks = collect(1:n_mu)
    covariance_plot = heatmap(
        ticks,
        ticks,
        covariance;
        xlabel="μ-grid index, j",
        ylabel="μ-grid index, i",
        title="Cμ at fixed σ = $(COVARIANCE_REPORT_SIGMA)",
        color=:viridis,
        colorbar_title="correlation",
        clims=(0.0, 1.0),
        aspect_ratio=:equal,
        xlims=(0.5, n_mu + 0.5),
        ylims=(0.5, n_mu + 0.5),
        xticks=ticks,
        yticks=ticks,
        widen=false,
        grid=false,
        framestyle=:box,
        tickfontsize=12,
        titlefontsize=18,
        guidefontsize=16,
        right_margin=8Plots.mm,
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
        top_margin=6Plots.mm,
        size=(900, 780),
        dpi=300,
    )
    return covariance_plot, covariance
end

function main()
    mkpath(OUTDIR)

    mugrid, kappagrid, alphagrid = flavor_grids()
    grid_sizes = (length(mugrid), length(kappagrid), length(alphagrid), 2)
    n_categories = prod(grid_sizes)
    kappa_index = argmin(abs.(kappagrid .- 1.0))
    capped_index = 1

    gaussian = gaussian_surface(grid_sizes)
    checkered_gaussian = gaussian_surface(grid_sizes; checkered=true)
    gaussian_theta = probability_vector_from_surface(gaussian, grid_sizes)
    checkered_theta = probability_vector_from_surface(checkered_gaussian, grid_sizes)
    gaussian_slice = conditional_mu_alpha_slice(gaussian_theta, grid_sizes, kappa_index, capped_index)
    checkered_slice = conditional_mu_alpha_slice(checkered_theta, grid_sizes, kappa_index, capped_index)

    covariance_geometries = [
        smoothing_geometry(grid_sizes, sigma, covariance_smoothing_factor)
        for sigma in SIGMA_VALUES
    ]
    legacy_geometries = [
        smoothing_geometry(grid_sizes, sigma, legacy_smoothing_factor)
        for sigma in SIGMA_VALUES
    ]
    for (method_name, geometries) in (
        ("directional covariance", covariance_geometries),
        ("zero-padded convolution", legacy_geometries),
    )
        isapprox(geometries[1].logdet_alr_covariance, log(n_categories); atol=1e-10) ||
            error("The sigma-zero ALR determinant check failed for $method_name")
    end
    gaussian_covariance_metrics = density_metrics(
        gaussian_theta,
        covariance_geometries,
        grid_sizes,
    )
    gaussian_legacy_metrics = density_metrics(
        gaussian_theta,
        legacy_geometries,
        grid_sizes,
    )
    checkered_covariance_metrics = density_metrics(
        checkered_theta,
        covariance_geometries,
        grid_sizes,
    )
    checkered_legacy_metrics = density_metrics(
        checkered_theta,
        legacy_geometries,
        grid_sizes,
    )

    report_index = findfirst(isapprox(REPORT_SIGMA), SIGMA_VALUES)

    summary_path = joinpath(OUTDIR, "gaussian_vs_checkered_density_summary.txt")
    open(summary_path, "w") do io
        println(io, "Gaussian versus checkered Gaussian smoothFLAVOR prior comparison")
        println(io, "Reported smooth fixed kernel sigma: ", REPORT_SIGMA)
        println(io, "Sigma values: ", join(SIGMA_VALUES, ", "))
        println(io, "Methods: zero-padded convolution; directional covariance; logit-normal")
        println(io, "Gaussian covariance jitter: ", COVARIANCE_JITTER)
        println(io, "Full category count: ", n_categories)
        println(io, "Slice: capped=false, kappa index=", kappa_index,
            ", kappa=", kappagrid[kappa_index])
        println(io, "Gaussian grid standard deviation: ", GAUSSIAN_GRID_STDDEV)
        println(io, "Gaussian maximum edge/peak ratio: ", edge_to_peak_ratio(gaussian))
        println(io, "Checkered Gaussian maximum edge/peak ratio: ", edge_to_peak_ratio(checkered_gaussian))
        println(io, "Checkered low/high multiplier ratio: ", CHECKERBOARD_LOW_MULTIPLIER)

        for (label, covariance_metrics, legacy_metrics) in (
            ("Gaussian", gaussian_covariance_metrics, gaussian_legacy_metrics),
            ("Checkered Gaussian", checkered_covariance_metrics, checkered_legacy_metrics),
        )
            println(io)
            println(io, label)
            println(io, "  Unsmoothed logistic-normal log density: ",
                covariance_metrics.unsmoothed_log_density)
            for (method_name, metrics) in (
                ("Directional covariance", covariance_metrics),
                ("Zero-padded convolution", legacy_metrics),
            )
                println(io, "  ", method_name)
                println(io, "    Logistic-normal log density at sigma=",
                    REPORT_SIGMA, ": ", metrics.log_densities[report_index])
                println(io, "    Projected Gaussian quadratic at sigma=", REPORT_SIGMA,
                    ": ", metrics.projected_quadratics[report_index])
                println(io, "    Maximum inverse-logit residual norm: ", maximum(metrics.residuals))
                println(io, "    Inverse ambient L2 norm at sigma=", REPORT_SIGMA,
                    ": ", metrics.ambient_norms[report_index])
            end
        end

        println(io)
        println(io, "Per-axis stability over sigma grid")
        println(io, "  Zero-padded convolution maximum covariance condition number: ",
            maximum(getproperty.(legacy_geometries, :covariance_condition_number)))
        println(io, "  Directional covariance maximum covariance condition number: ",
            maximum(getproperty.(covariance_geometries, :covariance_condition_number)))
        println(io, "  Zero-padded convolution minimum marginal variance: ",
            minimum(getproperty.(legacy_geometries, :minimum_marginal_variance)))
        println(io, "  Directional covariance minimum marginal variance: ",
            minimum(getproperty.(covariance_geometries, :minimum_marginal_variance)))
    end

    common_clims = (0.0, max(maximum(gaussian_slice), maximum(checkered_slice)))
    alpha_ticks = (collect(eachindex(alphagrid)), string.(round.(alphagrid; sigdigits=3)))
    mu_ticks = (collect(eachindex(mugrid)), string.(round.(mugrid; sigdigits=3)))
    gaussian_heat = heatmap(
        collect(eachindex(alphagrid)),
        collect(eachindex(mugrid)),
        gaussian_slice;
        xlabel="α",
        ylabel="μ",
        title="Gaussian",
        color=:viridis,
        colorbar=false,
        clims=common_clims,
        aspect_ratio=:equal,
        xlims=(0.5, length(alphagrid) + 0.5),
        ylims=(0.5, length(mugrid) + 0.5),
        xticks=alpha_ticks,
        yticks=mu_ticks,
        widen=false,
        grid=false,
        framestyle=:box,
        xrotation=35,
        tickfontsize=9,
        titlefontsize=18,
        guidefontsize=17,
        margin=5Plots.mm,
    )
    checkered_heat = heatmap(
        collect(eachindex(alphagrid)),
        collect(eachindex(mugrid)),
        checkered_slice;
        xlabel="α",
        yaxis=false,
        title="Checkered Gaussian",
        color=:viridis,
        colorbar=false,
        clims=common_clims,
        aspect_ratio=:equal,
        xlims=(0.5, length(alphagrid) + 0.5),
        ylims=(0.5, length(mugrid) + 0.5),
        xticks=alpha_ticks,
        widen=false,
        grid=false,
        framestyle=:box,
        xrotation=35,
        tickfontsize=9,
        titlefontsize=18,
        guidefontsize=17,
        margin=5Plots.mm,
    )
    colorbar_values = collect(range(common_clims[1], common_clims[2]; length=256))
    colorbar_ticks = collect(range(common_clims[1], common_clims[2]; length=6))
    shared_colorbar = heatmap(
        [1.0],
        colorbar_values,
        reshape(colorbar_values, :, 1);
        color=:viridis,
        colorbar=false,
        clims=common_clims,
        xaxis=false,
        ymirror=true,
        yguide_position=:right,
        ylabel="P(μᵢ, αₖ | κ=1, uncapped)",
        yticks=(colorbar_ticks, string.(round.(colorbar_ticks; digits=2))),
        widen=false,
        grid=false,
        framestyle=:box,
        tickfontsize=10,
        guidefontsize=16,
        left_margin=3Plots.mm,
        right_margin=12Plots.mm,
        top_margin=3Plots.mm,
        bottom_margin=3Plots.mm,
    )

    gaussian_density_plot = prior_density_plot(
        gaussian_covariance_metrics,
        gaussian_legacy_metrics,
    )
    checkered_density_plot = prior_density_plot(
        checkered_covariance_metrics,
        checkered_legacy_metrics;
        show_ylabel=false,
    )
    condition_plot, variance_plot = kernel_stability_plots(
        covariance_geometries,
        legacy_geometries,
    )

    plot_object = plot(
        gaussian_heat,
        checkered_heat,
        shared_colorbar,
        gaussian_density_plot,
        checkered_density_plot,
        condition_plot,
        variance_plot;
        layout=@layout([a b c{0.12w}; d e _; f g _]),
        size=(1500, 1900),
        dpi=300,
    )

    comparison_pdf_path = joinpath(OUTDIR, "gaussian_vs_checkered_kernel_stability.pdf")
    comparison_png_path = joinpath(OUTDIR, "gaussian_vs_checkered_kernel_stability.png")
    comparison_svg_path = joinpath(OUTDIR, "gaussian_vs_checkered_kernel_stability.svg")
    savefig(plot_object, comparison_pdf_path)
    savefig(plot_object, comparison_png_path)
    savefig(plot_object, comparison_svg_path)

    original_kernel_plot = original_kernel_prior_density_plot(
        gaussian_heat,
        checkered_heat,
        shared_colorbar,
        gaussian_legacy_metrics,
        checkered_legacy_metrics,
    )
    original_kernel_pdf_path = joinpath(OUTDIR, "original_kernel_prior_density.pdf")
    original_kernel_png_path = joinpath(OUTDIR, "original_kernel_prior_density.png")
    original_kernel_svg_path = joinpath(OUTDIR, "original_kernel_prior_density.svg")
    savefig(original_kernel_plot, original_kernel_pdf_path)
    savefig(original_kernel_plot, original_kernel_png_path)
    savefig(original_kernel_plot, original_kernel_svg_path)

    covariance_plot, covariance = covariance_matrix_plot(grid_sizes[1])
    covariance_pdf_path = joinpath(OUTDIR, "c_mu_sigma_2.pdf")
    covariance_png_path = joinpath(OUTDIR, "c_mu_sigma_2.png")
    covariance_svg_path = joinpath(OUTDIR, "c_mu_sigma_2.svg")
    savefig(covariance_plot, covariance_pdf_path)
    savefig(covariance_plot, covariance_png_path)
    savefig(covariance_plot, covariance_svg_path)

    covariance_values_path = joinpath(OUTDIR, "c_mu_sigma_2.csv")
    open(covariance_values_path, "w") do io
        println(io, join(["i\\j"; string.(axes(covariance, 2))], ','))
        for i in axes(covariance, 1)
            println(io, join([string(i); string.(covariance[i, :])], ','))
        end
    end

    println("Wrote:")
    println(comparison_pdf_path)
    println(comparison_png_path)
    println(comparison_svg_path)
    println(original_kernel_pdf_path)
    println(original_kernel_png_path)
    println(original_kernel_svg_path)
    println(covariance_pdf_path)
    println(covariance_png_path)
    println(covariance_svg_path)
    println(covariance_values_path)
    println(summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
