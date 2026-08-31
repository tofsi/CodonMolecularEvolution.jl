using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

Pkg.activate(REPO_ROOT)
using CodonMolecularEvolution
using LinearAlgebra
using Plots
using Statistics

const OUTDIR = joinpath(@__DIR__, "..", "results", "smoothflavor_prior_gaussian_slice")
const REPORT_SIGMA = 1.0
const SIGMA_VALUES = collect(0.0:0.05:1.0)
const GAUSSIAN_GRID_STDDEV = 2.0
const CHECKERBOARD_LOW_MULTIPLIER = 0.05
const DENSITY_AXIS_SCALE = 1_000.0

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

function convolution_matrix(n, sigma)
    matrix = Matrix{Float64}(undef, n, n)
    basis = zeros(n)
    kernel = CodonMolecularEvolution.gaussian_kernel(5, sigma^2)
    for j in 1:n
        fill!(basis, 0.0)
        basis[j] = 1.0
        matrix[:, j] .= CodonMolecularEvolution.same_conv(basis, kernel)
    end
    return matrix
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

function smoothing_geometry(grid_sizes, sigma)
    k_mu = convolution_matrix(grid_sizes[1], sigma)
    k_alpha = convolution_matrix(grid_sizes[3], sigma)
    ambient_ones = invert_smoothing(ones(prod(grid_sizes)), grid_sizes, k_mu, k_alpha)
    ones_precision_norm = dot(ambient_ones, ambient_ones)

    logdet_mu, sign_mu = logabsdet(k_mu)
    logdet_alpha, sign_alpha = logabsdet(k_alpha)
    iszero(sign_mu) && error("Singular mu smoothing matrix at sigma=$sigma")
    iszero(sign_alpha) && error("Singular alpha smoothing matrix at sigma=$sigma")

    n_blocks = grid_sizes[2] * grid_sizes[4]
    logabsdet_smoothing = n_blocks * (
        grid_sizes[3] * logdet_mu + grid_sizes[1] * logdet_alpha
    )
    # For ALR contrast matrix B, det(B * Sigma * B') = det(Sigma) * (1' * inv(Sigma) * 1).
    logdet_alr_covariance = 2 * logabsdet_smoothing + log(ones_precision_norm)

    return (
        k_mu=k_mu,
        k_alpha=k_alpha,
        ambient_ones=ambient_ones,
        ones_precision_norm=ones_precision_norm,
        logdet_alr_covariance=logdet_alr_covariance,
    )
end

function logistic_normal_log_density(theta, grid_sizes, sigma, geometry)
    log_theta = log.(theta)
    centered_logits = log_theta .- mean(log_theta)
    ambient_logits = invert_smoothing(
        centered_logits,
        grid_sizes,
        geometry.k_mu,
        geometry.k_alpha,
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

    reconstructed = CodonMolecularEvolution.apply_smoothing(
        CodonMolecularEvolution.FLAVORReshapingScheme(grid_sizes),
        ambient_logits,
        [sigma];
        dims=(1, 3),
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

    for (sigma, geometry) in zip(SIGMA_VALUES, geometries)
        log_density, ambient_logits, residual, projected_quadratic =
            logistic_normal_log_density(theta, grid_sizes, sigma, geometry)
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

function prior_density_plot(metrics, color, marker; show_ylabel=true)
    density_plot = plot(
        SIGMA_VALUES,
        density_axis.(metrics.log_densities);
        xlabel="smoothing kernel standard deviation",
        xlims=extrema(SIGMA_VALUES),
        ylabel=show_ylabel ? "asinh(log density / 1000)" : "",
        label="smoothFLAVOR density",
        color=color,
        linewidth=3,
        marker=marker,
        markersize=3,
        tickfontsize=9,
        titlefontsize=14,
        guidefontsize=14,
        legendfontsize=8,
        legend=:topright,
        margin=8Plots.mm,
    )
    hline!(density_plot, [density_axis(metrics.unsmoothed_log_density)];
        color=color, linestyle=:dot, linewidth=2, label="unsmoothed logit-normal")
    return density_plot
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

    geometries = [smoothing_geometry(grid_sizes, sigma) for sigma in SIGMA_VALUES]
    isapprox(geometries[1].logdet_alr_covariance, log(n_categories); atol=1e-10) ||
        error("The sigma-zero ALR determinant check failed")
    gaussian_metrics = density_metrics(gaussian_theta, geometries, grid_sizes)
    checkered_metrics = density_metrics(checkered_theta, geometries, grid_sizes)

    report_index = findfirst(isapprox(REPORT_SIGMA), SIGMA_VALUES)

    summary_path = joinpath(OUTDIR, "gaussian_vs_checkered_density_summary.txt")
    open(summary_path, "w") do io
        println(io, "Gaussian versus checkered Gaussian smoothFLAVOR prior comparison")
        println(io, "Reported smooth fixed kernel sigma: ", REPORT_SIGMA)
        println(io, "Sigma values: ", join(SIGMA_VALUES, ", "))
        println(io, "Full category count: ", n_categories)
        println(io, "Slice: capped=false, kappa index=", kappa_index,
            ", kappa=", kappagrid[kappa_index])
        println(io, "Gaussian grid standard deviation: ", GAUSSIAN_GRID_STDDEV)
        println(io, "Gaussian maximum edge/peak ratio: ", edge_to_peak_ratio(gaussian))
        println(io, "Checkered Gaussian maximum edge/peak ratio: ", edge_to_peak_ratio(checkered_gaussian))
        println(io, "Checkered low/high multiplier ratio: ", CHECKERBOARD_LOW_MULTIPLIER)

        for (label, metrics) in (("Gaussian", gaussian_metrics), ("Checkered Gaussian", checkered_metrics))
            println(io)
            println(io, label)
            println(io, "  Unsmoothed logistic-normal log density: ",
                metrics.unsmoothed_log_density)
            println(io, "  smoothFLAVOR logistic-normal log density at sigma=",
                REPORT_SIGMA, ": ", metrics.log_densities[report_index])
            println(io, "  Projected Gaussian quadratic at sigma=", REPORT_SIGMA,
                ": ", metrics.projected_quadratics[report_index])
            println(io, "  Maximum smooth inverse-logit residual norm: ", maximum(metrics.residuals))
            println(io, "  Smooth inverse ambient L2 norm at sigma=", REPORT_SIGMA,
                ": ", metrics.ambient_norms[report_index])
        end
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
        title="Gaussian μ-α slice",
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
        tickfontsize=7,
        titlefontsize=14,
        guidefontsize=14,
        margin=5Plots.mm,
    )
    checkered_heat = heatmap(
        collect(eachindex(alphagrid)),
        collect(eachindex(mugrid)),
        checkered_slice;
        xlabel="α",
        yaxis=false,
        title="Checkered Gaussian μ-α slice",
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
        tickfontsize=7,
        titlefontsize=14,
        guidefontsize=14,
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
        ylabel="category probability",
        yticks=(colorbar_ticks, string.(round.(colorbar_ticks; digits=2))),
        widen=false,
        grid=false,
        framestyle=:box,
        tickfontsize=8,
        guidefontsize=13,
        left_margin=3Plots.mm,
        right_margin=12Plots.mm,
        top_margin=3Plots.mm,
        bottom_margin=3Plots.mm,
    )

    gaussian_density_plot = prior_density_plot(
        gaussian_metrics,
        :steelblue,
        :circle,
    )
    checkered_density_plot = prior_density_plot(
        checkered_metrics,
        :firebrick,
        :diamond;
        show_ylabel=false,
    )

    plot_object = plot(
        gaussian_heat,
        checkered_heat,
        shared_colorbar,
        gaussian_density_plot,
        checkered_density_plot;
        layout=@layout([a b d{0.12w}; c e _]),
        size=(1500, 1400),
        dpi=300,
    )

    svg_path = joinpath(OUTDIR, "gaussian_vs_checkered_smoothflavor_prior.svg")
    savefig(plot_object, svg_path)

    println("Wrote:")
    println(svg_path)
    println(summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
