@testset "Gaussian covariance smoothing" begin
    jitter = 1e-6
    variance = 4.0
    covariance = CodonMolecularEvolution.gaussian_covariance_matrix(
        6,
        variance;
        jitter=jitter,
    )
    factor = CodonMolecularEvolution.gaussian_covariance_factor(
        6,
        variance;
        jitter=jitter,
    )

    @test covariance ≈ transpose(covariance)
    @test CodonMolecularEvolution.diag(covariance) ≈ ones(6)
    @test CodonMolecularEvolution.eigmin(
        CodonMolecularEvolution.Symmetric(covariance),
    ) > 0.0
    @test factor * transpose(factor) ≈ covariance
    @test_throws ArgumentError CodonMolecularEvolution.GaussianCovarianceSmoothing(0.0)

    grid_sizes = (3, 2, 4, 2)
    scheme = CodonMolecularEvolution.FLAVORReshapingScheme(grid_sizes)
    method = CodonMolecularEvolution.GaussianCovarianceSmoothing(jitter)
    ambient = collect(range(-1.0, 1.0; length=prod(grid_sizes)))
    bandwidth = 1.3
    transformed = CodonMolecularEvolution.apply_smoothing(
        scheme,
        ambient,
        [bandwidth];
        dims=(1, 3),
        smoothing_method=method,
    )
    @test CodonMolecularEvolution.apply_smoothing(
        scheme,
        ambient,
        [bandwidth];
        dims=[1, 3],
        smoothing_method=method,
    ) ≈ transformed

    ambient_array = CodonMolecularEvolution.reshape_probability_vector(scheme, ambient)
    expected_array = similar(ambient_array)
    mu_factor = CodonMolecularEvolution.gaussian_covariance_factor(
        grid_sizes[1],
        bandwidth^2;
        jitter=jitter,
    )
    alpha_factor = CodonMolecularEvolution.gaussian_covariance_factor(
        grid_sizes[3],
        bandwidth^2;
        jitter=jitter,
    )
    for shape_index in axes(ambient_array, 2), capped_index in axes(ambient_array, 4)
        block = ambient_array[:, shape_index, :, capped_index]
        expected_array[:, shape_index, :, capped_index] .=
            mu_factor * block * transpose(alpha_factor)
    end
    expected = CodonMolecularEvolution.unreshape_probability_vector(scheme, expected_array)
    @test transformed ≈ expected

    transform = CodonMolecularEvolution.AmbientToParameterTransform(
        scheme,
        1,
        0,
        4.0,
        0.0,
        (1, 3);
        smoothing_method=method,
    )
    @test transform.smoothing_method === method

    ambient_sample = vcat(bandwidth / transform.kernel_stddev, ambient)
    kernel_parameters, suppression_parameters, smoothed_parameters =
        CodonMolecularEvolution.transform_ambient_components(transform, ambient_sample)
    @test isempty(suppression_parameters)
    @test only(kernel_parameters) ≈ bandwidth
    @test smoothed_parameters ≈ transformed
    @test CodonMolecularEvolution.transform_ambient_sample(transform, ambient_sample) ≈
          vcat(kernel_parameters, suppression_parameters, smoothed_parameters)

    con_lik = reshape(
        collect(range(0.5, 1.5; length=prod(grid_sizes) * 3)),
        prod(grid_sizes),
        3,
    )
    parameter_grids = [collect(Float64, 1:n) for n in grid_sizes]
    codon_param_index_vec = vec([collect(Tuple(index)) for index in CartesianIndices(grid_sizes)])
    codon_param_vec = [Float64.(index) for index in codon_param_index_vec]
    model = CodonMolecularEvolution.SKBDIModel(
        parameter_grids,
        ["parameter_$i" for i in eachindex(grid_sizes)],
        nothing,
        identity,
        log.(con_lik),
        con_lik,
        codon_param_vec,
        codon_param_index_vec,
        transform,
        1,
        grid_sizes,
    )
    probability_vector = CodonMolecularEvolution.to_probability_vector(model, ambient_sample)
    shifted = smoothed_parameters .- maximum(smoothed_parameters)
    expected_probability_vector = exp.(shifted)
    expected_probability_vector ./= sum(expected_probability_vector)
    @test probability_vector ≈ expected_probability_vector
    @test CodonMolecularEvolution.log_likelihood(model, ambient_sample) ≈
          sum(log.(transpose(con_lik) * expected_probability_vector))

    objective = function (parameters)
        smoothed = CodonMolecularEvolution.apply_smoothing(
            scheme,
            parameters[2:end],
            parameters[1:1];
            dims=(1, 3),
            smoothing_method=method,
        )
        return sum(sin.(smoothed))
    end
    parameters = vcat(bandwidth, ambient)
    value, gradient = CodonMolecularEvolution.value_and_gradient(
        objective,
        CodonMolecularEvolution.AutoMooncake(; config=nothing),
        parameters,
    )
    step = 1e-5
    upper = copy(parameters)
    lower = copy(parameters)
    upper[1] += step
    lower[1] -= step
    finite_difference = (objective(upper) - objective(lower)) / (2step)

    @test isfinite(value)
    @test all(isfinite, gradient)
    @test gradient[1] ≈ finite_difference rtol = 1e-6 atol = 1e-7

    likelihood_value, likelihood_gradient = CodonMolecularEvolution.value_and_gradient(
        sample -> CodonMolecularEvolution.log_likelihood(model, sample),
        CodonMolecularEvolution.AutoMooncake(; config=nothing),
        ambient_sample,
    )
    @test isfinite(likelihood_value)
    @test all(isfinite, likelihood_gradient)
end

@testset "smoothFLAVOR category reshaping" begin
    mugrid = [0.5, 2.0]
    shapegrid = [0.5, 1.5]
    alphagrid = [0.1, 1.0, 10.0]
    gridpoints = [
        (mu, shape, alpha)
        for mu in mugrid
        for shape in shapegrid
        for alpha in alphagrid
    ]
    grid_sizes = (length(mugrid), length(shapegrid), length(alphagrid), 2)
    n_categories = prod(grid_sizes)
    n_sites = 3
    flavorgrid = CodonMolecularEvolution.FLAVORgrid(
        identity,
        identity,
        mugrid,
        shapegrid,
        alphagrid,
        gridpoints,
        collect(grid_sizes[1:3]),
        fill(inv(n_categories), n_categories, n_sites),
        zeros(n_sites),
    )

    metadata = CodonMolecularEvolution.flavor_parameter_metadata(flavorgrid)
    fast_scheme = CodonMolecularEvolution.FLAVORReshapingScheme(grid_sizes)
    general_scheme = CodonMolecularEvolution.GeneralCategoricalReshapingScheme(
        grid_sizes,
        metadata.codon_param_index_vec,
    )
    category_values = collect(1.0:n_categories)
    fast_array = CodonMolecularEvolution.reshape_probability_vector(
        fast_scheme,
        category_values,
    )
    general_array = CodonMolecularEvolution.reshape_probability_vector(
        general_scheme,
        category_values,
    )

    @test fast_array == general_array
    @test CodonMolecularEvolution.unreshape_probability_vector(
        fast_scheme,
        fast_array,
    ) == category_values
    @test CodonMolecularEvolution.flavor_con_lik_matrix(
        flavorgrid;
        normalized=false,
    ) == flavorgrid.prob_matrix

    @test_throws ArgumentError CodonMolecularEvolution.SKBDIModel_from_FLAVOR(
        flavorgrid;
        kernel_dim=2,
    )
    @test_throws ArgumentError CodonMolecularEvolution.SKBDIModel_from_FLAVOR(
        flavorgrid;
        kernel_stddev=-1.0,
    )

    custom_gridpoints = copy(gridpoints)
    custom_gridpoints[1], custom_gridpoints[2] = custom_gridpoints[2], custom_gridpoints[1]
    custom_flavorgrid = CodonMolecularEvolution.FLAVORgrid(
        identity,
        identity,
        mugrid,
        shapegrid,
        alphagrid,
        custom_gridpoints,
        collect(grid_sizes[1:3]),
        flavorgrid.prob_matrix,
        flavorgrid.site_scalers,
    )
    @test_throws ArgumentError CodonMolecularEvolution.SKBDIModel_from_FLAVOR(
        custom_flavorgrid;
        fast_reshaping=true,
    )
    custom_model = CodonMolecularEvolution.SKBDIModel_from_FLAVOR(
        custom_flavorgrid;
        fast_reshaping=false,
    )
    @test custom_model.ambient_to_parameter_transform.reshaping_scheme isa
          CodonMolecularEvolution.GeneralCategoricalReshapingScheme
end
