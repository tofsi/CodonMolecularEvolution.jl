include(joinpath(@__DIR__, "..", "parameter_study", "src", "SmoothFlavorStudy.jl"))
using .SmoothFlavorStudy
using CSV
using DataFrames
import CodonMolecularEvolution

@testset "parameter-study selection and aggregation" begin
    manifest = DataFrame(
        simulation_id=["sim_$(sim)_replicate_$(rep)" for sim in 0:4 for rep in 1:2],
        sim=[sim for sim in 0:4 for _ in 1:2],
        replicate=[rep for _ in 0:4 for rep in 1:2],
    )

    selected = select_random_omnibus_multi_simulations(
        manifest;
        count=3,
        replicate_ids=[1],
        rng_seed=1234,
    )
    repeated = select_random_omnibus_multi_simulations(
        manifest;
        count=3,
        replicate_ids=[1],
        rng_seed=1234,
    )

    @test selected.sim == repeated.sim
    @test selected.selection_order == 1:3
    @test all(selected.replicate .== 1)
    @test length(unique(selected.sim)) == 3
    @test all(selected.selection_seed .== 1234)
    @test all(selected.selection_method .== "uniform_random_without_replacement")
    @test_throws ArgumentError select_random_omnibus_multi_simulations(
        manifest;
        count=6,
        replicate_ids=[1],
        rng_seed=1234,
    )

    mktempdir() do outdir
        for simulation_id in ("sim_a", "sim_b")
            mkpath(joinpath(outdir, simulation_id))
        end

        complete = DataFrame(
            method=["original_BAME", "MEME", "smoothFLAVOR_BAME"],
            kernel_stddev=[NaN, NaN, 0.25],
        )
        partial = DataFrame(method=["original_BAME"], kernel_stddev=[NaN])
        CSV.write(joinpath(outdir, "sim_a", "method_sweep_summary.csv"), complete)
        CSV.write(joinpath(outdir, "sim_b", "method_sweep_summary.csv"), partial)

        selection_manifest = DataFrame(simulation_id=["sim_a", "sim_b"])
        completed = SmoothFlavorStudy._completed_simulation_ids(
            selection_manifest,
            outdir,
            [0.25],
        )
        aggregate = SmoothFlavorStudy._aggregate_summaries(
            outdir;
            simulation_ids=completed,
        )

        @test completed == ["sim_a"]
        @test unique(aggregate.simulation_id) == ["sim_a"]
    end

    mktempdir() do outdir
        summary_path = joinpath(outdir, "method_sweep_summary.csv")
        old_row = (
            method="original_BAME",
            kernel_stddev=NaN,
            elapsed_seconds=1.0,
        )
        diagnostic_row = (
            method="smoothFLAVOR_BAME",
            kernel_stddev=0.25,
            elapsed_seconds=2.0,
            mcmc_n_chains=8,
        )

        CSV.write(summary_path, DataFrame([old_row]))
        summary_rows = SmoothFlavorStudy._load_summary_rows(summary_path)
        push!(summary_rows, diagnostic_row)
        summary = SmoothFlavorStudy._write_summary(summary_path, summary_rows)

        @test nrow(summary) == 2
        @test :mcmc_n_chains in propertynames(summary)
        @test ismissing(summary.mcmc_n_chains[1])
        @test summary.mcmc_n_chains[2] == 8
    end
end

@testset "smoothFLAVOR convergence artifacts" begin
    ambient_samples = [
        [
            [
                sin(0.17 * iteration + chain),
                cos(0.11 * iteration + 0.3 * chain),
                0.05 * iteration + 0.2 * chain,
            ]
            for iteration in 1:40
        ]
        for chain in 1:3
    ]
    parameter_names = ["kernel_1", "suppression_1", "ambient_weight_1"]
    sampler_stats = [
        [
            (
                n_steps=2^(iteration % 5 + 1) - 1,
                is_accept=true,
                acceptance_rate=0.75 + 0.04 * sin(iteration + chain),
                log_density=-100.0 + 0.2 * sin(0.3 * iteration + chain),
                hamiltonian_energy=110.0 + 0.5 * cos(0.2 * iteration + chain),
                hamiltonian_energy_error=0.02 * sin(iteration),
                max_hamiltonian_energy_error=0.03 * abs(sin(iteration)),
                tree_depth=iteration % 17 == 0 ? 10 : iteration % 5 + 1,
                numerical_error=chain == 1 && iteration == 2,
                step_size=0.05 + 0.001 * chain,
                nom_step_size=0.05 + 0.001 * chain,
                is_adapt=iteration <= 10,
            )
            for iteration in 1:40
        ]
        for chain in 1:3
    ]

    mktempdir() do outdir
        prefix = joinpath(outdir, "kernel_stddev_1")
        artifacts = CodonMolecularEvolution.save_smoothFLAVOR_chain_artifacts!(
            prefix,
            ambient_samples,
            parameter_names;
            burnin=15,
            n_adapts=10,
            save_chain_samples=true,
        )

        @test size(artifacts.diagnostics, 1) == 3
        @test all(artifacts.diagnostics.n_chains .== 3)
        @test all(artifacts.diagnostics.retained_per_chain .== 25)
        @test isfile(prefix * "_chain_diagnostics.csv")
        @test isfile(prefix * "_chain_diagnostics_summary.csv")
        @test isfile(prefix * "_chain_samples.jld2")

        saved = CodonMolecularEvolution.JLD2.load(prefix * "_chain_samples.jld2")
        @test size(saved["samples"]) == (40, 3, 3)
        @test saved["parameter_names"] == parameter_names
        @test saved["burnin"] == 15
        @test saved["n_adapts"] == 10

        sampler_artifacts = CodonMolecularEvolution.save_NUTS_sampler_artifacts!(
            prefix,
            sampler_stats;
            burnin=15,
            n_adapts=10,
            max_tree_depth=10,
        )
        @test size(sampler_artifacts.trace, 1) == 120
        @test size(sampler_artifacts.diagnostics, 1) == 3
        @test only(sampler_artifacts.diagnostics_summary.total_adaptation_numerical_errors) == 1
        @test only(sampler_artifacts.diagnostics_summary.total_retained_max_tree_depth_hits) == 6
        @test isfinite(only(sampler_artifacts.diagnostics_summary.log_density_rhat))
        @test isfile(prefix * "_sampler_trace.csv")
        @test isfile(prefix * "_sampler_diagnostics.csv")
        @test isfile(prefix * "_sampler_diagnostics_summary.csv")
    end

    @test_throws ArgumentError CodonMolecularEvolution.save_smoothFLAVOR_chain_artifacts!(
        tempname(),
        ambient_samples,
        parameter_names;
        burnin=10,
        n_adapts=11,
    )
    @test_throws ArgumentError CodonMolecularEvolution.smoothFLAVOR_BAME(
        nothing,
        tempname();
        iters=20,
        burnin=8,
        n_adapts=9,
    )
end
