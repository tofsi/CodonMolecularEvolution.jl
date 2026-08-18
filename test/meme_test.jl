using DataFrames

@testset "MEME statistics" begin
    lrts, p_values = CodonMolecularEvolution.MEME_test(
        [10.0, 10.0004, 12.0],
        [10.1, 10.0, 10.0];
        verbosity=0,
    )

    @test lrts[1] == 0.0
    @test lrts[2] == 0.0
    @test lrts[3] == 4.0
    @test p_values[1] == CodonMolecularEvolution.p_value(0.0)
    @test p_values[3] < p_values[1]

    alternative_params = [(
        alpha=1.234567,
        omegas=[0.5, 2.345678],
        qminus=0.765432,
    )]
    unrounded = CodonMolecularEvolution.MEME_tabulate(
        [4.123456],
        [0.0123456],
        alternative_params,
        [-12.34567],
        "unused",
        false;
        round_digits=nothing,
    )
    rounded = CodonMolecularEvolution.MEME_tabulate(
        [4.123456],
        [0.0123456],
        alternative_params,
        [-12.34567],
        "unused",
        false,
    )

    @test unrounded[1, "p-value"] == 0.0123456
    @test unrounded[1, "LRT"] == 4.123456
    @test rounded[1, "p-value"] == 0.01
    @test rounded[1, "LRT"] == 4.12
end
