using MolecularEvolution, FASTX, CodonMolecularEvolution
using Test

@testset "CodonMolecularEvolution.jl" begin
    include("meme_test.jl")
    include("parameter_study_test.jl")

    @testset "difFUBAR" begin
        include("difFUBAR_test.jl")
    end
end
