#####
##### experimental tests, need to be completely rewritten
#####
using TransformVariables
using TransformVariables.Experimental, StaticArrays

@testset "API hook" begin
    t = as((a = as(SVector{3}), b = asℝ))
    @test dimension(t) == 4
    @test transform(t, range(1.0, 4.0; length = 4)) == (a = SVector(1.0, 2.0, 3.0), b = 4.0)
end

@testset "as static array" begin
    S = Tuple{2,3,4}
    t = as(SArray{S})
    x = 1:dimension(t)
    y = @inferred transform(t, x)
    @test y isa SArray{S}
end
