#####
##### experimental tests, need to be completely rewritten
#####
using TransformVariables
using TransformVariables.Experimental, StaticArrays

@testset "API hook" begin
    t = as((a = asℝ₋, b = as(SMatrix{2,3}), c = asℝ₊))
    @test dimension(t) == 8
    @test transform(t, 1:8) == (a = -exp(1), b = SMatrix{2,3}(2:7), c = exp(8))
end

@testset "as static array" begin
    S = Tuple{2,3,4}
    t = as(SArray{S})
    x = 1:dimension(t)
    y = @inferred transform(t, x)
    @test y isa SArray{S}
end
