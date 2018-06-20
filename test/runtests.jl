using TransformVariables
using TransformVariables: logistic, logit
using Compat.Test

@testset "logistic and logit" begin
    for _ in 1:10000
        x = randn(Float64) * 50
        bx = BigFloat(x)
        lbx = 1/(1+exp(-bx))
        @test logistic(x) ≈ lbx
        lx, ljx = logistic(LOGJAC, x)
        @test lx ≈ lbx
        ljbx = -(log(1+exp(-bx))+log(1+exp(bx)))
        @test ljx ≈ ljbx rtol = eps(Float64)
    end
    for _ in 1:10000
        y = rand(Float64)
        @test logistic(logit(y)) ≈ y
    end
end

@testset "to array scalar" begin
    dims = (3, 4, 5)
    t = to_𝕀
    ta = to_array(t, dims...)
    @test dimension(ta) == prod(dims)
    x = randn(dimension(ta))
    y = transform(ta, x)
    @test typeof(y) == Array{Float64, length(dims)}
    @test size(y) == dims
    for i in 1:length(x)
        @test transform(t, [x[i]]) == y[i]
    end
    @test inverse(ta, y) ≈ x
end

@testset "to tuple scalar" begin
    t1 = to_ℝ
    t2 = to_𝕀
    t3 = to_ℝ₊
    tt = to_tuple(t1, t2, t3)
    x = randn(dimension(tt))
    y = transform(tt, x)
    for (i, t) in enumerate((t1, t2, t3))
        @test y[i] == transform(t, x[i:i])
    end
    @test inverse(tt, y) ≈ x
end
