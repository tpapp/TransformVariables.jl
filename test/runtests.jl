using TransformVariables
using TransformVariables: logistic, logistic_logjac, logit
using Compat.Test
using ForwardDiff: derivative

srand(1)

@testset "logistic and logit" begin
    for _ in 1:10000
        x = randn(Float64) * 50
        bx = BigFloat(x)
        lbx = 1/(1+exp(-bx))
        @test logistic(x) ≈ lbx
        ljx = logistic_logjac(x)
        ljbx = -(log(1+exp(-bx))+log(1+exp(bx)))
        @test ljx ≈ ljbx rtol = eps(Float64)
    end
    for _ in 1:10000
        y = rand(Float64)
        @test logistic(logit(y)) ≈ y
    end
end

function test_scalar_transformation(t, is_valid_y; N = 10000)
    for _ in 1:N
        x = randn(Float64)
        y = transform(t, [x])
        @test y isa Float64
        @test is_valid_y(y)
        y2, lj = transform(t, LOGJAC, [x])
        @test y == y2
        @test log(abs(derivative(x -> transform(t, [x]), x))) ≈ lj atol = √eps()
        x2 = inverse(t, y)[1]
        @test x2 ≈ x atol = √eps()
    end
end

@testset "scalar transformations consistency" begin
    for _ in 1:100
        a = randn() * 100
        test_scalar_transformation(to_interval(-∞, a), y -> y < a)
        test_scalar_transformation(to_interval(a, ∞), y -> y > a)
        b = a + 0.5 + rand(Float64) + exp(randn() * 10)
        test_scalar_transformation(to_interval(a, b), y -> a < y < b)
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
