using TransformVariables
using TransformVariables:
    TransformReals, unit_triangular_length, logistic, logistic_logjac, logit

using Base: vect
using DocStringExtensions
using ForwardDiff: derivative, jacobian

# TODO remove/change when only depending on v0.7
using Test, Random
using LinearAlgebra: diag, logabsdet, UpperTriangular

include("test_utilities.jl")

srand(1)

const CIENV = get(ENV, "TRAVIS", "") == "true"  || get(ENV, "CI", "") == "true"

@testset "misc utilities" begin
    @test unit_triangular_length(1) == 0
    @test unit_triangular_length(2) == 1
    @test unit_triangular_length(5) == 10
end

@testset "logistic and logit" begin
    for _ in 1:1000
        x = randn(Float64) * 50
        bx = BigFloat(x)
        lbx = 1/(1+exp(-bx))
        @test logistic(x) ≈ lbx
        ljx = logistic_logjac(x)
        ljbx = -(log(1+exp(-bx))+log(1+exp(bx)))
        @test ljx ≈ ljbx rtol = eps(Float64)
    end
    for _ in 1:1000
        y = rand(Float64)
        @test logistic(logit(y)) ≈ y
    end
end

@testset "scalar transformations consistency" begin
    for _ in 1:100
        a = randn() * 100
        test_transformation(to_interval(-∞, a), y -> y < a, vect)
        test_transformation(to_interval(a, ∞), y -> y > a, vect)
        b = a + 0.5 + rand(Float64) + exp(randn() * 10)
        test_transformation(to_interval(a, b), y -> a < y < b, vect)
    end
    test_transformation(to_interval(-∞, ∞), _ -> true, vect)
end

@testset "scalar transformation corner cases" begin
    @test_throws ArgumentError to_interval("a fish", 9)
    @test to_interval(1, 4.0) == to_interval(1.0, 4.0)
    @test_throws ArgumentError to_interval(3.0, -4.0)
end

@testset "to unit vector" begin
    for K in 1:10
        t = to_unitvec(K)
        @test length(t) == K - 1
        if K > 1
            test_transformation(t, y -> sum(abs2, y) ≈ 1, y -> y[1:(end-1)])
        end
    end
end

@testset "to correlation cholesky factor" begin
    for K in 1:10
        t = to_corr_cholesky(K)
        @test length(t) == (K - 1)*K/2
        CIENV && println("correlation cholesky K = $(K)")
        if K > 1
            test_transformation(t, is_valid_corr_cholesky, vec_above_diagonal)
        end
    end
end

@testset "to array scalar" begin
    dims = (3, 4, 5)
    t = to_𝕀
    ta = to_array(t, dims...)
    @test length(ta) == prod(dims)
    x = randn(length(ta))
    y = transform(ta, x)
    @test typeof(y) == Array{Float64, length(dims)}
    @test size(y) == dims
    @test inverse(ta, y) ≈ x
    ℓacc = 0.0
    for i in 1:length(x)
        yi, ℓi = transform_and_logjac(t, [x[i]])
        @test yi == y[i]
        ℓacc += ℓi
    end
    y2, ℓ2 = transform_and_logjac(ta, x)
    @test y == y2
    @test ℓ2 ≈ ℓacc
end

@testset "to tuple" begin
    t1 = to_ℝ
    t2 = to_𝕀
    t3 = to_corr_cholesky(7)
    tt = to_tuple(t1, t2, t3)
    @test length(tt) == length(t1) + length(t2) + length(t3)
    x = randn(length(tt))
    y = transform(tt, x)
    @test inverse(tt, y) ≈ x
    index = 0
    ljacc = 0.0
    for (i, t) in enumerate((t1, t2, t3))
        d = length(t)
        xpart = x[index .+ (1:d)]
        @test y[i] == transform(t, xpart)
        ypart, ljpart = transform_and_logjac(t, xpart)
        @test ypart == y[i]
        ljacc += ljpart
        index += d
    end
    y2, lj2 = transform_and_logjac(tt, x)
    @test y == y2
    @test lj2 ≈ ljacc
end

@testset "to named tuple" begin
    t1 = to_ℝ
    t2 = to_𝕀
    t3 = to_corr_cholesky(7)
    tn = to_tuple((a = t1, b = t2, c = t3))
    @test length(tn) == length(t1) + length(t2) + length(t3)
    x = randn(length(tn))
    y = transform(tn, x)
    @test y isa NamedTuple{(:a,:b,:c)}
    @test inverse(tn, y) ≈ x
    index = 0
    ljacc = 0.0
    for (i, t) in enumerate((t1, t2, t3))
        d = length(t)
        xpart = x[index .+ (1:d)]
        @test y[i] == transform(t, xpart)
        ypart, ljpart = transform_and_logjac(t, xpart)
        @test ypart == y[i]
        ljacc += ljpart
        index += d
    end
    y2, lj2 = transform_and_logjac(tn, x)
    @test y == y2
    @test lj2 ≈ ljacc
end
