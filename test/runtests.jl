using TransformVariables
using TransformVariables:
    unit_triangular_dimension, logistic, logistic_logjac, logit

using Base: vect
using DocStringExtensions
using ForwardDiff: derivative, jacobian

# TODO remove/change when only depending on v0.7
using Compat: undef
using Compat.Test
using Compat.LinearAlgebra: diag, logabsdet, UpperTriangular
using Compat.Random

include("test_utilities.jl")

srand(1)

@testset "misc utilities" begin
    @test unit_triangular_dimension(1) == 0
    @test unit_triangular_dimension(2) == 1
    @test unit_triangular_dimension(5) == 10
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
end

@testset "to unit vector" begin
    for K in 1:10
        t = to_unitvec(K)
        @test dimension(t) == K - 1
        if K > 1
            test_transformation(t, y -> sum(abs2, y) ≈ 1, y -> y[1:(end-1)])
        end
    end
end

@testset "to correlation cholesky factor" begin
    for K in 1:10
        t = to_corr_cholesky(K)
        @test dimension(t) == (K - 1)*K/2
        println("Travis keepalive correlation cholesky K = $(K)")
        if K > 1
            test_transformation(t, is_valid_corr_cholesky, vec_above_diagonal)
        end
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
