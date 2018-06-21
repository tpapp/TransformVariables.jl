using TransformVariables
using TransformVariables: unit_triangular_dimension, logistic, logistic_logjac, logit
using Compat: undef
using Compat.Test
using Compat.LinearAlgebra: diag, logdet, UpperTriangular

using ForwardDiff: derivative, jacobian

srand(1)

@testset "misc utilities" begin
    @test unit_triangular_dimension(1) == 0
    @test unit_triangular_dimension(2) == 1
    @test unit_triangular_dimension(5) == 10
end

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

function AD_logjac(t::TransformReals, x, fvec)
    J = jacobian(x -> fvec(transform(t, x)), x)
    logdet(J)
end

function test_vector_transformation(t::TransformReals, isvalid, fvec; N = 10000)
    for _ in 1:N
        x = randn(dimension(t))
        y = transform(t, x)
        @test isvalid(y)
        x2 = inverse(t, y)
        @test x ≈ x2
        y2, lj = transform(t, LOGJAC, x)
        @test y2 == y
        @test lj ≈ AD_logjac(t, x, fvec)
    end
end

@testset "to unit vector" begin
    for K in 1:10
        t = to_unitvec(K)
        @test dimension(t) == K - 1
        if K > 1
            test_vector_transformation(t, y -> sum(abs2, y) ≈ 1, y -> y[1:(end-1)])
        end
    end
end

function vec_unit_triangular(U::UpperTriangular{T}) where T
    n = size(U, 1)
    index = 1
    x = Vector{T}(undef, unit_triangular_dimension(n))
    for col in 1:n
        for row in 1:(col-1)
            x[index] = U[row, col]
            index += 1
        end
    end
    x
end

function is_valid_corr_cholesky(U::UpperTriangular)
    Ω = U'*U
    all(isapprox.(diag(Ω), 1)) && all(@. abs(Ω) ≤ (1+√eps()))
end

@testset "to correlation cholesky factor" begin
    for K in 1:10
        t = to_corr_cholesky(K)
        @test dimension(t) == (K - 1)*K/2
        if K > 1
            test_vector_transformation(t, is_valid_corr_cholesky,
                                       vec_unit_triangular)
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
