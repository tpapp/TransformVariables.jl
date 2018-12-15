using DocStringExtensions, LinearAlgebra, LogDensityProblems, OffsetArrays, Parameters,
    Random, Test, TransformVariables
import Flux, ForwardDiff, ReverseDiff
using LogDensityProblems: Value, ValueGradient
using TransformVariables:
    AbstractTransform, ScalarTransform, VectorTransform, ArrayTransform,
    unit_triangular_dimension, logistic, logistic_logjac, logit

include("test_utilities.jl")

Random.seed!(1)

const CIENV = get(ENV, "TRAVIS", "") == "true"  || get(ENV, "CI", "") == "true"

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
        test_transformation(as(Real, -∞, a), y -> y < a)
        test_transformation(as(Real, a, ∞), y -> y > a)
        b = a + 0.5 + rand(Float64) + exp(randn() * 10)
        test_transformation(as(Real, a, b), y -> a < y < b)
    end
    test_transformation(as(Real, -∞, ∞), _ -> true)
end

@testset "scalar transformation corner cases" begin
    @test_throws ArgumentError as(Real, "a fish", 9)
    @test as(Real, 1, 4.0) == as(Real, 1.0, 4.0)
    @test_throws ArgumentError as(Real, 3.0, -4.0)
end

@testset "to unit vector" begin
    for K in 1:10
        t = UnitVector(K)
        @test dimension(t) == K - 1
        if K > 1
            test_transformation(t, y -> sum(abs2, y) ≈ 1,
                                vec_y = y -> y[1:(end-1)])
        end
    end
end

@testset "to correlation cholesky factor" begin
    for K in 1:8
        t = CorrCholeskyFactor(K)
        @test dimension(t) == (K - 1)*K/2
        CIENV && @info "testing correlation cholesky K = $(K)"
        if K > 1
            test_transformation(t, is_valid_corr_cholesky;
                                vec_y = vec_above_diagonal, N = 100)
        end
    end
end

@testset "to array scalar" begin
    dims = (3, 4, 5)
    t = as𝕀
    ta = as(Array, t, dims...)
    @test dimension(ta) == prod(dims)
    x = random_arg(ta)
    y = @inferred transform(ta, x)
    @test typeof(y) == Array{Float64, length(dims)}
    @test size(y) == dims
    @test inverse(ta, y) ≈ x
    ℓacc = 0.0
    for i in 1:length(x)
        yi, ℓi = @inferred transform_and_logjac(t, x[i])
        @test yi == y[i]
        ℓacc += ℓi
    end
    y2, ℓ2 = transform_and_logjac(ta, x)
    @test y == y2
    @test ℓ2 ≈ ℓacc
end

@testset "as array fallback" begin
    is_expected(t, dims) = t isa ArrayTransform && t.transformation == asℝ && t.dims == dims
    @test is_expected(as(Array, 2, 3), (2, 3))
    @test is_expected(as(Array, (2, 3)), (2, 3))
    @test is_expected(as(Matrix, 2, 3), (2, 3))
    @test is_expected(as(Matrix, (2, 3)), (2, 3))
    @test_throws ArgumentError as(Vector, 2, 3)
    @test_throws ArgumentError as(Vector, (2, 3))
    @test is_expected(as(Vector, 2), (2, ))
end

@testset "as array w/ Identity" begin
    d = (2, 3, 4)
    t = as(Array, d)
    v = random_arg(t)
    y1 = transform(t, v)
    y2, lj = transform_and_logjac(t, v)
    @test y1 == y2 == reshape(v, d)
    @test lj == 0
end

@testset "to tuple" begin
    t1 = asℝ
    t2 = as𝕀
    t3 = CorrCholeskyFactor(7)
    tt = as((t1, t2, t3))
    @test dimension(tt) == dimension(t1) + dimension(t2) + dimension(t3)
    x = random_arg(tt)
    y = @inferred transform(tt, x)
    @test inverse(tt, y) ≈ x
    TransformVariables.inverse_eltype(tt, y)
    index = 0
    ljacc = 0.0
    for (i, t) in enumerate((t1, t2, t3))
        d = dimension(t)
        xpart = t isa ScalarTransform ? x[index + 1] : x[index .+ (1:d)]
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
    t1 = asℝ
    t2 = CorrCholeskyFactor(7)
    t3 = UnitVector(3)
    tn = as((a = t1, b = t2, c = t3))
    @test dimension(tn) == dimension(t1) + dimension(t2) + dimension(t3)
    x = randn(dimension(tn))
    y = @inferred transform(tn, x)
    @test y isa NamedTuple{(:a,:b,:c)}
    @test inverse(tn, y) ≈ x
    index = 0
    ljacc = 0.0
    for (i, t) in enumerate((t1, t2, t3))
        d = dimension(t)
        xpart = t isa ScalarTransform ? x[index + 1] : x[index .+ (1:d)]
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

@testset "empty Tuple and NamedTuple aggregators" begin
    zt = as(())
    znt = as(NamedTuple())
    @test dimension(zt) == dimension(znt) == 0
    @test transform(zt, Float64[]) == ()
    @test_skip inverse(zt, ()) == []
    @test transform_and_logjac(zt, Float64[]) == ((), 0.0)
    @test transform(znt, Float64[]) == NamedTuple()
    @test transform_and_logjac(znt, Float64[]) == (NamedTuple(), 0.0)
    @test_skip inverse(znt, ()) == []
end

@testset "transform logdensity" begin
    # the density is p(σ) = σ⁻³
    # let z = log(σ), so σ = exp(z)
    # the transformed density is q(z) = -3z + z = -2z
    f(σ) = -3*log(σ)
    q(z) = -2*z
    for _ in 1:1000
        z = randn()
        qz = transform_logdensity(asℝ₊, f, z)
        @test q(z) ≈ qz
    end
end

@testset "custom transformation: triangle below diagonal in [0,1]²" begin
    tfun(y) = y[1], y[1]*y[2]   # triangle below diagonal in unit square
    t = CustomTransform(as(Array, as𝕀, 2), tfun, collect;)
    test_transformation(t, ((y1, y2),) -> 0 ≤ y2 ≤ y1 ≤ 1;
                        vec_y = collect, test_inverse = false)

    # test inference w/ manually specified chunk
    function f(x)
        CustomTransform(as(Array, as𝕀, 2), tfun, collect;
                            chunk = ForwardDiff.Chunk{2}())
        transform_and_logjac(t, x)
    end
    y, lj = @inferred f(zeros(2))
    @test y == (0.5, 0.25)
    @test lj ≈ -3.465735902799726
end

@testset "custom transformation: covariance matrix" begin
    "Transform to a `n×n` covariance matrix."
    to_covariance(n) = CustomTransform(
        # pre-transform to standard deviations and correlation Cholesky factor
        as((as(Array, asℝ₊, n), CorrCholeskyFactor(n))),
        # use these to construct a covariance matrix
        (((σ, Ω),) -> (Ω*Diagonal(σ) |> x -> Symmetric(x'*x))),
        # flatten to elements above the diagonal
        A -> A[axes(A, 1) .≤ axes(A, 2)'])
    C5 = to_covariance(5)
    test_transformation(C5, A -> all(diag(cholesky(A).U) .> 0);
                        vec_y = C5.flatten, test_inverse = false)
end

@testset "offset arrays" begin
    t = as((λ = asℝ₊, a = CorrCholeskyFactor(4),
            θ = as((as(Array, as𝕀, 2, 3), UnitVector(4)))))
    x = random_arg(t)
    xo = OffsetVector(x, axes(x, 1) .- 7)
    @test transform(t, x) == transform(t, xo)
    @test transform_and_logjac(t, x) == transform_and_logjac(t, xo)
end

@testset "random value" begin
    t1 = asℝ
    t2 = CorrCholeskyFactor(7)
    t3 = UnitVector(3)
    tn = as((a = t1, b = t2, c = t3))
    y = random_value(tn)
    @test y isa NamedTuple{(:a, :b, :c), <: Tuple{Float64, AbstractMatrix, Vector}}
    @test size(y.b) == (7, 7)
    @test size(y.c) == (3, )
end

@testset "random arg" begin
    t = as(Array, 5)
    for _ in 1:1000
        @test sum(abs2, random_arg(t; cauchy = false, scale = 1.0)) ≤ 100
    end
end

@testset "AD tests" begin
    t = as((μ = asℝ, σ = asℝ₊, β = asℝ₋, α = as(Real, 0.0, 1.0),
            u = UnitVector(3), L = CorrCholeskyFactor(4)))
    function f(θ)
        @unpack μ, σ, β, α = θ
        -(abs2(μ) + abs2(σ) + abs2(β) + α)
    end
    P = TransformedLogDensity(t, f)
    x = zeros(dimension(t))
    v = logdensity(Value, P, x)

    # ForwardDiff
    P1 = ADgradient(:ForwardDiff, P)
    @test v == logdensity(Value, P1, x)
    g1 = @inferred logdensity(ValueGradient, P1, x)
    @test g1.value == v.value

    # Flux # NOTE @inferred removed as it currently fails, cf
    # https://github.com/FluxML/Flux.jl/issues/497
    P2 = ADgradient(:Flux, P)
    g2 = logdensity(ValueGradient, P2, x) #
    @test g2.value == v.value
    @test g2.gradient ≈ g1.gradient

    # ReverseDiff
    P3 = ADgradient(:ReverseDiff, P)
    g3 = @inferred logdensity(ValueGradient, P3, x)
    @test g3.value == v.value
    @test g3.gradient ≈ g1.gradient
end

@testset "inverse_and_logjac" begin
    # WIP, test separately until integrated
    for _ in 1:100
        x = randn()
        a = randn()
        t = as(Real, a, a + abs(randn()) + 0.1)
        y, lj = transform_and_logjac(t, x)
        x2, lj2 = TransformVariables.inverse_and_logjac(t, y)
        @test x2 ≈ x
        @test lj2 ≈ -lj
    end
end
