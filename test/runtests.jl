using DocStringExtensions, LinearAlgebra, LogDensityProblems, OffsetArrays, Random, Test,
    TransformVariables, StaticArrays, TransformedLogDensities, LogDensityProblemsAD,
    Accessors
import ForwardDiff
using LogDensityProblems: logdensity, logdensity_and_gradient
using LogDensityProblemsAD
using TransformVariables:
    AbstractTransform, ScalarTransform, VectorTransform, ArrayTransformation,
    unit_triangular_dimension, logistic, logistic_logjac, logit, inverse_and_logjac,
    NOLOGJAC, transform_with
import ChangesOfVariables, InverseFunctions
using Enzyme: autodiff, ReverseWithPrimal, Active, Const
using Unitful: @u_str, ustrip, uconvert

"are we in a CI environment (fewer iterations)"
const CIENV = get(ENV, "CI", "") == "true"
"are we testing for allocations (coverage does not interfere)"
const ALLOCS = get(ENV, "BUILD_IS_PRODUCTION_BUILD", "false") == "true"

@info "test environment" CIENV ALLOCS

include("utilities.jl")

Random.seed!(1)

####
#### utilities
####

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

####
#### scalar transformations correctness checks
####

@testset "scalar show" begin
    @test string(asℝ) == "asℝ"
    @test string(asℝ₊) == "asℝ₊"
    @test string(asℝ₋) == "asℝ₋"
    @test string(as𝕀) == "as𝕀"
    @test string(TVNeg() ∘ TVExp()) == "asℝ₋"
    @test string(as(Real, 0.0, 2.0)) == "as(Real, 0.0, 2.0)"
    @test string(as(Real, 1.0, ∞)) == "as(Real, 1.0, ∞)"
    @test string(as(Real, -∞, 1.0)) == "as(Real, -∞, 1.0)"

    @test string(TVExp()) == "asℝ₊"
    @test string(TVLogistic()) == "as𝕀"
    @test string(TVShift(4.0)) == "TVShift(4.0)"
    @test string(TVScale(4.0)) == "TVScale(4.0)"
    @test string(TVNeg()) == "TVNeg()"

    @test string(TVScale(2.0) ∘ TVNeg() ∘ TVExp()) == "TVScale(2.0) ∘ TVNeg() ∘ asℝ₊"
    @test string(TVScale(5.0u"m") ∘ TVLogistic()) == "TVScale(5.0 m) ∘ as𝕀"
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

@testset "scalar non-Real (Unitful) consistency" begin
    for _ in 1:10
        a = randn() * 100
        b = a + 0.5 + rand(Float64) + exp(randn() * 10)
        t1 = TVScale(2u"m") ∘ TVShift(a) ∘ TVExp()
        test_transformation(t1, y -> y > a*2u"m", jac=false)
        t2 = TVScale(1u"s") ∘ TVShift(a) ∘ TVScale(b-a) ∘ TVLogistic()
        test_transformation(t2, y -> (a*u"s" < y < b*u"s"), jac=false)

        @test_throws MethodError transform_and_logjac(t1, 0.5)
        @test_throws MethodError transform_and_logjac(t2, 0.5)
        y1 = transform(t1, 0.5)
        y2 = transform(t1, 0.5)
        @test_throws MethodError inverse_and_logjac(t1, y1)
        @test_throws MethodError inverse_and_logjac(t2, y2)
    end
end

@testset "vector with TVScale unitful" begin
    t = as(Vector, TVScale(2u"m"), 4)
    test_transformation(t, y -> y isa Vector && eltype(y) ≡ typeof(2.0u"m"); jac = false)
end

@testset "composite scalar transformations" begin
    all_transforms = [TVShift(3.0), TVScale(2.0), TVExp(), TVLogistic(), TVNeg()]
    for t1 in all_transforms, t2 in all_transforms, t3 in all_transforms
        t = t1 ∘ t2 ∘ t3
        # Basic functionality
        @test t isa TransformVariables.CompositeScalarTransform{Tuple{typeof(t1), typeof(t2), typeof(t3)}}
        # Other constructions
        @test t ∘ t isa TransformVariables.CompositeScalarTransform{Tuple{typeof(t1), typeof(t2), typeof(t3), typeof(t1), typeof(t2), typeof(t3)}}
        @test t == t1 ∘ (t2 ∘ t3)
        @test t == ∘(t1, t2, t3)
        @test t == TransformVariables.compose( ∘(t1), ∘(t2), ∘(t3))
    end
end

@testset "semiarbitrary compositions" begin
    same_domain_transforms = [TVShift(3.0), TVScale(2.0), TVNeg()]
    new_domain_transforms = [TVExp(), TVLogistic()]
    for s1 in same_domain_transforms, s2 in same_domain_transforms, n in new_domain_transforms
        for s3 in same_domain_transforms, s4 in same_domain_transforms
            # don't worry about valid output here, let inverse check that
            test_transformation(s1 ∘ s2 ∘ n ∘ s3 ∘ s4, _ -> true; N=100)
            @test TransformVariables.compose(s1, ∘(s2), n, ∘(s3), s4) == s1 ∘ s2 ∘ n ∘ s3 ∘ s4
        end
    end
end

@testset "scalar transformation corner cases" begin
    @test_throws ArgumentError as(Real, "a fish", 9)
    @test_throws ArgumentError as(Real, 3.0, -4.0)

    t = as(Real, 1.0, ∞)
    @test_throws DomainError inverse(t, 0.5)

    t = as(Real, -∞, 10.0)
    @test_throws DomainError inverse(t, 11.0)

    t = as(Real, 1.0, 10.0)
    @test_throws DomainError inverse(t, 0.5)
    @test_throws DomainError inverse(t, 11.0)
    @test_throws DomainError inverse_and_logjac(t, 0.5)
    @test_throws DomainError inverse_and_logjac(t, 11.0)

    t = TVExp ∘ 5
    @test_throws MethodError transform(t, 0.5)
end

@testset "scalar alternatives" begin
    @test as_real ≡ asℝ
    @test as_positive_real ≡ asℝ₊
    @test as_negative_real ≡ asℝ₋
    @test as_unit_interval ≡ as𝕀
end

@testset "scalar promotion" begin
    a = 0f0
    @test transform(asℝ, a) isa Float32
    @test transform(asℝ₊, a) isa Float32
    @test transform(asℝ₋, a) isa Float32
    @test transform(as𝕀, a) isa Float32
end

@testset "inverse_and_logjac" begin
    for _ in 1:100
        x = randn()
        a = randn()
        t = as(Real, a, a + abs(randn()) + 0.1)
        y, lj = transform_and_logjac(t, x)
        x2, lj2 = TransformVariables.inverse_and_logjac(t, y)
        @test x2 ≈ x
        @test lj2 ≈ -lj

        t = as(Real, a, ∞)
        y, lj = transform_and_logjac(t, x)
        x2, lj2 = TransformVariables.inverse_and_logjac(t, y)
        @test x2 ≈ x
        @test lj2 ≈ -lj

        t = as(Real, -∞, a)
        y, lj = transform_and_logjac(t, x)
        x2, lj2 = TransformVariables.inverse_and_logjac(t, y)
        @test x2 ≈ x
        @test lj2 ≈ -lj

        # For completeness sake,
        t = as(Real, -∞, ∞)
        y, lj = transform_and_logjac(t, x)
        x2, lj2 = TransformVariables.inverse_and_logjac(t, y)
        @test x2 ≈ x
        @test lj2 ≈ -lj
    end
end

####
#### special array transformation correctness checks
####

# NOTE deprecated, remove in 0.9
@testset "to unit vector" begin
    @testset "dimension checks" begin
        U = UnitVector(3)
        x = zeros(3)               # incorrect
        @test_throws ArgumentError transform(U, x)
        @test_throws ArgumentError transform_and_logjac(U, x)
    end

    @testset "consistency checks" begin
        for K in 1:10
            t = UnitVector(K)
            @test dimension(t) == K - 1
            if K > 1
                test_transformation(t, y -> sum(abs2, y) ≈ 1,
                                    vec_y = y -> y[1:(end-1)])
            end
        end
    end
end

@testset "to unit vector & norm" begin
    @testset "dimension checks" begin
        U = unit_vector_norm(3)
        x = zeros(4)               # incorrect
        @test_throws ArgumentError transform(U, x)
        @test_throws ArgumentError transform_and_logjac(U, x)
    end

    @testset "consistency checks" begin
        _is_valid_yr((y, r)) = sum(abs2, y) ≈ 1 && r ≥ 0
        _vec_yr((y, r)) = vcat(y, r)

        for K in 2:10
            # default
            t = unit_vector_norm(K)
            @test dimension(t) == K
            @test TransformVariables.nonzero_logprior(t)
            test_transformation(t, _is_valid_yr; vec_y = _vec_yr)

            @test all(isnan, transform(t, zeros(dimension(t)))[1][1:(end-1)])
            @test inverse(t, (zeros(K), 0.0)) == zeros(K)

            # no chi prior
            t = unit_vector_norm(K; chi_prior = false)
            @test dimension(t) == K
            @test !TransformVariables.nonzero_logprior(t)
            test_transformation(t, _is_valid_yr; vec_y = _vec_yr)
        end
    end
end

@testset "to unit simplex" begin
    @testset "dimension checks" begin
        S = UnitSimplex(3)
        x = zeros(3)               # incorrect
        @test_throws ArgumentError transform(S, x)
        @test_throws ArgumentError transform_and_logjac(S, x)
    end

    @testset "consistency checks" begin
        for K in 1:10
            t = UnitSimplex(K)
            @test dimension(t) == K - 1
            if K > 1
                test_transformation(t, y -> (sum(y) ≈ 1) & (all(y.>=0)),
                                    vec_y = y -> y[1:(end-1)])
            end
            x = zeros(dimension(t))
            @test transform(t, x) ≈ 1 ./ fill(K, K)
        end
    end
end

@testset "tanh helpers" begin
    for _ in 1:10000
        x = (rand() - 0.5) * 100
        (; log_l2_rem, logjac) =  TransformVariables.tanh_helpers(x)
        @test Float64(AD_logjac(tanh, BigFloat(x))) ≈ logjac atol = 1e-4
        @test Float64(log(sech(BigFloat(x))^2)) ≈ log_l2_rem atol = 1e-4
    end
end

@testset "to correlation cholesky factor" begin
    @testset "dimension checks" begin
        C = corr_cholesky_factor(3)
        wrong_x = zeros(dimension(C) + 1)

        @test_throws ArgumentError transform(C, wrong_x)
        @test_throws ArgumentError transform_and_logjac(C, wrong_x)
    end

    @testset "consistency checks" begin
        for K in 1:8
            t = corr_cholesky_factor(K)
            @test dimension(t) == (K - 1)*K/2
            CIENV && @info "testing correlation cholesky K = $(K)"
            if K > 1
                test_transformation(t, is_valid_corr_cholesky;
                                    vec_y = vec_above_diagonal, N = 100)
            end
        end
    end
end

@testset "logprior fallbacks" begin
    struct DummyTransformation <: AbstractTransform end
    @test !TransformVariables.nonzero_logprior(DummyTransformation())
    @test TransformVariables.logprior(DummyTransformation(), 2.0) == 0.0
end

####
#### aggregation
####

###
### array correctness checks
###

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
    is_expected(t, dims) = t isa ArrayTransformation && t.inner_transformation == asℝ && t.dims == dims
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

@testset "empty array transformation" begin
    t = as(Vector, 0)
    x = Float64[]
    y = transform(t, x)
    @test x == inverse(t, y)
    y, lj = transform_and_logjac(t, x)
    @test lj == 0.0
end

###
### tuple correctness checks
###

@testset "to tuple" begin
    t1 = asℝ
    t2 = as𝕀
    t3 = CorrCholeskyFactor(7)
    inner = (t1, t2, t3)
    tt = as(inner)
    @test dimension(tt) == mapreduce(dimension, +, inner)

    @testset "transformation correctness check" begin
        x = random_arg(tt)
        y = @inferred transform(tt, x)
        @test inverse(tt, y) ≈ x
        @test @inferred(TransformVariables.inverse_eltype(tt, y)) === Float64
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

    @testset "tuple elements access with indexing and Accessors.jl" begin
        @test length(tt) == 3
        @test tt[1] == inner[1]
        @test tt[2] == inner[2]
        @test tt[3] == inner[3]
        @test_throws BoundsError tt[4]
        @test propertynames(tt) == propertynames(inner)
        @test (@set tt[3] = t2) == as((t1, t2, t2))
        @test (@delete tt[3]) == as((t1, t2))
        @test (@insert tt[2] = t3) == as((t1, t3, t2, t3))
    end
end

###
### named tuple correctness checks
###

@testset "to named tuple" begin
    t1 = asℝ
    t2 = CorrCholeskyFactor(7)
    t3 = unit_vector_norm(3)
    inner = (a = t1, b = t2, c = t3)
    tn = as(inner)
    @test dimension(tn) == dimension(t1) + dimension(t2) + dimension(t3)

    @testset "transformation correctness check" begin
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

    @testset "tuple elements access with indexing and Accessors.jl" begin
        @test length(tn) == 3
        @test tn[1] == tn.a == inner[1]
        @test tn[2] == tn.b == inner[2]
        @test tn[3] == tn.c == inner[3]
        @test_throws ErrorException tn.d
        @test_throws BoundsError tn[4]
        @test propertynames(tn) == propertynames(inner)
        @test (@set tn[3] = t2) == (@set tn.c = t2) == as((a = t1, b = t2, c = t2))
        @test (@delete tn[3]) == (@delete tn.c) == as((a = t1, b = t2))
        @test (@insert tn.d = t3) == as((a = t1, b = t2, c = t3, d = t3))
    end
end

@testset "empty Tuple, NamedTuple aggregators" begin
    zt = as(())
    znt = as(NamedTuple())
    za = as(Array, asℝ₊, 0)
    @test dimension(zt) == dimension(znt) == 0
    @test @inferred(transform(zt, Float64[])) == ()
    @test inverse(zt, ()) == []
    @test @inferred(transform_and_logjac(zt, Float64[])) == ((), 0.0)
    @test @inferred(transform(znt, Float64[])) == NamedTuple()
    @test @inferred(transform_and_logjac(znt, Float64[])) == (NamedTuple(), 0.0)
    @test inverse(znt, (;)) == []
    @test @inferred(transform(za, Float64[])) == Float64[]
    @test @inferred(transform_and_logjac(za, Float64[])) == (Float64[], 0.0)
    @test inverse(za, []) == []
end

@testset "nested combinations" begin
    # for https://github.com/tpapp/TransformVariables.jl/issues/57
    for _ in 1:10
        N = rand(3:7)
        tt = as((a = as(Tuple(as(Vector, asℝ₊, 2) for _ in 1:N)),
                 b = as(Tuple(unit_vector_norm(n) for n in 2:N))))
        x = randn(dimension(tt))
        y = transform(tt, x)
        x′ = inverse(tt, y)
        @test x ≈ x′
    end
end

@testset "nested NamedTuple" begin
    # test for https://github.com/tpapp/TransformVariables.jl/issues/93
    t1 = as((a = as(Real, 0, 0.1), b = as((b1 = as(Real, 1, 5), b2 = as(Real, 10, 50)))))
    x = randn(dimension(t1))
    y = transform(t1, x)
    x′ = inverse(t1, y)
    @test x ≈ x′
end

@testset "different order and superset of NamedTuple" begin
    # test for #100
    t2 = corr_cholesky_factor(SMatrix{2,2})
    b = transform(t2, [0.0])
    t = as((a = asℝ, b = t2))
    @test @inferred(inverse(t, (a = 1.0, b))) == [1.0, 0.0]
    @test @inferred(inverse(t, (b, a = 1.0))) == [1.0, 0.0]
    @test_throws ArgumentError("Property :b not in (:a,).") inverse(t, (; a = 1.0))
    @test_throws ArgumentError("Names (:a, :b, :c) has extras compared to (:a, :b).") inverse(t, (a = 1.0, b = 2.0, c = 3.0))
    @test_throws ArgumentError("Property :b not in (:a, :c).") inverse(t, (a = 1.0, c = 2.0))
end

@testset "merging NamedTuple" begin
    t1 = as((a = asℝ, b = as𝕀))
    t2 = as((c = CorrCholeskyFactor(3), d = unit_vector_norm(4)))
    t3 = as((e = asℝ₊, f = as𝕀))
    tm = @inferred(merge(t1))
    @test tm == t1
    tm = @inferred(merge(t1, t2))
    @test tm == as((a = asℝ, b = as𝕀, c = CorrCholeskyFactor(3), d = unit_vector_norm(4)))
    tm = @inferred(merge(t1, t2, t3))
    @test tm == as((a = asℝ, b = as𝕀, c = CorrCholeskyFactor(3), d = unit_vector_norm(4),
                    e = asℝ₊, f = as𝕀))
    x = randn(dimension(tm))
    y = transform(tm, x)
    x′ = inverse(tm, y)
    @test x ≈ x′
    # Check merge collision behavior: rightmost gets kept
    t4 = as((b = asℝ₋, c = TVScale(2.0)))
    tm = @inferred(merge(t1, t4))
    @test tm == as((a = asℝ, b = asℝ₋, c = TVScale(2.0)))
    @test tm != as((a = asℝ, b = as𝕀, c = TVScale(2.0)))
    tm = @inferred(merge(t1, t4, t2))
    @test tm == as((a = asℝ, b = asℝ₋, c = CorrCholeskyFactor(3), d = unit_vector_norm(4)))

end

@testset "transform to custom type" begin
    
    struct CustomType{A, B}
        a::A
        b::B
    end
    @kwdef struct KwCustomType{A, B}
        a::A
        b::B
    end
    struct MyType{C}
        c::C
    end
    struct YourType
        d::Float64
    end

    # From named tuple to type
    t1 = as((a = asℝ, b = asℝ))
    t2 = as(CustomType, t1)
    @test_throws MethodError transform(t2, [1.0, 2.0])

    t1 = as((a = asℝ, b = asℝ))
    t2 = as(KwCustomType, t1)
    x = [1.0, 2.0]
    test_transformation(t2, y -> y isa KwCustomType; N=1, jac=false)
    @test_throws ArgumentError inverse(t2, [1.0, 2.0])
    @test_throws ArgumentError inverse(t2, MyType(3.0))

    # Named tuple with different ordering
    t1 = as((b = asℝ, a = asℝ))    
    t2 = as(KwCustomType, t1)
    y = @inferred transform(t2, [1.0, 2.0])
    @test y == KwCustomType(a = 2.0, b = 1.0) 
    test_transformation(t2, y -> y isa KwCustomType; N=1, jac=false)

    # Named tuple with wrong number or names of fields
    t1 = as((;b = asℝ))    
    t2 = as(KwCustomType, t1)
    @test_throws UndefKeywordError transform(t2, [1.0])
    @test inverse(t2, KwCustomType(1.0, 3.0)) == [3.0]
    t1 = as((a = asℝ, c = asℝ))    
    t2 = as(KwCustomType, t1)
    @test_throws UndefKeywordError transform(t2, [1.0, 3.0])
    @test_throws ArgumentError inverse(t2, KwCustomType(1.0, 3.0))
    t1 = as((b = asℝ, a = asℝ, c = asℝ))    
    t2 = as(KwCustomType, t1)
    @test_throws MethodError transform(t2, [1.0, 2.0, 3.0])
    @test_throws ArgumentError inverse(t2, KwCustomType(1.0, 3.0))

    # Type with shortened constructor
    struct MaskedType{A, B}
        a::A
        b::B
    end
    MaskedType(x) = MaskedType(x, nothing)
    MaskedType(;b=0.0) = MaskedType(nothing, b)
    t = as(MaskedType, as((;b=asℝ)))
    test_transformation(t, y -> y isa MaskedType; N=1, jac=false)

    # No kwarg constructor accepts `a` arg, so errors
    t = as(MaskedType, as((;a=asℝ)))
    @test_throws MethodError transform(t, [1.0])

    # When constructor accepts less args than struct has fields, 
    # inverse takes only the first fields of the struct and warns user
    t = as(MaskedType, asℝ)
    test_transformation(t, y -> y isa MaskedType; N=1, jac=false)
    @test_logs (:warn,r"more fields") inverse(t, MaskedType(1.0, nothing))
    @test inverse(t, MaskedType(1.0, 2.0)) == [1.0]
    # test_transformation(t, y -> y isa MaskedType; N=1, jac=false)

    # Insufficient arguments in provided tuple for constructor
    # Not specially caught, but good to check
    t1 = as(CustomType, (asℝ₊))
    @test_throws MethodError transform(t1, [1.0])

    # From tuple to type
    t1 = as(ntuple(i->asℝ₊, Val(2)))
    t2 = as(CustomType, t1)
    test_transformation(t2, y -> y isa CustomType; N=1, jac=false)
    # Trying to invert from another type should error
    @test_throws ArgumentError inverse(t2, [1.0, 2.0])
    @test_throws ArgumentError inverse(t2, MyType(3.0))

    # Nested custom types
    t1 = as(MyType, (asℝ₋,))
    t2 = as(YourType, asℝ₋)
    t3 = as(KwCustomType, as((a = t1, b = t2)))
    x = [0.0, -1]
    y = transform(t3, x)
    test_transformation(t3, y -> y isa KwCustomType; N=1, jac=false)
    # Switched order should still work
    @test y == KwCustomType(;b = YourType(-exp(-1)), a = MyType(-1.0))
    # Inverting with wrong type
    @test_throws ArgumentError inverse(t3, KwCustomType(-1.0, YourType(-exp(-1))))
    @test_throws ArgumentError inverse(t3, CustomType(MyType(-1.0), YourType(-exp(-1))))

    # Type with scalar transform argument wraps it in a tuple
    t1 = as(MyType, asℝ₋)
    x = [0.0] # Needs to be vector, since full scalar interface not implemented
    y = transform(t1, x)
    test_transformation(t1, y -> y isa MyType; N=1, jac=false)
    @test_throws ArgumentError inverse(t1, -exp(0.0))

end

####
#### log density correctness checks
####

@testset "transform logdensity: correctness" begin
    # the density is p(σ) = σ⁻³
    # let z = log(σ), so σ = exp(z)
    # the transformed density is q(z) = -3z + z = -2z
    f(σ) = -3*log(σ)
    q(z) = -2*z
    for _ in 1:1000
        z = randn()
        qz = @inferred transform_logdensity(asℝ₊, f, z)
        @test q(z) ≈ qz
    end
end

@testset "transform logdensity: type inference" begin
    t = as((a = asℝ₋, b = as𝕀, c = as((d = unit_vector_norm(7), e = CorrCholeskyFactor(3))),
            f = as(Array, 9)))
    z = zeros(dimension(t))
    f(θ) = θ.a + θ.b + sum(abs2, θ.c.d[1]) + sum(abs2, θ.c.e)
    @test (@inferred f(transform(t, z))) isa Float64
    @test (@inferred transform_logdensity(t, f, z)) isa Float64
end

####
#### custom transformations
####

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
            θ = as((as(Array, as𝕀, 2, 3), unit_vector_norm(4)))))
    x = random_arg(t)
    xo = OffsetVector(x, axes(x, 1) .- 7)
    @test transform(t, x) == transform(t, xo)
    @test transform_and_logjac(t, x) == transform_and_logjac(t, xo)
end

####
#### AD compatibility tests
####

@testset "AD tests" begin
    t = as((μ = asℝ, σ = asℝ₊, β = asℝ₋, α = as(Real, 0.0, 1.0),
            u = unit_vector_norm(3), L = CorrCholeskyFactor(4),
            δ = as((asℝ₋, as𝕀))))
    function f(θ)
        (; μ, σ, β, α, δ, u) = θ
        -(abs2(μ) + abs2(σ) + abs2(β) + α + δ[1] + δ[2] + u[2])
    end
    P = TransformedLogDensities.TransformedLogDensity(t, f)
    x = randn(dimension(t))
    v = logdensity(P, x)
    g = ForwardDiff.gradient(x -> logdensity(P, x), x)

    @testset "ForwardDiff" begin
        P1 = ADgradient(:ForwardDiff, P)
        @test v == logdensity(P1, x)
        v1, g1 = @inferred logdensity_and_gradient(P1, x)
        @test v1 == v
        @test g1 ≈ g

        xd = ForwardDiff.Dual(-800.0, 1.0)
        @test first(ForwardDiff.partials(transform(as(Real, 0.0, 1.0), xd))) == 0.0
    end

    # Tests https://github.com/tpapp/TransformVariables.jl/pull/102
    @testset "Enzyme ScaledShifted" begin
        ss = as(Real, 0.0, 3.0)
        function enzyme(ss, x)
            y, lj = transform_and_logjac(ss, x)
            return -abs2(y) + lj
        end
        g, _ = autodiff(ReverseWithPrimal, enzyme, Const(ss), Active(0.5))
        g2 = ForwardDiff.derivative(x -> enzyme(ss, x), 0.5)
        @test g[2] ≈ g2
    end
end

# if VERSION ≥ v"1.1"
#     if CIENV
#         @info "installing Zygote"
#         import Pkg
#         Pkg.API.add(Pkg.PackageSpec(; name = "Zygote"))
#     end

#     import Zygote

#     @testset "Zygote AD" begin
#         # Zygote
#         # NOTE @inferred removed as it currently fails
#         # NOTE tests simplified disabled as they currently fail
#         t = as((μ = asℝ, ))
#         function f(θ)
#             (; μ) = θ
#             -(abs2(μ))
#         end
#         P = TransformedLogDensity(t, f)
#         x = zeros(dimension(t))
#         PF = ADgradient(:ForwardDiff, P)
#         PZ = ADgradient(:Zygote, P)
#         @test @inferred(logdensity(PZ, x)) == logdensity(P, x)
#         vZ, gZ = logdensity_and_gradient(PZ, x)
#         @test vZ == logdensity(P, x)
#         @test gZ ≈ last(logdensity_and_gradient(PF, x))
#     end
# end

@testset "inference of nested tuples" begin
    # An MWE adapted from a real-life problem
    ABOVE1 = as(Real, 1, ∞)   # transformation for μ ≥ 1

    trans_β̃s = as((asℝ, asℝ))     # a tuple of 2 elements, otherwise identity

    PARAMS_TRANSFORMATIONS =
        (EE = as((β̃s = trans_β̃s, μs = as((as𝕀, as𝕀)))),
         EN = as((w̃₂ = asℝ, β̃s = trans_β̃s, μs = as((as𝕀, ABOVE1)))),
         NE = as((w̃₁ = asℝ, β̃s = trans_β̃s, μs = as((ABOVE1, as𝕀)))),
         NN = as((w̃s = as((asℝ, asℝ)), β̃s = trans_β̃s, μs = as((ABOVE1, ABOVE1)))))

    function make_transformation(ls)
        as((hyper_parameters = as((μ = as(Array, 6),
                                   σ = as(Array, asℝ₊, 6),
                                   LΩ = CorrCholeskyFactor(6))),
            couple_parameters = as(map((t, l) -> as(Array, t, l),
                                       PARAMS_TRANSFORMATIONS, ls))))
    end
    t = make_transformation((EE = 1, EN = 2 , NE = 3, NN = 4,))
    x = zeros(dimension(t))
    @test_nowarn @inferred transform(t, x)
    @test_nowarn @inferred transform_and_logjac(t, x)
end

@testset "support abstract array inverses in ArrayTransformation" begin
    t = as(Array, 2, 3)
    @test inverse(t, ones(SMatrix{2,3})) == ones(6)
end

@testset "constant transformations" begin
    c = rand(3, 3)
    ta = as(Vector, asℝ₊, 3)
    t = as((a = ta, b = Constant(c)))
    t0 = as((a = ta,))
    N = dimension(t)
    @test N == dimension(t0)
    x = rand(N)
    y = @inferred transform(t, x)
    @test y == merge(transform(t0, x), (b = c,))
    let (y2, l) = transform_and_logjac(t0, x)
        @test transform_and_logjac(t, x) == (merge(y2, (b = c,)), l)
    end
    @test inverse(t, y) ≈ x
end

####
#### broadcasting
####

@testset "broadcasting" begin
    @test transform.(as𝕀, [0, 0]) == [0.5, 0.5]

    t = unit_vector_norm(3)
    d = dimension(t)
    x = [randn(d), randn(d)]
    @test transform.(t, x) == map(x -> transform(t, x), x)
end

####
#### show
####

@testset "sum dimensions allocations" begin
    shifted = TVShift(0.0) ∘ TVExp()
    tr = (a = shifted, b = TransformVariables.Identity(), c = shifted, d = shifted, e = shifted, f = shifted)
    ALLOCS && @test iszero(@allocated TransformVariables._sum_dimensions(tr))
end

@testset "inverse_eltype inference" begin
    t0 = TVShift(0f0) ∘ TVExp()
    t = as((x0 = TVShift(0f0) ∘ TVExp(), x1 = TransformVariables.Identity(),
            x2 = UnitSimplex(7), x3 = TransformVariables.CorrCholeskyFactor(5),
            x4 = as(Real, -∞, 1), x5 = as(Array, 10, 2), x6 = as(Array, as𝕀, 10),
            x7 = as((a = asℝ₊, b = as𝕀)), x8 = unit_vector_norm(10),
            x9 = t0, x10 = t0, x11 = t0, x12 = t0, x13 = TransformVariables.Identity(),
            x14 = t0, x15 = t0, x16 = t0, x17  = t0))
    x = randn(@inferred(TransformVariables.dimension(t)))
    y = TransformVariables.transform(t, x)
    @test @inferred(TransformVariables.inverse_eltype(t, y)) === Float64
end

####
#### partial application
####

@testset "partial application" begin
    t = asℝ₊
    x = 0.7
    @test transform(t)(x) == transform(t, x)
    y = transform(t, x)
    @test inverse(t)(y) == inverse(t, y) == inverse(transform(t))(y) ≈ x
end

@testset "ChangesOfVariables" begin
    t = as(Real, 1.0, 3.0)
    f = transform(t)
    inv_f = inverse(t)
    ChangesOfVariables.test_with_logabsdet_jacobian(f, -4.2, ForwardDiff.derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(inv_f, 1.7, ForwardDiff.derivative)
end

@testset "InverseFunctions" begin
    t = as(Real, 1.0, 3.0)
    f = transform(t)
    inv_f = inverse(t)
    InverseFunctions.test_inverse(f, -4.2)
    InverseFunctions.test_inverse(inv_f, 1.7)
end

@testset "as static array" begin
    S = Tuple{2,3,4}
    t = as(SArray{S})
    x = 1:dimension(t)
    y = @inferred transform(t, x)
    @test y isa SArray{S}
end

@testset "as static array with inner transformation" begin
    S = Tuple{2,3}
    i = corr_cholesky_factor(SMatrix{2,2})
    t = as(SArray{S}, i)
    @test dimension(t) == dimension(i) * 6
    x = rand(dimension(t))
    y = @inferred transform(t, x)
    @test y isa SArray{S}
    @test y == transform(as(Array, i, 2, 3), x)
    @test inverse(t, y) ≈ x
end

@testset "static corr cholesky factor" begin
    for K in 1:5
        for _ in 1:10
            t = corr_cholesky_factor(SMatrix{K,K})
            t2 = corr_cholesky_factor(K)
            @test dimension(t) == dimension(t2)
            x = rand(dimension(t))
            y = @inferred transform(t, x)
            @test parent(y) isa SMatrix
            @test y == transform(t2, x)
            @test inverse(t, y) ≈ x
        end
    end

    # allocations
    t7 = corr_cholesky_factor(SMatrix{7,7})
    z7 = zeros(dimension(t7))
    VERSION ≥ v"1.12" && ALLOCS && @show @allocations(transform(t7, z7))
end

@testset "corr cholesky factor large inputs" begin
    t = corr_cholesky_factor(7)
    d = dimension(t)
    for _ in 1:100
        x = sign.(rand(d) .- 0.5) .* 100
        @test isfinite(logdet(transform(t, x)) )
    end
end

@testset "pretty printing" begin
    t = as((a = asℝ₊,
            b = as(Array, asℝ₋, 3, 3),
            c = corr_cholesky_factor(13),
            d = as((asℝ, corr_cholesky_factor(SMatrix{3,3}), UnitSimplex(3), unit_vector_norm(4)))))
    repr_t = """
[1:97] NamedTuple of transformations
  [1:1] :a → asℝ₊
  [2:10] :b → 3×3×asℝ₋
  [11:88] :c → 13×13 correlation cholesky factor
  [89:97] :d → Tuple of transformations
    [98:98] 1 → asℝ
    [108:110] 2 → SMatrix{3,3} correlation cholesky factor
    [120:121] 3 → 3 element unit simplex transformation
    [131:133] 4 → 4 element (unit vector, norm) transformation"""
    repr(MIME("text/plain"), t) == repr_t
end

@testset "print ∞" begin
    @test repr(MIME("text/plain"), ∞) == "∞"
    @test repr(MIME("text/plain"), -∞) == "-∞"
end

@testset "domain labels" begin
    t = as((a = asℝ₊,
            b = as(Array, asℝ₋, 1, 1),
            c = corr_cholesky_factor(2),
            d = as(SVector{2}, asℝ₊),
            v = as(view, 2)))
    @test [domain_label(t, i) for i in 1:dimension(t)] ==
        [".a", ".b[1,1]", ".c[1]", ".d[1]", ".d[2]", ".v[1]", ".v[2]"]
end

@testset "static arrays inference" begin
    @test @inferred transform_with(NOLOGJAC, as(SVector{3, Float64}), zeros(3), 1) == (SVector(0.0, 0.0, 0.0), NOLOGJAC, 4)
    @test @inferred transform_with(NOLOGJAC, as(SVector{1, Float64}), zeros(1), 1) == (SVector(0.0), NOLOGJAC, 2)
end

@testset "view transformations" begin
    x = randn(10)
    t = as((a = asℝ, b = as(view, 2, 4), c = asℝ))
    y, lj = transform_and_logjac(t, x)
    @test typeof(y.b) <: AbstractMatrix
    @test size(y.b) == (2, 4)
    # test inverse
    @test inverse(t, y) == x
    # test that it is a view
    z = y.b[3]
    y.b[3] += 1
    @test x[4] == z + 1
end

@testset "near-singular Cholesky factor" begin
    x = [8.348500225024523, -3.80486310849193, -15.115725300837742, 5.840812234057503,
         7.548980701857334, 0.6546495312434718, 2.863837638357627, 1.0081703617568052,
         -38.543769810398466, -14.252165683848483, -22.75952203884357, 1.9543987098768612,
         5.415229912144962, -1.4360948924991273, 4.957606068283541, -5.443369115798325,
         -2.536087079311158, -2.0710241403850635, -0.982209305513312, 6.821758239096414,
         5.925173901833287]
    t = corr_cholesky_factor(7)
    U = transform(t, x)
    @test isfinite(logabsdet(U)[1])
end

@testset "inverse_eltype of scalar transforms with parameters" begin
    # `Float64` parameters and `Float32` input
    for t in (as(Real, 0.5, ∞), as(Real, -∞, 2.1), as(Real, 0.5, 2.1))
        @test @inferred(inverse_eltype(t, 1.1f0)) === Float64
        @test @inferred(inverse(t, 1.1f0)) isa Float64
    end

    # Derivatives wrt parameters of the transforms
    d1 = ForwardDiff.derivative(5.3) do x
        return @inferred only(inverse(as(Vector, as(Real, x, ∞), 1), [10]))
    end
    d2 = ForwardDiff.derivative(5.3) do x
        return @inferred inverse(as(Real, x, ∞), 10)
    end
    @test d1 == d2
    d1 = ForwardDiff.derivative(-3) do x
        return @inferred only(inverse(as(Vector, as(Real, -∞, x), 1), [-6.1]))
    end
    d2 = ForwardDiff.derivative(-3) do x
        return @inferred inverse(as(Real, -∞, x), -6.1)
    end
    @test d1 == d2
    d1 = ForwardDiff.gradient([-0.3, 4.7]) do x
        return @inferred only(inverse(as(Vector, as(Real, x[1], x[2]), 1), [2.3]))
    end
    d2 = ForwardDiff.gradient([-0.3, 4.7]) do x
        return @inferred inverse(as(Real, x[1], x[2]), 2.3)
    end
    @test d1 == d2
end

@testset "inverse of VectorTransform" begin
    # Empty `inverse(::VectorTransform, _)`
    for a in (3, 4.7, [5], 3f0, 4.7f0, [5f0])
        x = @inferred(inverse(as((; a = Constant(a))), (; a)))
        @test x isa Vector{Bool}
        @test isempty(x)

        x = @inferred(inverse(as((Constant(a),)), (a,)))
        @test x isa Vector{Bool}
        @test isempty(x)

        x = @inferred(inverse(as(Vector, Constant(a), 1), [a]))
        @test x isa Vector{Bool}
        @test isempty(x)
    end

    # Element type of `inverse(::VectorTransform, _)`
    for t in (asℝ, asℝ₊)
        for a in (3, 3.0, 3f0)
            z = inverse(t, a)
            T = typeof(z)

            x = @inferred(inverse(as((; a = t)), (; a)))
            @test x isa Vector{T}
            @test x == [z]

            x = @inferred(inverse(as((t,)), (a,)))
            @test x isa Vector{T}
            @test x == [z]

            x = @inferred(inverse(as(Vector, t, 1), [a]))
            @test x isa Vector{T}
            @test x == [z]
        end
    end
end

@testset "nested transformations element type" begin
    t = as(Vector, as((a = asℝ,)), 4)
    x = zeros(dimension(t))
    y = transform(t, x)
    @test @inferred(inverse(t, y)) == x
    @test inverse_eltype(t, typeof(y)) ≡ eltype(x)

    t0 = as(Vector, as((a = asℝ,)), 0)
    y0 = @inferred(transform(t0, Float64[]))
    @test @inferred(transform(t0, Float64[])) == y0
    @test inverse_eltype(t, typeof(y0)) ≡ Float64
end

@testset "element type corner cases" begin
    t = as(Vector, asℝ₊, 3)
    x = @inferred inverse(t, Any[1, 2.0, 3f0])
    @test eltype(x) ≡ Float64
end

@testset "inverse error on complex elements" begin
    t = as(Vector, asℝ₊, 3)
    @test_throws InexactError inverse(t, fill(Complex(0, 1), 3))
end

####
#### static analysis with JET
####

import JET
JET.test_package(TransformVariables; target_modules = (TransformVariables,))
