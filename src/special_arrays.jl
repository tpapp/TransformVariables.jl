export to_unitvec, to_corr_cholesky

"""
    (y, r, ℓ) = $SIGNATURES

Given ``x ∈ ℝ`` and ``0 ≤ r ≤ 1``, return `(y, r′)` such that

1. ``y² + r′² = r²``,

2. ``y: |y| ≤ r`` is mapped with a bijection from `x`.

`ℓ` is the log Jacobian (whether it is evaluated depends on `flag`).
"""
@inline function l2_remainder_transform(flag::LogJacFlag, x, r)
    z = 2*logistic(x) - 1
    (z * √r, r*(1 - abs2(z)),
     flag isa NoLogJac ? flag : log(2) + logistic_logjac(x) + 0.5*log(r))
end

"""
    (x, r′) = $SIGNATURES

Inverse of [`l2_remainder_transform`](@ref) in `x` and `y`.
"""
@inline l2_remainder_inverse(y, r) = logit((y/√r+1)/2), r-abs2(y)


"""
    UnitVector(n)

Transform `n-1` real numbers to a unit vector of length `n`, under the
Euclidean norm.
"""
struct UnitVector <: TransformReals
    n::Int
    function UnitVector(n::Int)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

"""
$(SIGNATURES)

Return a transformation that transforms `n - 1` real numbers to a unit vector
(under Euclidean norm).
"""
to_unitvec(n) = UnitVector(n)

dimension(t::UnitVector) = t.n - 1

function transform_with(flag::LogJacFlag, t::UnitVector, x::RealVector{T}) where T
    @unpack n = t
    r = one(T)
    y = Vector{T}(undef, n)
    ℓ = logjac_zero(flag, T)
    index = 1
    @inbounds for i in 1:(n - 1)
        xi = x[index]
        index += 1
        y[i], r, ℓi = l2_remainder_transform(flag, xi, r)
        ℓ += ℓi
    end
    y[end] = √r
    y, ℓ
end

function inverse(t::UnitVector, y::AbstractVector{T}) where T
    @unpack n = t
    @argcheck length(y) == n
    r = one(T)
    x = Vector{T}(undef, n - 1)
    @inbounds for (xi, yi) in zip(1:(n - 1), axes(y, 1))
        x[xi], r = l2_remainder_inverse(y[yi], r)
    end
    x
end


# correlation cholesky factor

"""
    CorrelationCholeskyFactor(n)

Cholesky factor of a correlation matrix of size `n`. See
[`to_corr_cholesky`](@ref) for details.
"""
struct CorrelationCholeskyFactor <: TransformReals
    n::Int
    function CorrelationCholeskyFactor(n)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

"""
$(SIGNATURES)

Return a transformation that transforms real numbers to an ``n×n``
upper-triangular matrix `Ω`, such that `Ω'*Ω` is a correlation matrix (positive
definite, with unit diagonal).
"""
to_corr_cholesky(n) = CorrelationCholeskyFactor(n)

dimension(t::CorrelationCholeskyFactor) = unit_triangular_dimension(t.n)

function transform_with( flag::LogJacFlag, t::CorrelationCholeskyFactor,
                         x::RealVector{T}) where T
    @unpack n = t
    ℓ = logjac_zero(flag, T)
    U = zeros(T, n, n)
    index = 1
    @inbounds for col in 1:n
        r = one(T)
        for row in 1:(col-1)
            xi = x[index]
            U[row, col], r, ℓi = l2_remainder_transform(flag, xi, r)
            ℓ += ℓi
            index += 1
        end
        U[col, col] = √r
    end
    UpperTriangular(U), ℓ
end

function inverse(t::CorrelationCholeskyFactor,
                 U::UpperTriangular{T}) where T
    @unpack n = t
    @argcheck size(U, 1) == n
    x = Vector{T}(undef, dimension(t))
    index = 1
    @inbounds for col in 1:n
        r = one(T)
        for row in 1:(col-1)
            x[index], r = l2_remainder_inverse(U[row, col], r)
            index += 1
        end
    end
    x
end
