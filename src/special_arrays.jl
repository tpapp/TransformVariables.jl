export UnitVector, UnitSimplex, CorrCholeskyFactor

####
#### building blocks
####

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

####
#### UnitVector
####

"""
    UnitVector(n)

Transform `n-1` real numbers to a unit vector of length `n`, under the
Euclidean norm.
"""
@calltrans struct UnitVector <: VectorTransform
    n::Int
    function UnitVector(n::Int)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

dimension(t::UnitVector) = t.n - 1

function transform_with(flag::LogJacFlag, t::UnitVector, x::AbstractVector, index)
    @unpack n = t
    T = extended_eltype(x)
    r = one(T)
    y = Vector{T}(undef, n)
    ℓ = logjac_zero(flag, T)
    @inbounds for i in 1:(n - 1)
        xi = x[index]
        index += 1
        y[i], r, ℓi = l2_remainder_transform(flag, xi, r)
        ℓ += ℓi
    end
    y[end] = √r
    y, ℓ, index
end

inverse_eltype(t::UnitVector, y::AbstractVector) = extended_eltype(y)

function inverse_at!(x::AbstractVector, index, t::UnitVector, y::AbstractVector)
    @unpack n = t
    @argcheck length(y) == n
    r = one(eltype(y))
    @inbounds for yi in axes(y, 1)[1:(end-1)]
        x[index], r = l2_remainder_inverse(y[yi], r)
        index += 1
    end
    index
end


####
#### UnitSimplex
####

"""
    UnitSimplex(n)

Transform `n-1` real numbers to a vector of length `n` whose elements are non-negative and sum to one.
"""
@calltrans struct UnitSimplex <: VectorTransform
    n::Int
    function UnitSimplex(n::Int)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

dimension(t::UnitSimplex) = t.n - 1

function transform_with(flag::LogJacFlag, t::UnitSimplex, x::AbstractVector, index)
    @unpack n = t
    T = extended_eltype(x)

    ℓ = logjac_zero(flag, T)
    stick = one(T)
    y = Vector{T}(undef, n)
    @inbounds for i in 1:n-1
        xi = x[index]
        index += 1
        z = logistic(xi - log(n-i))
        y[i] = z * stick

        if !(flag isa NoLogJac)
            ℓ += log(stick) - logit_logjac(z)
        end

        stick *= 1 - z
    end

    y[end] = stick

    y, ℓ, index
end

inverse_eltype(t::UnitSimplex, y::AbstractVector) = extended_eltype(y)

function inverse_at!(x::AbstractVector, index, t::UnitSimplex, y::AbstractVector)
    @unpack n = t
    @argcheck length(y) == n

    stick = one(eltype(y))
    @inbounds for i in axes(y, 1)[1:end-1]
        z = y[i]/stick
        x[index] = logit(z) + log(n-i)
        stick -= y[i]
        index += 1
    end
    index
end

####
#### correlation cholesky factor
####

"""
    CorrCholeskyFactor(n)

Cholesky factor of a correlation matrix of size `n`.

Transforms ``n×(n-1)/2`` real numbers to an ``n×n`` upper-triangular matrix `U`, such that
`U'*U` is a correlation matrix (positive definite, with unit diagonal).

# Notes

If

- `z` is a vector of `n` IID standard normal variates,

- `σ` is an `n`-element vector of standard deviations,

- `U` is obtained from `CorrCholeskyFactor(n)`,

then `Diagonal(σ) * U' * z` will be a multivariate normal with the given variances and
correlation matrix `U' * U`.
"""
@calltrans struct CorrCholeskyFactor <: VectorTransform
    n::Int
    function CorrCholeskyFactor(n)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

dimension(t::CorrCholeskyFactor) = unit_triangular_dimension(t.n)

function transform_with(flag::LogJacFlag, t::CorrCholeskyFactor, x::AbstractVector, index)
    @unpack n = t
    T = extended_eltype(x)
    ℓ = logjac_zero(flag, T)
    U = Matrix{T}(undef, n, n)
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
    UpperTriangular(U), ℓ, index
end

inverse_eltype(t::CorrCholeskyFactor, U::UpperTriangular) = extended_eltype(U)

function inverse_at!(x::AbstractVector, index, t::CorrCholeskyFactor, U::UpperTriangular)
    @unpack n = t
    @argcheck size(U, 1) == n
    @inbounds for col in 1:n
        r = one(eltype(U))
        for row in 1:(col-1)
            x[index], r = l2_remainder_inverse(U[row, col], r)
            index += 1
        end
    end
    index
end
