export to_unitvec, to_corr_cholesky

"""
    (y, r) = $SIGNATURES

Given ``x ∈ ℝ`` and ``0 ≤ r ≤ 1``, return `(y, r′)` such that

1. ``y² + r′² = r²``,

2. ``y: |y| ≤ r`` is mapped with a bijection from `x`.
"""
@inline function l2_remainder_transform(x, r)
    z = 2*logistic(x) - 1
    z * √r, r*(1 - abs2(z))
end

"""
    $SIGNATURES

The log Jacobian determinant for `y(x)` in [`l2_remainder_transform`].
"""
@inline l2_remainder_logjac(x, r) = log(2) + logistic_logjac(x) + 0.5*log(r)

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

to_unitvec(n) = UnitVector(n)

dimension(t::UnitVector) = t.n - 1

function transform_at(t::UnitVector, ::LogJac, x::RealVector{T},
                      index::Int) where T
    @unpack n = t
    r = one(T)
    ℓ = zero(T)
    y = Vector{T}(undef, n)
    for i in 1:(n - 1)
        xi = x[index]
        index += 1
        ℓ += l2_remainder_logjac(xi, r)
        y[i], r = l2_remainder_transform(xi, r)
    end
    y[end] = √r
    y, ℓ
end

transform_at(t::UnitVector, x::RealVector, index::Int) =
    first(transform_at(t, LOGJAC, x, index))

function inverse(t::UnitVector, y::AbstractVector{T}) where T
    @unpack n = t
    @argcheck length(y) == n
    r = one(T)
    x = Vector{T}(undef, n - 1)
    for (xi, yi) in zip(1:(n - 1), axes(y, 1))
        x[xi], r = l2_remainder_inverse(y[yi], r)
    end
    x
end


# correlation cholesky factor

"""
    CorrelationCholeskyFactor(n)

Cholesky factor of a correlation matrix of size `n`.
"""
struct CorrelationCholeskyFactor <: TransformReals
    n::Int
    function CorrelationCholeskyFactor(n)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

to_corr_cholesky(n) = CorrelationCholeskyFactor(n)

dimension(t::CorrelationCholeskyFactor) = unit_triangular_dimension(t.n)

function transform_at(t::CorrelationCholeskyFactor, ::LogJac,
                      x::RealVector{T}, index::Int) where T
    @unpack n = t
    lj = zero(T)
    U = zeros(T, n, n)
    for col in 1:n
        r = one(T)
        for row in 1:(col-1)
            xi = x[index]
            lj += l2_remainder_logjac(xi, r)
            U[row, col], r = l2_remainder_transform(xi, r)
            index += 1
        end
        U[col, col] = √r
    end
    UpperTriangular(U), lj
end

transform_at(t::CorrelationCholeskyFactor, x::RealVector, index::Int) =
    first(transform_at(t, LOGJAC, x, index))

function inverse(t::CorrelationCholeskyFactor,
                 U::UpperTriangular{T}) where T
    @unpack n = t
    @argcheck size(U, 1) == n
    x = Vector{T}(undef, unit_triangular_dimension(n))
    index = 1
    for col in 1:n
        r = one(T)
        for row in 1:(col-1)
            x[index], r = l2_remainder_inverse(U[row, col], r)
            index += 1
        end
    end
    x
end
