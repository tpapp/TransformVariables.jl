export UnitVector, unit_vector_norm, UnitSimplex, CorrCholeskyFactor, corr_cholesky_factor

####
#### building blocks
####

"""
$(SIGNATURES)

Return a `NamedTuple` of

- `log_l2_rem`, for `log(1 - tanh(x)^2)`,

- `logjac`, for `log(abs( ∂(log(abs(tanh(x))) / ∂x ))`

Caller ensures that `x ≥ 0`. `x == 0` is handled correctly, but results in infinities.
"""
function tanh_helpers(x)
    d = 2*x
    log_denom = log1pexp(d)             # log(exp(2x) + 1)
    logjac = log(4) + d - 2 * log_denom # log(ab
    log_l2_rem = 2*(log(2) + x - log_denom)   # log(2exp(x) / (exp(2x) + 1))
    (; logjac, log_l2_rem)
end

"""
    (y, log_r, ℓ) = $(FUNCTIONNAME)(flag, x, log_r)

Given ``x ∈ ℝ`` and ``0 ≤ r ≤ 1``, we define `(y, r′)` such that

1. ``y² + (r′)² = r²``,

2. ``y: |y| ≤ r`` is mapped with a bijection from `x`, with the sign depending on `x`,

but use `log(r)` for actual calculations so that large `y`s still give nonsingular results.

`ℓ` is the log Jacobian (whether it is evaluated depends on `flag`).
"""
@inline function l2_remainder_transform(flag::LogJacFlag, x, log_r)
    (; logjac, log_l2_rem) = tanh_helpers(x)
    # note that 1-tanh(x)^2 = sech(x)^2
    (tanh(x) * exp(log_r / 2),
     log_r + log_l2_rem,
     flag isa NoLogJac ? flag : logjac + 0.5*log_r)
end

"""
    (x, r′) = $(FUNCTIONNAME)(y, log_r)

Inverse of [`l2_remainder_transform`](@ref) in `x` and `y`.
"""
@inline function l2_remainder_inverse(y, log_r)
    x = atanh(y / exp(log_r / 2))
    log_r′ = logsubexp(log_r, 2 * log(abs(y)))
    x, log_r′
end

####
#### UnitVector
####

"""
    UnitVector(n)

Transform `n-1` real numbers to a unit vector of length `n`, under the
Euclidean norm.
"""
struct UnitVector <: VectorTransform
    n::Int
    function UnitVector(n::Int)
        Base.depwarn("UnitVector is deprecated. See `unit_vector_norm`.", :UnitVector)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

dimension(t::UnitVector) = t.n - 1

function _summary_rows(transformation::UnitVector, mime)
    _summary_row(transformation, "$(transformation.n) element unit vector transformation")
end

function transform_with(flag::LogJacFlag, t::UnitVector, x::AbstractVector, index)
    (; n) = t
    T = _ensure_float(eltype(x))
    log_r = zero(T)
    y = Vector{T}(undef, n)
    ℓ = logjac_zero(flag, T)
    @inbounds for i in 1:(n - 1)
        xi = x[index]
        index += 1
        y[i], log_r, ℓi = l2_remainder_transform(flag, xi, log_r)
        ℓ += ℓi
    end
    y[end] = exp(log_r / 2)
    y, ℓ, index
end

function inverse_eltype(t::UnitVector,
                        ::Type{T}) where T <: AbstractVector
    _ensure_float(eltype(T))
end

function inverse_at!(x::AbstractVector, index, t::UnitVector, y::AbstractVector)
    (; n) = t
    @argcheck length(y) == n
    log_r = zero(eltype(y))
    @inbounds for yi in axes(y, 1)[1:(end-1)]
        x[index], log_r = l2_remainder_inverse(y[yi], log_r)
        index += 1
    end
    index
end

####
#### unit_vector_norm
####

struct UnitVectorNorm <: VectorTransform
    n::Int
    chi_prior::Bool
    function UnitVectorNorm(n::Int; chi_prior::Bool = true)
        @argcheck n ≥ 2 "Dimension should be at least 2."
        new(n, chi_prior)
    end
end

"""
$(SIGNATURES)

Transform `n ≥ 2` real numbers to a unit vector of length `n` and a radius, under the
Euclidean norm. Returns the tuple `(normalized_vector, radius)`.

When `chi_prior = true`, a prior correction is applied to the radius, which only
affects the log Jacobian determinant. The purpose of this is to make the
distribution proper. If you wish to use another prior, set this to `false` and use
a manual correction, see also [`logprior`](@ref).

!!! note
    At the origin, this transform is non-bijective and non-differentiable. If
    maximizing a target distribution whose density is constant for the unit vector,
    then the maximizer using the Chi prior is at the origin, and behavior is undefined.

!!! note
    While ``n = 1`` would be technically possible, for practical purposes it would
    likely suffer from numerical issues, since the transform is undefined at ``x = 0``,
    and for a Markov chain to travel from ``y=[-1]`` to ``y=[1]``, it would have to leap
    over the origin, which is only even possible due to discretization and likely will
    often not work. Because of this, it is disallowed.
"""
unit_vector_norm(n::Int; chi_prior::Bool = true) = UnitVectorNorm(n; chi_prior)

nonzero_logprior(t::UnitVectorNorm) = t.chi_prior

function logprior(t::UnitVectorNorm, (y, r)::Tuple{AbstractVector,Real})
    (; n, chi_prior) = t
    if chi_prior
        (t.n - 1) * log(r) - r^2 / 2
    else
        float(zero(r))
    end
end

dimension(t::UnitVectorNorm) = t.n

function _summary_rows(t::UnitVectorNorm, mime)
    _summary_row(t, "$(t.n) element (unit vector, norm) transformation")
end

function transform_with(flag::LogJacFlag, t::UnitVectorNorm, x::AbstractVector, index)
    (; n, chi_prior) = t
    T = _ensure_float(eltype(x))
    log_r = zero(T)
    y = Vector{T}(undef, n)
    copyto!(y, 1, x, index, n)
    r = norm(y, 2)
    __normalize!(y, r)
    ℓ = flag isa NoLogJac ? flag : (chi_prior ? -r^2 / 2 : -(t.n - 1) * log(r))
    index += n
    (y, r), ℓ, index
end

function inverse_eltype(t::UnitVectorNorm,
                        ::Type{Tuple{V,T}}) where {V <: AbstractVector,T}
    _ensure_float(eltype(T))
end

function inverse_at!(x::AbstractVector, index, t::UnitVectorNorm,
                     (y, r)::Tuple{AbstractVector,Real})
    (; n) = t
    @argcheck length(y) == n
    @argcheck r ≥ 0
    _x = @view x[index:(index + n - 1)]
    if r == 0
        fill!(_x, zero(eltype(x)))
    else
        copyto!(_x, y)
        yN = norm(y, 2)
        @argcheck isapprox(yN, 1; atol = √eps(r) * n) # somewhat generous tolerance
        __normalize!(_x, yN / r)
    end
    index + n
end

####
#### UnitSimplex
####

"""
    UnitSimplex(n)

Transform `n-1` real numbers to a vector of length `n` whose elements are non-negative and sum to one.
"""
struct UnitSimplex <: VectorTransform
    n::Int
    function UnitSimplex(n::Int)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

function _summary_rows(transformation::UnitSimplex, mime)
    _summary_row(transformation, "$(transformation.n) element unit simplex transformation")
end

dimension(t::UnitSimplex) = t.n - 1

function transform_with(flag::LogJacFlag, t::UnitSimplex, x::AbstractVector, index)
    (; n) = t
    T = _ensure_float(eltype(x))
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

function inverse_eltype(t::UnitSimplex,
                        ::Type{T}) where T <: AbstractVector
    _ensure_float(eltype(T))
end

function inverse_at!(x::AbstractVector, index, t::UnitSimplex, y::AbstractVector)
    (; n) = t
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

!!! note
    It is better style to use [`corr_cholesky_factor`](@ref), this will be deprecated.

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
struct CorrCholeskyFactor <: VectorTransform
    n::Int
    function CorrCholeskyFactor(n)
        @argcheck n ≥ 1 "Dimension should be positive."
        new(n)
    end
end

function _summary_rows(transformation::CorrCholeskyFactor, mime)
    (; n) = transformation
    _summary_row(transformation, "$(n)×$(n) correlation cholesky factor")
end

"""
$(SIGNATURES)

Transform into a Cholesky factor of a correlation matrix.

If the argument is a (positive) integer `n`, it determines the size of the output `n × n`,
resulting in a `Matrix`.

If the argument is `SMatrix{N,N}`, an `SMatrix` is produced.
"""
function corr_cholesky_factor(n::Int)
    @argcheck n ≥ 1 "Dimension should be positive."
    CorrCholeskyFactor(n)
end

dimension(t::CorrCholeskyFactor) = unit_triangular_dimension(t.n)

result_size(transformation::CorrCholeskyFactor) = transformation.n

"""
Static version of cholesky correlation factor.
"""
struct StaticCorrCholeskyFactor{D,S} <: VectorTransform end

function _summary_rows(transformation::StaticCorrCholeskyFactor{D,S}, mime) where {D,S}
    _summary_row(transformation, "SMatrix{$(S),$(S)} correlation cholesky factor")
end


result_size(::StaticCorrCholeskyFactor{D,S}) where {D,S} = S

function corr_cholesky_factor(::Type{SMatrix{S,S}}) where S
    D = unit_triangular_dimension(S)
    StaticCorrCholeskyFactor{D,S}()
end

dimension(transformation::StaticCorrCholeskyFactor{D}) where D = D

"""
$(SIGNATURES)

Implementation of Cholesky factor calculation.
"""
function calculate_corr_cholesky_factor!(U::AbstractMatrix{T}, flag::LogJacFlag,
                                          x::AbstractVector, index::Int) where {T<:Real}
    n = size(U, 1)
    ℓ = logjac_zero(flag, T)
    @inbounds for col_index in 1:n
        log_r = zero(T)
        for row_index in 1:(col_index-1)
            xi = x[index]
            U[row_index, col_index], log_r, ℓi = l2_remainder_transform(flag, xi, log_r)
            ℓ += ℓi
            index += 1
        end
        U[col_index, col_index] = exp(log_r / 2)
    end
    U, ℓ, index
end

function transform_with(flag::LogJacFlag, t::CorrCholeskyFactor, x::AbstractVector, index)
    n = result_size(t)
    T = _ensure_float(eltype(x))
    U, ℓ, index′ = calculate_corr_cholesky_factor!(Matrix{T}(undef, n, n),
                                                   flag, x, index)
    UpperTriangular(U), ℓ, index′
end

@generated function calculate_corr_cholesky_factor(::Type{T}, flag::LogJacFlag,
                                                   t::StaticCorrCholeskyFactor{D,S},
                                                   x::AbstractVector, index) where {T,D,S}
    exprs = [:(ℓ = logjac_zero(flag, T)), :(z = zero(T))]
    u(row, col) = row ≤ col ? Symbol("u_", row, "_", col) : :z
    for col in 1:S
        push!(exprs, :(log_r = z))
        # above diagonal
        for row in 1:(col-1)
            push!(exprs,
                  :(($(u(row, col)), log_r, Δℓ) = l2_remainder_transform(flag, x[index], log_r)),
                  :(ℓ += Δℓ),
                  :(index += 1))
        end
        # diagonal
        push!(exprs, :($(u(col, col)) = exp(log_r / 2)))
    end
    U_elements = (u(row, col) for row in 1:S, col in 1:S)
    push!(exprs, :(UpperTriangular(SMatrix{$S,$S}($(U_elements...))), ℓ, index))
    Expr(:block, exprs...)
end

function transform_with(flag::LogJacFlag, transformation::StaticCorrCholeskyFactor{D,S},
                        x::AbstractVector, index) where {D,S}
    T = _ensure_float(eltype(x))
    calculate_corr_cholesky_factor(T, flag, transformation, x, index)
end

function inverse_eltype(t::Union{CorrCholeskyFactor,StaticCorrCholeskyFactor},
                        ::Type{T}) where {T<:UpperTriangular}
    _ensure_float(eltype(T))
end

function inverse_at!(x::AbstractVector, index,
                     t::Union{CorrCholeskyFactor,StaticCorrCholeskyFactor}, U::UpperTriangular)
    n = result_size(t)
    @argcheck size(U, 1) == n
    @inbounds for col in 1:n
        log_r = zero(eltype(U))
        for row in 1:(col-1)
            x[index], log_r = l2_remainder_inverse(U[row, col], log_r)
            index += 1
        end
    end
    index
end
