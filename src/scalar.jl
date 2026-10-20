export TVExp, TVScale, TVShift, TVLogistic, TVNeg, TVSoftPlus, TVCauchy

export ∞, asℝ, asℝ₊, asℝ₋, as𝕀, as_real, as_positive_real, as_negative_real,
    as_unit_interval

"""
$(TYPEDEF)

Transform a scalar (real number) to another scalar.

Subtypes must define `transform`, `transform_and_logjac`, and `inverse`.
Other methods of of the interface should have the right defaults.

!!! note
    This type is for code organization within the package, and is not part of the public API.
"""
abstract type ScalarTransform <: AbstractTransform end

dimension(::ScalarTransform) = 1

function transform_with(flag::NoLogJac, t::ScalarTransform, x::AbstractVector, index)
    transform(t, @inbounds tv_getindex(x, index)), flag, index + 1
end

function transform_with(::LogJac, t::ScalarTransform, x::AbstractVector, index)
    transform_and_logjac(t, @inbounds tv_getindex(x, index))..., index + 1
end

function inverse_at!(x::AbstractVector, index::Int, t::ScalarTransform, y)
    x[index] = inverse(t, y)
    index + 1
end

_domain_label(::ScalarTransform, index::Int) = ()

####
#### identity
####

"""
$(TYPEDEF)

Identity ``x ↦ x``.
"""
struct Identity <: ScalarTransform end

transform(::Identity, x::Number) = x

transform_and_logjac(::Identity, x::Number) = x, logjac_zero(LogJac(), typeof(x))

inverse_eltype(t::Identity, ::Type{T}) where T = T

inverse(::Identity, x::Number) = x

inverse_and_logjac(::Identity, x::Number) = x, logjac_zero(LogJac(), typeof(x))

####
#### elementary scalar transforms
####

"""
$(TYPEDEF)

Exponential transformation `x ↦ eˣ`. Maps from all reals to the positive reals.
"""
struct TVExp <: ScalarTransform end

transform(::TVExp, x::Number) = exp(x)

transform_and_logjac(t::TVExp, x::Number) = transform(t, x), x

inverse_eltype(t::TVExp, ::Type{T}) where T = _ensure_float(T)

function inverse(::TVExp, x::Number)
    log(x)
end

function inverse_and_logjac(t::TVExp, x::Number)
    log_x = log(x)
    return log_x, -log_x
end

"""
$(TYPEDEF)

Logistic transformation `x ↦ logit(x)`. Maps from all reals to (0, 1).
"""
struct TVLogistic <: ScalarTransform end

transform(::TVLogistic, x::Number) = logistic(x)

transform_and_logjac(t::TVLogistic, x::Number) = transform(t, x), logistic_logjac(x)

inverse_eltype(t::TVLogistic, ::Type{T}) where T = _ensure_float(T)

function inverse(::TVLogistic, x::Number)
    logit(x)
end

inverse_and_logjac(t::TVLogistic, x::Number) = inverse(t, x), logit_logjac(x)

"""
$(TYPEDEF)

Shift transformation `x ↦ x + shift`.
"""
struct TVShift{T} <: ScalarTransform
    shift::T
end

transform(t::TVShift, x::Number) = x + t.shift

transform_and_logjac(t::TVShift, x::Number) = transform(t, x), logjac_zero(LogJac(), typeof(x))

inverse_eltype(t::TVShift{S}, ::Type{T}) where {S,T} = typeof(zero(_ensure_float(T)) - zero(S))

inverse(t::TVShift, x::Number) = x - t.shift

inverse_and_logjac(t::TVShift, x::Number) = inverse(t, x), logjac_zero(LogJac(), typeof(x))

"""
$(TYPEDEF)

Scale transformation `x ↦ scale * x`.
"""
struct TVScale{T} <: ScalarTransform
    scale::T
    function TVScale{T}(scale::T) where {T}
        @argcheck scale > zero(scale) DomainError
        new(scale)
    end
end

TVScale(scale::T) where {T} = TVScale{T}(scale)

transform(t::TVScale, x::Number) = t.scale * x

transform_and_logjac(t::TVScale{<:Real}, x::Number) = transform(t, x), log(t.scale)

inverse_eltype(t::TVScale{S}, ::Type{T}) where {S,T} = typeof(oneunit(T) / oneunit(S))

inverse(t::TVScale, x::Number) = x / t.scale

inverse_and_logjac(t::TVScale, x::Number) = inverse(t, x), -log(t.scale)

"""
$(TYPEDEF)

Negative transformation `x ↦ -x`.
"""
struct TVNeg <: ScalarTransform
end

transform(::TVNeg, x::Number) = -x
transform_and_logjac(t::TVNeg, x::Number) = transform(t, x), logjac_zero(LogJac(), typeof(x))

inverse_eltype(::TVNeg, ::Type{T}) where T = typeof(-oneunit(T))
inverse(::TVNeg, x::Number) = -x
inverse_and_logjac(::TVNeg, x::Number) = -x, logjac_zero(LogJac(), typeof(x))

"""
$(TYPEDEF)

“Softplus” transformation ``x ↦ log(1+exp(x))``, mapping the real line to positive
numbers.
"""
struct TVSoftPlus <: ScalarTransform end

transform(::TVSoftPlus, x::Number) = log1pexp(x)
transform_and_logjac(t::TVSoftPlus, x::Number) = transform(t, x), -log1pexp(-x)

inverse_eltype(::TVSoftPlus, ::Type{T}) where T = _ensure_float(T)
inverse(::TVSoftPlus, y::Number) = logexpm1(y)
inverse_and_logjac(::TVSoftPlus, y::Number) = logexpm1(y), -log1mexp(-y)

"""
$(TYPEDEF)

Transformation ``x ↦ 1/2 + arctan(x)/π``, mapping the real line to `[0,1]`, which is
also the CDF of the Cauchy distribution.
"""
struct TVCauchy <: ScalarTransform end

function _cauchy_cdf(x::T) where {T<:Real}
    _π = T(π)
    _x = float(x)
    if _x > -1
        one(_x) / 2 + atan(_x) / _π
    else
        # avoid 1/2 - 1/2 cancellations
        atan(inv(-_x)) / _π
    end
end

transform(::TVCauchy, x::Number) = _cauchy_cdf(float(x))

function transform_and_logjac(t::TVCauchy, x::Number)
    y = transform(t, x)
    y,  -log(oftype(y, π)) - log1psq(x)
end

inverse_eltype(::TVCauchy, ::Type{T}) where T = _ensure_float(T)

function inverse(::TVCauchy, y::Number)
    @argcheck 0 <= y <= 1 DomainError
    cotpi(p) = cospi(p) / sinpi(p)
    y > 0.5 ? cotpi(1 - y) : -cotpi(y)
end

function inverse_and_logjac(t::TVCauchy, y::Number)
    x = inverse(t, y)
    x, log(oftype(x, π)) - 2 * log(sinpi(y))
end

####
#### composite scalar transforms
####

"""
$(TYPEDEF)

A composite scalar transformation, i.e. a sequence of scalar transformations.
"""
struct CompositeScalarTransform{Ts <: Tuple} <: ScalarTransform
    transforms::Ts
    function CompositeScalarTransform(transforms::Ts) where {Ts <: Tuple{ScalarTransform,Vararg{ScalarTransform}}}
        new{Ts}(transforms)
    end
end

transform(t::CompositeScalarTransform, x) = foldr(transform, t.transforms, init=x)
function transform_and_logjac(ts::CompositeScalarTransform, x)
    foldr(ts.transforms, init=(x, logjac_zero(LogJac(), typeof(x)))) do t, (x, logjac)
        nx, nlogjac = transform_and_logjac(t, x)
        (nx, logjac + nlogjac)
    end
end

function inverse_eltype(ts::CompositeScalarTransform, ::Type{T}) where T
    foldl((T, t) -> inverse_eltype(t, T), ts.transforms; init = T)
end

function inverse(ts::CompositeScalarTransform, x)
    foldl((y, t) -> inverse(t, y), ts.transforms, init = x)
end

function inverse_and_logjac(ts::CompositeScalarTransform, x)
    foldl(ts.transforms, init=(x, logjac_zero(LogJac(), typeof(x)))) do (x, logjac), t
        nx, nlogjac = inverse_and_logjac(t, x)
        (nx, logjac + nlogjac)
    end
end

Base.:∘(t::ScalarTransform, s::ScalarTransform) = CompositeScalarTransform((t, s))
Base.:∘(t::ScalarTransform, ct::CompositeScalarTransform) = CompositeScalarTransform((t, ct.transforms...))
Base.:∘(ct::CompositeScalarTransform, t::ScalarTransform) = CompositeScalarTransform((ct.transforms..., t))
Base.:∘(ct1::CompositeScalarTransform, ct2::CompositeScalarTransform) = CompositeScalarTransform((ct1.transforms..., ct2.transforms...))
Base.:∘(t::ScalarTransform, tt::Vararg{ScalarTransform}) = foldl(∘, tt; init=t)

####
#### to_interval interface
####

struct Infinity{ispositive}
    Infinity{T}() where T = (@argcheck T isa Bool; new{T}())
end

"""
Placeholder representing of infinity for specifing interval boundaries. Supports
the `-` operator, ie `-∞`.
"""
const ∞ = Infinity{true}()

Base.show(io::IO, ::Infinity{T}) where T = print(io, T ? "∞" : "-∞")

Base.:(-)(::Infinity{T}) where T = Infinity{!T}()

"""
    as(Real, left, right)

Return a transformation that transforms a single real number to the given (open)
interval.

`left < right` is required, but may be `-∞` or `∞`, respectively, in which case
the appropriate transformation is selected. See [`∞`](@ref).

Some common transformations are predefined as constants, see [`asℝ`](@ref),
[`asℝ₋`](@ref), [`asℝ₊`](@ref), [`as𝕀`](@ref).

!!! note
    The finite arguments are promoted to a common type and affect promotion. Eg
    `transform(as(0, ∞), 0f0) isa Float32`, but `transform(as(0.0, ∞), 0f0) isa Float64`.
"""
as(::Type{Real}, left, right) =
    throw(ArgumentError("($(left), $(right)) must be an interval"))

as(::Type{Real}, ::Infinity{false}, ::Infinity{true}) = Identity()

as(::Type{Real}, left::Real, ::Infinity{true}) = TVShift(left) ∘ TVExp()

as(::Type{Real}, ::Infinity{false}, right::Real) = TVShift(right) ∘ TVNeg() ∘ TVExp()

function as(::Type{Real}, left::Real, right::Real)
    @argcheck left < right "the interval ($(left), $(right)) is empty"
    shift, scale = promote(left, right - left)
    TVShift(shift) ∘ TVScale(scale) ∘ TVLogistic()
end

"""
Transform to a positive real number. See [`as`](@ref).

`asℝ₊` and `as_positive_real` are equivalent alternatives.
"""
const asℝ₊ = TVExp()

const as_positive_real = asℝ₊

"""
Transform to a negative real number. See [`as`](@ref).

`asℝ₋` and `as_negative_real` are equivalent alternatives.
"""
const asℝ₋ = TVNeg() ∘ TVExp()

const as_negative_real = asℝ₋

"""
Transform to the unit interval `(0, 1)`. See [`as`](@ref).

`as𝕀` and `as_unit_interval` are equivalent alternatives.
"""
const as𝕀 = TVLogistic()

const as_unit_interval = as𝕀

"""
Transform to the real line (identity). See [`as`](@ref).

`asℝ` and `as_real` are equivalent alternatives.
"""
const asℝ = as(Real, -∞, ∞)

const as_real = asℝ

# Single scalar transforms
Base.show(io::IO, ::Identity) = print(io, "asℝ")
Base.show(io::IO, ::TVExp) = print(io, "asℝ₊")
Base.show(io::IO, ::TVLogistic) = print(io, "as𝕀")
function Base.show(io::IO, t::TVScale)
    print(io, "TVScale(", t.scale, ")")
end
function Base.show(io::IO, t::TVShift)
    print(io, "TVShift(", t.shift, ")")
end

# Fallback method: print all transforms in order
Base.show(io::IO, ct::CompositeScalarTransform) = join(io, ct.transforms, " ∘ ")

# Special cases which are constructed by as(Real, ...)
function Base.show(io::IO, ct::CompositeScalarTransform{Tuple{TVShift{T}, TVExp}}) where T
    print(io, "as(Real, ", ct.transforms[1].shift, ", ∞)")
end
function Base.show(io::IO, ct::CompositeScalarTransform{Tuple{TVShift{T}, TVNeg, TVExp}}) where T
    print(io, "as(Real, -∞, ", ct.transforms[1].shift, ")")
end
function Base.show(io::IO, ct::CompositeScalarTransform{Tuple{TVShift{T1}, TVScale{T2}, TVLogistic}}) where {T1, T2}
    print(io, "as(Real, ", ct.transforms[1].shift, ", ", ct.transforms[1].shift +
    ct.transforms[2].scale, ")")
end

# Special case for asR-
Base.show(io::IO, ::CompositeScalarTransform{Tuple{TVNeg, TVExp}}) = print(io, "asℝ₋")
