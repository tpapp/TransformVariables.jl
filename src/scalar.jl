export TVExp, TVScale, TVShift, TVLogistic, TVNeg
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

function transform_with(flag::NoLogJac, t::ScalarTransform, x::AbstractVector, index::Int)
    transform(t, @inbounds x[index]), flag, index + 1
end

function transform_with(::LogJac, t::ScalarTransform, x::AbstractVector, index::Int)
    transform_and_logjac(t, @inbounds x[index])..., index + 1
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

transform(::Identity, x::Real) = x

transform_and_logjac(::Identity, x::Real) = x, logjac_zero(LogJac(), typeof(x))

inverse_eltype(t::Identity, ::Type{T}) where T = T

inverse(::Identity, x::Number) = x

inverse_and_logjac(::Identity, x::Real) = x, logjac_zero(LogJac(), typeof(x))

####
#### elementary scalar transforms
####

"""
$(TYPEDEF)

Exponential transformation `x ↦ eˣ`. Maps from all reals to the positive reals.
"""
struct TVExp <: ScalarTransform end

transform(::TVExp, x::Real) = exp(x)

transform_and_logjac(t::TVExp, x::Real) = transform(t, x), x

inverse_eltype(t::TVExp, ::Type{T}) where T = _ensure_float(T)

function inverse(::TVExp, x::Number)
    log(x)
end

inverse_and_logjac(t::TVExp, x::Number) = inverse(t, x), -log(x)

"""
$(TYPEDEF)

Logistic transformation `x ↦ logit(x)`. Maps from all reals to (0, 1).
"""
struct TVLogistic <: ScalarTransform end

transform(::TVLogistic, x::Real) = logistic(x)

transform_and_logjac(t::TVLogistic, x::Real) = transform(t, x), logistic_logjac(x)

inverse_eltype(t::TVLogistic, ::Type{T}) where T = _ensure_float(T)

function inverse(::TVLogistic, x::Number)
    logit(x)
end

inverse_and_logjac(t::TVLogistic, x::Number) = inverse(t, x), logit_logjac(x)

"""
$(TYPEDEF)

Shift transformation `x ↦ x + shift`.
"""
struct TVShift{T <: Real} <: ScalarTransform
    shift::T
end

transform(t::TVShift, x::Real) = x + t.shift

transform_and_logjac(t::TVShift, x::Real) = transform(t, x), logjac_zero(LogJac(), typeof(x))

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

transform(t::TVScale, x::Real) = t.scale * x

transform_and_logjac(t::TVScale{<:Real}, x::Real) = transform(t, x), log(t.scale)

inverse_eltype(t::TVScale{S}, ::Type{T}) where {S,T} = typeof(oneunit(T) / oneunit(S))

inverse(t::TVScale, x::Number) = x / t.scale

inverse_and_logjac(t::TVScale{<:Real}, x::Number) = inverse(t, x), -log(t.scale)

"""
$(TYPEDEF)

Negative transformation `x ↦ -x`.
"""
struct TVNeg <: ScalarTransform
end

transform(::TVNeg, x::Real) = -x
transform_and_logjac(t::TVNeg, x::Real) = transform(t, x), logjac_zero(LogJac(), typeof(x))

inverse_eltype(::TVNeg, ::Type{T}) where T = typeof(-oneunit(T))
inverse(::TVNeg, x::Number) = -x
inverse_and_logjac(::TVNeg, x::Number) = -x, logjac_zero(LogJac(), typeof(x))

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
#### type-wrapped scalar transforms
####

"""
$(TYPEDEF)
"""
struct ScalarWrapperTransform{T, S} <: ScalarTransform
    inner::S
end

# Scalar inner transformation, forward and reverse
function as(::Type{T}, inner::S) where {T,S<:ScalarTransform}
    @argcheck isstructtype(T) && length(fieldtypes(T)) == 1
    ScalarWrapperTransform{T, typeof(inner)}(inner)
end

function transform(t::ScalarWrapperTransform{T, S}, x::Number) where {T, S}
    ctor = constructorof(T)
    y = transform(t.inner, x)
    ctor(y)
end
function transform_and_logjac(t::ScalarWrapperTransform{T, S}, x::Number) where {T, S<:ScalarTransform}
    ctor = constructorof(T)
    y, ℓ = transform_and_logjac(t.inner, x)
    ctor(y), ℓ 
end

function inverse(t::ScalarWrapperTransform{C, S}, y::T) where {C, T<:C, S}
    inverse(t.inner, getfield(y, 1))
end
function inverse_and_logjac(t::ScalarWrapperTransform{C, S}, y::T) where {C, T<:C, S}
    inverse_and_logjac(t.inner, getfield(y, 1))
end
# Informative error for trying to invert a different type
function inverse(t::ScalarWrapperTransform{C, S}, y::T) where {C, T, S}
    throw(ArgumentError("Cannot invert a value of type $T as if it were a $C"))
end
function inverse_and_logjac(t::ScalarWrapperTransform{C, S}, y::T) where {C, T, S}
    throw(ArgumentError("Cannot invert a value of type $T as if it were a $C"))
end

function inverse_eltype(t::ScalarWrapperTransform{C, S}, ::Type{T}) where {C, T<:C, S}
    inverse_eltype(t.inner, fieldtype(C, 1))
end
# Printing for scalar case
function Base.show(io::IO, t::ScalarWrapperTransform{T, S}) where {T, S}
    print(io, "as($T, ", t.inner, ")")
end

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
