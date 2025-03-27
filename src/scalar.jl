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

function inverse_at!(x::AbstractVector, index::Int, t::ScalarTransform, y::Real)
    x[index] = inverse(t, y)
    index + 1
end

function inverse_eltype(t::ScalarTransform, y::Real)
    # NOTE this is a shortcut to get sensible types for all subtypes of ScalarTransform, which
    # we test for. If it breaks it should be extended accordingly.
    return Base.promote_typejoin_union(Base.promote_op(inverse, typeof(t), typeof(y)))
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

transform_and_logjac(::Identity, x::Real) = x, zero(x)

inverse(::Identity, x::Real) = x

####
#### shifted exponential
####

"""
$(TYPEDEF)

Shifted exponential. When `D::Bool == true`, maps to `(shift, ∞)` using `x ↦
shift + eˣ`, otherwise to `(-∞, shift)` using `x ↦ shift - eˣ`.
"""
struct ShiftedExp{D, T <: Real} <: ScalarTransform
    shift::T
    function ShiftedExp{D,T}(shift::T) where {D, T <: Real}
        @argcheck D isa Bool
        new(shift)
    end
end

ShiftedExp(D::Bool, shift::T) where {T <: Real} = ShiftedExp{D,T}(shift)

transform(t::ShiftedExp{D}, x::Real) where D =
    D ? t.shift + exp(x) : t.shift - exp(x)

transform_and_logjac(t::ShiftedExp, x::Real) = transform(t, x), x

function inverse(t::ShiftedExp{D}, x::Real) where D
    (; shift) = t
    if D
        @argcheck x > shift DomainError
        log(x - shift)
    else
        @argcheck x < shift DomainError
        log(shift - x)
    end
end

####
#### scaled and shifted logistic
####

"""
$(TYPEDEF)

Maps to `(scale, shift + scale)` using `logistic(x) * scale + shift`.
"""
struct ScaledShiftedLogistic{T <: Real} <: ScalarTransform
    scale::T
    shift::T
    function ScaledShiftedLogistic{T}(scale::T, shift::T) where {T <: Real}
        @argcheck scale > 0
        new(scale, shift)
    end
end

ScaledShiftedLogistic(scale::T, shift::T) where {T <: Real} =
    ScaledShiftedLogistic{T}(scale, shift)

ScaledShiftedLogistic(scale::Real, shift::Real) =
    ScaledShiftedLogistic(promote(scale, shift)...)

# Switch to muladd and now it does have a DiffRule defined
transform(t::ScaledShiftedLogistic, x::Real) = muladd(logistic(x), t.scale, t.shift)

transform_and_logjac(t::ScaledShiftedLogistic, x) =
    transform(t, x), log(t.scale) + logistic_logjac(x)

function inverse(t::ScaledShiftedLogistic, y)
    @argcheck y > t.shift           DomainError
    @argcheck y < t.scale + t.shift DomainError
    logit((y - t.shift)/t.scale)
end

# NOTE: inverse_and_logjac interface experimental and sporadically implemented for now
function inverse_and_logjac(t::ScaledShiftedLogistic, y)
    @argcheck y > t.shift           DomainError
    @argcheck y < t.scale + t.shift DomainError
    z = (y - t.shift) / t.scale
    logit(z), logit_logjac(z) - log(t.scale)
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

Base.show(::Infinity{T}) where T = print(io, T ? "∞" : "-∞")

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

as(::Type{Real}, left::Real, ::Infinity{true}) = ShiftedExp(true, left)

as(::Type{Real}, ::Infinity{false}, right::Real) = ShiftedExp(false, right)

function as(::Type{Real}, left::Real, right::Real)
    @argcheck left < right "the interval ($(left), $(right)) is empty"
    ScaledShiftedLogistic(right - left, left)
end

"""
Transform to a positive real number. See [`as`](@ref).

`asℝ₊` and `as_positive_real` are equivalent alternatives.
"""
const asℝ₊ = as(Real, 0, ∞)

const as_positive_real = asℝ₊

"""
Transform to a negative real number. See [`as`](@ref).

`asℝ₋` and `as_negative_real` are equivalent alternatives.
"""
const asℝ₋ = as(Real, -∞, 0)

const as_negative_real = asℝ₋

"""
Transform to the unit interval `(0, 1)`. See [`as`](@ref).

`as𝕀` and `as_unit_interval` are equivalent alternatives.
"""
const as𝕀 = as(Real, 0, 1)

const as_unit_interval = as𝕀

"""
Transform to the real line (identity). See [`as`](@ref).

`asℝ` and `as_real` are equivalent alternatives.
"""
const asℝ = as(Real, -∞, ∞)

const as_real = asℝ

function Base.show(io::IO, t::ShiftedExp)
    if t === asℝ₊
        print(io, "asℝ₊")
    elseif t === asℝ₋
        print(io, "asℝ₋")
    elseif t isa ShiftedExp{true}
        print(io, "as(Real, ", t.shift, ", ∞)")
    else
        print(io, "as(Real, -∞, ", t.shift, ")")
    end
end

function Base.show(io::IO, t::ScaledShiftedLogistic)
    if t === as𝕀
        print(io, "as𝕀")
    else
        print(io, "as(Real, ", t.shift, ", ", t.shift + t.scale, ")")
    end
end

Base.show(io::IO, t::Identity) = print(io, "asℝ")
