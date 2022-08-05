export ∞, asℝ, asℝ₊, asℝ₋, as𝕀, as_real, as_positive_real, as_negative_real,
    as_unit_interval

"""
$(TYPEDEF)

Transform a scalar (real number) to another scalar.

Subtypes mustdefine `transform`, `transform_and_logjac`, and `inverse`; other
methods of of the interface should have the right defaults.
"""
abstract type ScalarTransform <: AbstractTransform end

dimension(::ScalarTransform) = 1

function transform_with(flag::NoLogJac, t::ScalarTransform, x::AbstractVector, index)
    transform(t, @inbounds x[index]), flag, index + 1
end

function transform_with(::LogJac, t::ScalarTransform, x::AbstractVector, index)
    transform_and_logjac(t, @inbounds x[index])..., index + 1
end

function inverse_at!(x::AbstractVector, index, t::ScalarTransform, y::Real)
    x[index] = inverse(t, y)
    index + 1
end

inverse_eltype(t::ScalarTransform, y::T) where {T <: Real} = float(T)

random_arg(t::ScalarTransform; kwargs...) = random_real(; kwargs...)

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

Shifted exponential. When `D::Bool == true`, maps to `(shift, scale,  ∞)` using `x ↦
shift + exp(x/scale)`, otherwise to `(-∞, shift)` using `x ↦ shift - exp(x/scale)`.
"""
struct ShiftedExp{D, T <: Real, S <: Real} <: ScalarTransform
    shift::T
    scale::S
    function ShiftedExp{D,T,S}(shift::T, scale::S) where {D, T <: Real, S <: Real}
        @argcheck D isa Bool
        new(shift, scale)
    end
end
ShiftedExp(D::Bool, shift::T) where {T <: Real} = ShiftedExp{D,T,T}(shift, one(T))
ShiftedExp(D::Bool, shift::T, scale::S) where {T <: Real, S <: Real} = ShiftedExp{D,T,S}(shift, scale)

transform(t::ShiftedExp{D}, x::Real) where D =
    D ? t.shift + exp(x/t.scale) : t.shift - exp(x/t.scale)

transform_and_logjac(t::ShiftedExp, x::Real) = transform(t, x), x/t.scale - log(t.scale)

function inverse(t::ShiftedExp{D}, x::Real) where D
    @unpack shift, scale = t
    if D
        @argcheck x > shift DomainError
        scale*log(x - shift)
    else
        @argcheck x < shift DomainError
        scale*log(shift - x)
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

transform(t::ScaledShiftedLogistic, x::Real) = logistic(x) * t.scale + t.shift

# NOTE: would prefer fma(logistic(x), t.scale, t.shift) for all types, but cf
# https://github.com/JuliaDiff/DiffRules.jl/issues/28
transform(t::ScaledShiftedLogistic, x::AbstractFloat) = fma(logistic(x), t.scale, t.shift)

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
the appropriate transformation is selected. See [`∞`](@ref). If `left` or `right`
are infinite, optionally the scale of the variable can be provied, e.g.:
```
as(Real, left, ∞; scale=10)
````

Some common transformations are predefined as constants, see [`asℝ`](@ref),
[`asℝ₋`](@ref), [`asℝ₊`](@ref), [`as𝕀`](@ref).

!!! note
    The finite arguments are promoted to a common type and affect promotion. E.g.
    `transform(as(0, ∞; scale=10f0), 0f0) isa Float32`, but
    `transform(as(0.0, ∞), 0f0) isa Float64`.
"""
as(::Type{Real}, left, right) =
    throw(ArgumentError("($(left), $(right)) must be an interval"))

as(::Type{Real}, ::Infinity{false}, ::Infinity{true}) = Identity()

function as(::Type{Real}, left::T, ::Infinity{true}; scale=1) where T <: Real
    ShiftedExp(true, left, scale)
end

function as(::Type{Real}, ::Infinity{false}, right::T; scale=1) where T <: Real
    ShiftedExp(false, right, scale)
end

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

Base.show(io::IO, t::ShiftedExp) =
    if t === asℝ₊
        print(io, "asℝ₊")
    elseif t === asℝ₋
        print(io, "asℝ₋")
    elseif t isa ShiftedExp{true}
        print(io, "as(Real, ", t.shift, ", ∞; scale = ", t.scale, ")")
    else
        print(io, "as(Real, -∞, ", t.shift, "; scale = ", t.scale, ")")
    end

Base.show(io::IO, t::ScaledShiftedLogistic) =
    if t === as𝕀
        print(io, "as𝕀")
    else
        print(io, "as(Real, ", t.shift, ", ", t.shift + t.scale, ")")
    end

Base.show(io::IO, t::Identity) = print(io, "asℝ")
