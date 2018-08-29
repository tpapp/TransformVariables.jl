export ∞, to_interval, to_ℝ, to_ℝ₊, to_ℝ₋, to_𝕀

abstract type TransformScalar <: TransformReals end

dimension(::TransformScalar) = 1

transform_at(t::TransformScalar, flag::NoLogJac, x::RealVector, index::Int) =
    transform_scalar(t, x[index]), flag

function transform_at(t::TransformScalar, ::LogJac, x::RealVector, index::Int)
    xi = x[index]
    transform_scalar(t, xi), logjac_scalar(t, xi)
end

inverse(t::TransformScalar, y::Real) = [inverse_scalar(t, y)]


# identity

struct Identity <: TransformScalar end

transform_scalar(::Identity, x::Real) = x

inverse_scalar(::Identity, x::Real) = x

logjac_scalar(::Identity, x) = zero(x)


# shifted exponential

struct ShiftedExp{D, T <: Real} <: TransformScalar
    shift::T
    function ShiftedExp{D,T}(shift::T) where {D, T <: Real}
        @argcheck D isa Bool
        new(shift)
    end
end

ShiftedExp(ispositive::Bool, shift::T) where {T <: Real} =
    ShiftedExp{ispositive,T}(shift)

transform_scalar(t::ShiftedExp{D}, x::Real) where D =
    D ? t.shift + exp(x) : t.shift - exp(x)

logjac_scalar(t::ShiftedExp, x::Real) = x

function inverse_scalar(t::ShiftedExp{D}, x::Real) where D
    @unpack shift = t
    if D
        @argcheck x > shift DomainError
        log(x - shift)
    else
        @argcheck x < shift DomainError
        log(shift - x)
    end
end


# scaled and shifted logistic

struct ScaledShiftedLogistic{T <: Real} <: TransformScalar
    scale::T
    shift::T
    function ScaledShiftedLogistic{T}(scale::T, shift::T) where {T <: Real}
        @argcheck scale > 0 ArgumentError
        new(scale, shift)
    end
end

ScaledShiftedLogistic(scale::T, shift::T) where {T <: Real} =
    ScaledShiftedLogistic{T}(scale, shift)

ScaledShiftedLogistic(scale::Real, shift::Real) =
    ScaledShiftedLogistic(promote(scale, shift)...)

transform_scalar(t::ScaledShiftedLogistic, x::Real) =
    fma(logistic(x), t.scale, t.shift)

logjac_scalar(t::ScaledShiftedLogistic, x) = log(t.scale) + logistic_logjac(x)

inverse_scalar(t::ScaledShiftedLogistic, x) =
    logit(fma(x, inv(t.scale), - t.shift/ t.scale))


# to_interval interface

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
    $(SIGNATURES)

Return a transformation that transforms a single real number to the given (open)
interval.

`left < right` is required, but may be `-∞` or `∞`, respectively, in which case
the appropriate transformation is selected. See [`∞`](@ref).

Some common transformations are predefined as constants, see [`to_ℝ`](@ref),
[`to_ℝ₋`](@ref), [`to_ℝ₊`](@ref), [`to_𝕀`](@ref).
"""
to_interval(left, right) =
    throw(ArgumentError("($(left), $(right)) must be an interval"))

to_interval(::Infinity{false}, ::Infinity{true}) = Identity()

to_interval(left::Real, ::Infinity{true}) = ShiftedExp(true, left)

to_interval(::Infinity{false}, right::Real) = ShiftedExp(false, right)

function to_interval(left::Real, right::Real)
    @argcheck left < right "the interval ($(left), $(right)) is empty"
    ScaledShiftedLogistic(right - left, left)
end

"Transform to a non-negative real number."
const to_ℝ₊ = to_interval(0.0, ∞)

"Transform to a non-positive real number."
const to_ℝ₋ = to_interval(-∞, 0.0)

"Transform to the unit interval `(0, 1)`."
const to_𝕀 = to_interval(0.0, 1.0)

"Transform to the real line (identity)."
const to_ℝ = to_interval(-∞, ∞)
