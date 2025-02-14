module UnitfulExt

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using LogExpFunctions: logistic, logit
import Unitful: Quantity, Units, ustrip, NoUnits
import TransformVariables: ScalarTransform, transform, transform_and_logjac, inverse, as, Infinity, asℝ₊


####
#### shifted exponential with units
####
"""
$(TYPEDEF)

Shifted exponential with units. When `D::Bool == true`, maps to `(shift, ∞)` using `x ↦
shift + eˣ`, otherwise to `(-∞, shift)` using `x ↦ shift - eˣ`.
"""
struct DimShiftedExp{D, T <: Quantity} <: ScalarTransform
    shift::T
    function DimShiftedExp{D,T}(shift::T) where {D, T <: Quantity}
        @argcheck D isa Bool
        new(shift)
    end
end

DimShiftedExp(D::Bool, shift::T) where {T <: Quantity} = DimShiftedExp{D,T}(shift)

transform(t::DimShiftedExp{D, Quantity{V,DD,U}}, x::Real) where {D, V, DD, U} =
    D ? t.shift + exp(x)*U() : t.shift - exp(x)*U()

# NOTE: not sure if or how this should be defined, since units.
# transform_and_logjac(t::DimShiftedExp, x::Real) = transform(t, x), x

function inverse(t::DimShiftedExp{D, Quantity{V,DD,U}}, x::Quantity) where {D, V, DD, U}
    (; shift) = t
    if D
        @argcheck x > shift DomainError
        log(ustrip(U(), x - shift))
    else
        @argcheck x < shift DomainError
        log(ustrip(U(), shift - x))
    end
end

###
#### scaled-shifted logistic with units
####

"""
$(TYPEDEF)

Maps to `(scale, shift + scale)` using `logistic(x) * scale + shift`.
"""
struct DimScaledShiftedLogistic{T <: Quantity} <: ScalarTransform
    scale::T
    shift::T
    function DimScaledShiftedLogistic{T}(scale::T, shift::T) where {T <: Quantity}
        @argcheck scale > zero(typeof(scale))
        new(scale, shift)
    end
end

DimScaledShiftedLogistic(scale::T, shift::T) where {T <: Quantity} =
    DimScaledShiftedLogistic{T}(scale, shift)

function DimScaledShiftedLogistic(scale::T1, shift::T2) where {T1 <: Quantity, T2 <: Quantity}
    DimScaledShiftedLogistic(promote(scale, shift)...)
end

# # Switch to muladd and now it does have a DiffRule defined
transform(t::DimScaledShiftedLogistic, x::Real) = muladd(logistic(x), t.scale, t.shift)

# NOTE: not sure if or how this should be defined, since units.
# transform_and_logjac(t::ScaledShiftedLogistic, x) =
#     transform(t, x), log(t.scale) + logistic_logjac(x)

function inverse(t::DimScaledShiftedLogistic{Quantity{N,D,U}}, y) where {N,D,U}
    @argcheck y > t.shift           DomainError
    @argcheck y < t.scale + t.shift DomainError
    logit(ustrip(NoUnits, (y - t.shift)/t.scale))
end

# NOTE: not sure if or how this should be defined, since units.
# # NOTE: inverse_and_logjac interface experimental and sporadically implemented for now
# function inverse_and_logjac(t::ScaledShiftedLogistic, y)
#     @argcheck y > t.shift           DomainError
#     @argcheck y < t.scale + t.shift DomainError
#     z = (y - t.shift) / t.scale
#     logit(z), logit_logjac(z) - log(t.scale)
# end

function as(::Type{Real}, left::Quantity, right::Quantity)
    @argcheck left < right "the interval ($(left), $(right)) is empty"
    DimScaledShiftedLogistic(right - left, left)
end

as(::Type{Real}, left::Quantity, ::Infinity{true}) = DimShiftedExp(true, left)

as(::Type{Real}, ::Infinity{false}, right::Quantity) = DimShiftedExp(false, right)

Base.:(*)(a::typeof(asℝ₊), u::Units) = as(Real, 0.0*u, Infinity{true}())
Base.:(*)(a::typeof(asℝ₋), u::Units) = as(Real, Infinity{false}(), 0.0*u)

end # module