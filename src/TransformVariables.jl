module TransformVariables

using ArgCheck: @argcheck
import ForwardDiff
using LinearAlgebra: UpperTriangular, logabsdet
using DocStringExtensions: SIGNATURES, TYPEDEF
using Parameters: @unpack

export
    dimension, transform, transform_and_logjac, transform_logdensity, inverse,
    logjac_forwarddiff, value_and_logjac_forwarddiff, CustomTransform


# utilities

"""
$(SIGNATURES)

Test if the argument vector has 1-based indexing.
"""
@inline isoneindexed(v::AbstractVector) = axes(v, 1) isa Base.OneTo

"""
$(SIGNATURES)

If the argument vector does not have 1-based indexing, convert it to `Vector`,
otherwise return as is.
"""
@inline ensureoneindexed(v::AbstractVector) =
    isoneindexed(v) ? v : convert(Vector, v)


# log absolute Jacobian determinant

"""
$(TYPEDEF)

Flag used internally by the implementation of transformations, as explained below.

When calculating the log jacobian determinant for a matrix, initialize with
```julia
logjac_zero(flag, x)
```
and then accumulate with log jacobians as needed with `+`.

When `flag` is `LogJac`, methods should return the log Jacobian as the second
argument, otherwise `NoLogJac`, which simply combines to itself with `+`,
serving as an empty placeholder. This allows methods to share code of the two
implementations.
"""
abstract type LogJacFlag end

"""
Calculate log Jacobian as the second value.
"""
struct LogJac <: LogJacFlag end

const LOGJAC = LogJac()

"""
Don't calculate log Jacobian, return `NOLOGJAC` as the second value.
"""
struct NoLogJac <: LogJacFlag end

const NOLOGJAC = NoLogJac()

Base.:+(::NoLogJac, ::NoLogJac) = NOLOGJAC

"""
$(SIGNATURES)

Initial value for log Jacobian calculations.
"""
logjac_zero(::LogJac, x) = zero(x)

logjac_zero(::NoLogJac, _) = NOLOGJAC


# general

const RealVector{T <: Real} = AbstractVector{T}

"""
$(TYPEDEF)

A transformation of real numbers.

# Interface

Implements [`dimension`](@ref) (part of the user interface) and
[`transform_with`] (internal).

The latter is used by wrapper functions [`transform`](@ref) and
[`transform_and_logjac`](@ref) which are part of the user interface.
"""
abstract type TransformReals end

"""
    transform_with(flag::LogJacFlag, t::TransformReals, x::RealVector)

Transform elements of `x`, starting using `transformation`.

The first value returned is the transformed value, the second the log Jacobian
determinant or a placeholder, depending on `flag`.

Types should implement this method. It should assume

1. that `length(x) ≥ dimension(transformation)`, this is checked by the wrapper.

2. generalized indexing, ie start with `first(x)` or `x[firstindex(x)]` and
increment the index as necessary as it traverses `x`.

"""
function transform_with end

@inline function _transform(flag::LogJacFlag, t::TransformReals, x::RealVector)
    @argcheck dimension(t) == length(x)
    transform_with(flag, t, ensureoneindexed(x))
end

"""
$(SIGNATURES)

Transform `x` using `t`.
"""
transform(t::TransformReals, x::RealVector) = first(_transform(NOLOGJAC, t, x))

"""
$(SIGNATURES)

Transform `x` using `t`; calculating the log Jacobian determinant, returned as
the second value.
"""
transform_and_logjac(t::TransformReals, x::RealVector) = _transform(LOGJAC, t, x)

"""
    inverse(t::TransformReals, y)

Return `x` so that `transform(t, x) ≈ y`.
"""
function inverse end

"""
$(SIGNATURES)

Let ``y = t(x)``, and ``f(y)`` a log density at `y`. This function evaluates `f
∘ t` as a log density, taking care of the log Jacobian correction.
"""
function transform_logdensity(t::TransformReals, f, x)
    y, ℓ = transform_and_logjac(t, x)
    ℓ + f(y)
end

"""
    dimension(t::TransformReals)

The dimension (number of elements) that `t` transforms.

Types should implement this method.
"""
function dimension end

include("utilities.jl")
include("scalar.jl")
include("special_arrays.jl")
include("aggregation.jl")
include("custom.jl")

end # module
