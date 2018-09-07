module TransformVariables

using ArgCheck: @argcheck
import ForwardDiff
using LinearAlgebra: UpperTriangular, logabsdet
using DocStringExtensions: SIGNATURES, TYPEDEF
using Parameters: @unpack

export dimension, transform, transform_and_logjac, transform_logdensity,
    inverse, as


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

"""
$(TYPEDEF)

Supertype for all transformations in this package.

# Interface

The user interface consists of

- [`dimension`](@ref)
- [`transform`](@ref)
- [`transform_and_logjac`](@ref)
- [`inverse`]@(ref), [`inverse!`](@ref)
- [`inverse_eltype!`].
"""
abstract type AbstractTransform end

"""
    transform_with(flag::LogJacFlag, t::AbstractTransform, x::RealVector)

Transform elements of `x`, starting using `transformation`.

The first value returned is the transformed value, the second the log Jacobian
determinant or a placeholder, depending on `flag`.

Some types implement [`transform`] and [`transform_and_logjac`] via this method.

`length(x) ≥ dimension(transformation)`, this is checked by the wrapper.

2. generalized indexing, ie start with `first(x)` or `x[firstindex(x)]` and
increment the index as necessary as it traverses `x`.

"""
function transform_with end

"""
    inverse(t::AbstractTransform, y)

Return `x` so that `transform(t, x) ≈ y`.
"""
function inverse end

"""
    inverse_eltype(t::AbstractTransform, y)

The element type for vector `x` so that `inverse!(x, t, y)` works.
"""
function inverse_eltype end

"""
    inverse!(x, t::AbstractTransform, y)

Put `inverse(t, y)` into a preallocated vector `x`, returning `x`.
"""
function inverse! end

"""
$(SIGNATURES)

Let ``y = t(x)``, and ``f(y)`` a log density at `y`. This function evaluates `f
∘ t` as a log density, taking care of the log Jacobian correction.
"""
function transform_logdensity(t::AbstractTransform, f, x)
    y, ℓ = transform_and_logjac(t, x)
    ℓ + f(y)
end

"""
    dimension(t::AbstractTransform)

The dimension (number of elements) that `t` transforms.

Types should implement this method.
"""
function dimension end

"""
    as(T, args...)

Shorthand for constructing transformations with image in `T`. `args` determines
or modifies behavior, details depend on `T`.
"""
function as end


# vector transformations

"""
An `AbstractVector` of `<:Real` elements.
"""
const RealVector{T <: Real} = AbstractVector{T}

"""
$(TYPEDEF)

Transformation that transforms `<: RealVector`s to other values.

# Implementation

Implements `transform` and `transform_logjac` via `transform_with`, `inverse`
via `inverse!`.
"""
abstract type VectorTransform <: AbstractTransform end

"""
$(SIGNATURES)

Transform `x` using `t`.
"""
transform(t::VectorTransform, x::RealVector) = first(transform_with(NOLOGJAC, t, x))

"""
$(SIGNATURES)

Transform `x` using `t`; calculating the log Jacobian determinant, returned as
the second value.
"""
transform_and_logjac(t::VectorTransform, x::RealVector) = transform_with(LOGJAC, t, x)

inverse(t::VectorTransform, y) =
    inverse!(Vector{inverse_eltype(t, y)}(undef, dimension(t)), t, y)

include("utilities.jl")
include("scalar.jl")
include("special_arrays.jl")
include("aggregation.jl")
# include("custom.jl")

end # module
