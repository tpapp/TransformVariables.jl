export dimension, transform, transform_and_logjac, transform_logdensity, inverse, inverse!,
    inverse_eltype, as, random_arg, random_value

####
#### log absolute Jacobian determinant
####

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
logjac_zero(::LogJac, T::Type{<:Real}) = log(one(T))

logjac_zero(::NoLogJac, _) = NOLOGJAC

####
#### general
####

"""
$(TYPEDEF)

Supertype for all transformations in this package.

# Interface

The user interface consists of

- [`dimension`](@ref)
- [`transform`](@ref)
- [`transform_and_logjac`](@ref)
- [`inverse`]@(ref), [`inverse!`](@ref)
- [`inverse_eltype`](@ref).
"""
abstract type AbstractTransform end

"""
    transform_with(flag::LogJacFlag, t::AbstractTransform, x::RealVector)

Transform elements of `x`, starting using `transformation`.

The first value returned is the transformed value, the second the log Jacobian
determinant or a placeholder, depending on `flag`.

In contrast to [`transform`] and [`transform_and_logjac`], this method always
assumes that `x` is a `RealVector`, for efficient traversal. Some types
implement the latter two via this method.

Implementations should assume generalized indexing on `x`.
"""
function transform_with end

"""
$(TYPEDEF)

Inverse of the wrapped transform. Use the 1-argument version of
[`inverse`](@ref) to construct.
"""
struct InverseTransform{T}
    transform::T
end

"""
    inverse(t::AbstractTransform, y)

Return `x` so that `transform(t, x) ≈ y`.

    $SIGNATURES

Return a callable equivalen to `y -> inverse(t, y)`.
"""
inverse(t::AbstractTransform) = InverseTransform(t)

(ι::InverseTransform)(y) = inverse(ι.transform, y)

"""
    inverse_eltype(t::AbstractTransform, y)

The element type for vector `x` so that `inverse!(x, t, y)` works.
"""
function inverse_eltype end

"""
    inverse!(x, t::AbstractTransform, y)

Put `inverse(t, y)` into a preallocated vector `x`, returning `x`.

Generalized indexing should be assumed on `x`.

See [`inverse_eltype`](@ref) for determining the type of `x`.
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

Not all transformations have an `as` method, some just have direct constructors.
See `methods(as)` for a list.

# Examples

```julia
as(Real, -∞, 1)          # transform a real number to (-∞, 1)
as(Array, 10, 2)         # reshape 20 real numbers to a 10x2 matrix
as((a = asℝ₊, b = as𝕀)) # transform 2 real numbers a NamedTuple, with a > 0, 0 < b < 1
```
"""
function as end

####
#### vector transformations
####

"""
An `AbstractVector` of `<:Real` elements.

Used internally as a type for transformations from vectors.
"""
const RealVector{T <: Real} = AbstractVector{T}

"""
$(TYPEDEF)

Transformation that transforms `<: RealVector`s to other values.

# Implementation

Implements [`transform`](@ref) and [`transform_and_logjac`](@ref) via
[`transform_with`](@ref), and [`inverse`](@ref) via [`inverse!`](@ref).
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

"""
$(SIGNATURES)

A random argument for a transformation.

# Keyword arguments

$(_RANDOM_REALS_KWARGS_DOC)
"""
random_arg(x::VectorTransform; kwargs...) = random_reals(dimension(x); kwargs...)

"""
$(SIGNATURES)

Random value from a transformation.

# Keyword arguments

$(_RANDOM_REALS_KWARGS_DOC)
"""
random_value(t::AbstractTransform; kwargs...) = transform(t, random_arg(t; kwargs...))
