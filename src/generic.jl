export dimension, transform, transform_and_logjac, transform_logdensity, inverse, inverse!,
    inverse_eltype, as, random_arg, random_value

###
### log absolute Jacobian determinant
###

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

###
### internal methods that implement transformations
###

"""
    transform_with(flag::LogJacFlag, transformation, x::AbstractVector, index)

Transform elements of `x` from `index`, using `transformation`.

Return `(y, logjac), index′`, where

- `y` is the result of the transformation,

- `logjac` is the the log Jacobian determinant or a placeholder, depending on `flag`,

- `index′` is the next index in `x` after the elements used for the transformation

**Internal function**. Implementations

1. can assume that `x` has enough elements for `transformation` (ie `@inbounds` can be
used),

2. should work with generalized indexing on `x`.
"""
function transform_with end

"""
    inverse_at!(x, index, transformation, y)

Invert transformation at `y` and put the result in `x` starting at `index`.

**Internal function**. Implementations

1. can assume that `x` has enough elements for the result (ie `@inbounds` can be used),

2. should work with generalized indexing on `x`.
"""
function inverse_at! end

####
#### API
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

Base.broadcastable(t::AbstractTransform) = Ref(t)

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
$(SIGNATURES)

Put `inverse(t, y)` into a preallocated vector `x`, returning `x`.

Generalized indexing should be assumed on `x`.

See [`inverse_eltype`](@ref) for determining the type of `x`.
"""
function inverse!(x::AbstractVector, transformation::AbstractTransform, y)
    @argcheck dimension(transformation) == length(x)
    inverse_at!(x, firstindex(x), transformation, y)
    x
end

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
$(TYPEDEF)

Transformation that transforms `<: AbstractVector`s to other values.

# Implementation

Implements [`transform`](@ref) and [`transform_and_logjac`](@ref) via
[`transform_with`](@ref), and [`inverse`](@ref) via [`inverse!`](@ref).
"""
abstract type VectorTransform <: AbstractTransform end

"""
$(SIGNATURES)

Transform `x` using `t`.
"""
function transform(t::VectorTransform, x::AbstractVector)
    @argcheck dimension(t) == length(x)
    first(transform_with(NOLOGJAC, t, x, firstindex(x)))
end

"""
$(SIGNATURES)

Transform `x` using `t`; calculating the log Jacobian determinant, returned as
the second value.
"""
function transform_and_logjac(t::VectorTransform, x::AbstractVector)
    @argcheck dimension(t) == length(x)
    y, ℓ, _ = transform_with(LOGJAC, t, x, firstindex(x))
    y, ℓ
end

function inverse(t::VectorTransform, y)
    inverse!(Vector{inverse_eltype(t, y)}(undef, dimension(t)), t, y)
end

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
