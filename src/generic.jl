export dimension, transform, transform_and_logjac, transform_logdensity, inverse, inverse!,
    inverse_eltype, as, domain_label

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
`$(FUNCTIONNAME)(flag::LogJacFlag, transformation, x::AbstractVector, index)`

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
`$(FUNCTIONNAME)(x, index, transformation, y)`

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
`$(FUNCTIONNAME)(t, x)`

Transform `x` using `t`.

`$(FUNCTIONNAME)(t)`.

Return a callable equivalent to `x -> transform(t, x)` that transforms its argument:

```julia
transform(t, x) == transform(t)(x)
```
"""
function transform end

"""
$(TYPEDEF)

Partial application of `transform(t, x)`, callable with `x`. Use `transform(t)` to
construct.
"""
struct CallableTransform{T}
    t::T
end

(f::CallableTransform)(x) = transform(f.t, x)

function Base.show(io::IO, f::CallableTransform)
    print(io, "callable for the transformation $(f.t)")
end

transform(t) = CallableTransform(t)

ChangesOfVariables.with_logabsdet_jacobian(f::CallableTransform, x) = transform_and_logjac(f.t, x)

"""
$(TYPEDEF)

Partial application of `inverse(t, y)`, callable with `y`. Use `inverse(t)` to
construct.
"""
struct CallableInverse{T}
    t::T
end

function Base.show(io::IO, f::CallableInverse)
    print(io, "callable inverse for the transformation $(f.t)")
end

inverse(f::CallableInverse) = CallableTransform(f.t)
InverseFunctions.inverse(f::CallableInverse) = CallableTransform(f.t)

inverse(f::CallableTransform) = CallableInverse(f.t)
InverseFunctions.inverse(f::CallableTransform) = CallableInverse(f.t)

ChangesOfVariables.with_logabsdet_jacobian(f::CallableInverse, x) = inverse_and_logjac(f.t, x)

"""
`$(FUNCTIONNAME)(t, y)`

Return `x` so that `transform(t, x) ≈ y`.

`$(FUNCTIONNAME)(t)`

Return a callable equivalent to `y -> inverse(t, y)`. `t` can also be a callable created
with transform, so the following holds:
```julia
inverse(t)(y) == inverse(t, y) == inverse(transform(t))(y)
```
"""
inverse(t::AbstractTransform) = CallableInverse(t)

(f::CallableInverse)(y) = inverse(f.t, y)

"""
`$(FUNCTIONNAME)(t::AbstractTransform, y)`

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
function transform_logdensity(t::AbstractTransform, f::F, x) where F
    y, ℓ = transform_and_logjac(t, x)
    ℓ + f(y)
end

"""
`$(FUNCTIONNAME)(t::AbstractTransform)`

The dimension (number of elements) that `t` transforms.

Types should implement this method.
"""
function dimension end

"""
`$(FUNCTIONNAME)(T, args...)`

Shorthand for constructing transformations with image in `T`. `args` determines
or modifies behavior, details depend on `T`.

Not all transformations have an `as` method, some just have direct constructors.
See `methods(as)` for a list.

# Examples

```julia
as(Real, -∞, 1)          # transform a real number to (-∞, 1)
as(Array, 10, 2)         # reshape 20 real numbers to a 10x2 matrix
as(Array, as𝕀, 10)       # transform 10 real numbers to (0, 1)
as((a = asℝ₊, b = as𝕀)) # transform 2 real numbers a NamedTuple, with a > 0, 0 < b < 1
as(SArray{1,2,3}, as𝕀)  # transform to a static array of positive numbers
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

Result size for various transformations. For simplifying the internal API, not exported.
Return type can be anything, as long as it is consistent.
"""
function result_size end

####
#### printing
####

"""
$(SIGNATURES)

Shift displayed range of indices by `o`.
"""
_offset(indices::UnitRange, o) = (first(indices) + o):(last(indices) + o)

_offset(indices::Nothing, o) = nothing

"""
$(SIGNATURES)

Utility function to make a single row for [`_summary_rows`](@ref), for transformations
that just need one.
"""
function _summary_row(transformation, repr)
    [NamedTuple{(:level,:indices,:repr),Tuple{Int,Union{Nothing,UnitRange{Int}},Any}}((1, 1:dimension(transformation), repr))]
end

"""
$(SIGNATURES)

Return a vector of rows, each consisting of a `NamedTuple` with the following fields:

- `level::Int`, nesting level of that row, starting from `1`,
- `indices::UnitRange{Int}`, the indices it applies to. `nothing` is used when this is not
  applicable.
- `repr`, representation relevant for `mime`, usually a string.

Not exported, used to generate displayed output.

!!! note
    Transformations should define either this method, or `Base.show` when they have a really
    short name that has trivial indexing (eg scalar transformations).
"""
function _summary_rows(transformation::AbstractTransform, mime)
    _summary_row(transformation, repr(transformation))
end

function Base.show(io::IO,  mime::MIME"text/plain", transformation::AbstractTransform)
    rows = _summary_rows(transformation, mime)
    if length(rows) == 1
        print(io, only(rows).repr, " (dimension $(dimension(transformation)))")
    else
        for (i, row) in enumerate(rows)
            @unpack level, indices, repr = row
            i > 1 && println(io)
            print(io, ' '^(2 * (level - 1)))
            indices ≢ nothing && print(io, '[', indices, "] ")
            print(io, repr)
        end
    end
end

####
#### labels
####

"""
$(SIGNATURES)

Return a string that can be used to for identifying a coordinate. Mainly for debugging and
generating graphs and data summaries.

Transformations may provide a heuristic label.

Transformations should implement [`_domain_label`](@ref).

# Example

```jldoctest
julia> t = as((a = asℝ₊,
            b = as(Array, asℝ₋, 1, 1),
            c = corr_cholesky_factor(2)));

julia> [domain_label(t, i) for i in 1:dimension(t)]
3-element Vector{String}:
 ".a"
 ".b[1,1]"
 ".c[1]"
```
"""
function domain_label(transformation, index::Integer)
    @argcheck 1 ≤ index ≤ dimension(transformation)
    io = IOBuffer()
    for e in _domain_label(transformation, Int(index))
        if e isa Symbol
            print(io, '.', e)
        elseif e isa Tuple{Vararg{Int}}
            print(io, '[')
            join(io, e, ',')
            print(io, ']')
        else
            error("Internal error: invalid label $e")
        end
    end
    String(take!(io))
end

"""
$(SIGNATURES)

Internal function for implementing [`domain label`](@ref). Skips bounds checking, returns a
tuple interpreted as follows:

- tuples of integers are array indices
- symbols are keys

Transformations should implement this function.

# Implementation note

Returning the semantic parts allows future extensions, eg have `domain_label` format to
other MIME types.
"""
function _domain_label(transformation, index::Int)
    ((index,), )                   # fall back to a flat index
end
