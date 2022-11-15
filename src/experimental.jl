"""
This is an in-progress rewrite of the package that will be released, primarily for testing
it in production.

For the purposes of SemVer, this is **not** part of the API. Changes will be reflected as
patch releases until it is merged.
"""
module Experimental

using ArgCheck: @argcheck
using StaticArrays: SArray
using TransformVariables: LogJacFlag, LogJac, NoLogJac,
    LOGJAC, NOLOGJAC, logjac_zero, AbstractTransform
import TransformVariables: as, dimension, transform, transform_with

####
#### generic
####

# NOTE: it is fortunate that we mispelled the abstract type name previously ;-)
abstract type AbstractTransformation <: AbstractTransform end

"""
_transform(flag::LogJacFlag, transformation::AbstractTransformation, x::AbstractVector)

Transform elements of `x` using `transformation`.

Return `(y, logjac)′`, where

- `y` is the result of the transformation,

- `logjac` is the the log Jacobian determinant or a placeholder, depending on `flag`,

**Internal function**. Methods can assume that `length(x) == dimension(transformation)`.
"""
function _transform end

# hook into existing API
function transform_with(flag::LogJacFlag, transformation::AbstractTransformation,
                        x::AbstractVector, index::Int)
    d = dimension(transformation)
    y, lj = _transform(flag, transformation, view(x, index:(index+d-1)))
    y, lj, index + d
end

function transform(transformation::AbstractTransformation, x::AbstractVector)
    first(_transform(NOLOGJAC, transformation, x))
end

function transform_and_logjac(transformation::AbstractTransformation, x::AbstractVector)
    _transform(LOGJAC, transformation, x)
end

####
#### transformations
####

# struct asArrayView{N,S<:NTuple{N,Int}}
#     dimension::S
# end

# function as_array_view(dimension::NTuple{N,Int}) where N
#     @argcheck all(x -> x ≥ 1, dimension)
#     asArrayView{N,typeof(dimension)}(dimension)
# end

struct asStaticArray{D,S} <: AbstractTransformation end

function as(::Type{<:SArray{S}}) where S
    dim = fieldtypes(S)
    @argcheck all(x -> x ≥ 1, dim)
    asStaticArray{prod(dim),S}()
end

dimension(transformation::asStaticArray{D}) where D = D

function _transform(flag::LogJacFlag, transformation::asStaticArray{D,S},
                    x::AbstractVector{T}) where {D,S,T}
    SArray{S}(x), logjac_zero(flag, T)
end

end
