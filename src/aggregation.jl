export to_array, to_tuple


# arrays

struct TransformationArray{T <: AbstractTransform,M} <: VectorTransform
    transformation::T
    dims::NTuple{M, Int}
end

dimension(t::TransformationArray) = dimension(t.transformation) * prod(t.dims)

"""
$(SIGNATURES)

Return a transformation that applies `transformation` repeatedly to create an
array with the given `dims`.
"""
as(::Type{Array}, transformation::AbstractTransform, dims::Tuple{Vararg{Int}}) =
    TransformationArray(transformation, dims)

as(::Type{Array}, dims::Tuple{Vararg{Int}}) = as(Array, Identity, dims)

as(::Type{Array}, transformation::AbstractTransform, dims::Int...) =
    TransformationArray(transformation, dims)

as(::Type{Array}, dims::Int...) = as(Array, Identity, dims)

function transform_with(flag::LogJacFlag, t::TransformationArray, x::RealVector)
    @unpack transformation, dims = t
    d = dimension(transformation)
    index = firstindex(x)
    yℓ = reshape([(y = transform_with(flag, transformation, index_into(x, index, d));
                   index += d;
                   y)
                  for _ in Base.OneTo(prod(dims))],
                 dims)
    first.(yℓ), sum(last, yℓ)
end

transform_with(flag, t::TransformationArray{Identity}, x::RealVector) =
    reshape(copy(x), t.dims)

inverse_eltype(t::TransformationArray, x::AbstractArray) =
    inverse_eltype(t.transformation, first(x))

function inverse!(x::RealVector,
                  transformation_array::TransformationArray,
                  y::Array)
    @unpack transformation, dims = transformation_array
    @argcheck size(y) == dims
    index = firstindex(x)
    d = dimension(transformation)
    for elt in vec(y)
        inverse!(index_into(x, index, d), transformation, elt)
        index += d;
    end
    x
end


# tuples

const NTransformations{N} = Tuple{Vararg{AbstractTransform,N}}

struct TransformationTuple{K, T <: NTransformations{K}} <: VectorTransform
    transformations::T
    dimension::Int
    function TransformationTuple(transformations::T) where {K, T <: NTransformations{K}}
        new{K,T}(transformations, sum(dimension, transformations))
    end
end

dimension(tt::TransformationTuple) = tt.dimension

"""
$(SIGNATURES)

Return a transformation that transforms consecutive groups of real numbers to a
(named) tuple, using the given transformations.
"""
as(transformations::NTransformations) = TransformationTuple(transformations)

"""
$(SIGNATURES)

Helper function for transforming tuples. Used internally.
"""
function _transform_tuple(flag::LogJacFlag, tt::NTransformations, x::RealVector)
    index = firstindex(x)
    yℓ = map(t -> begin
             d = dimension(t)
             result = transform_with(flag, t, index_into(x, index, d))
             index += d
             result
             end, tt)
    first.(yℓ), sum(last, yℓ)
end

"""
$(SIGNATURES)

Helper function determining element type of inverses from tuples. Used
internally.
"""
_inverse_eltype_tuple(ts::NTransformations{K}, ys::NTuple{K,Any}) where K =
    mapreduce(((t, y),) -> inverse_eltype(t, y), promote_type, zip(ts, ys))

"""
$(SIGNATURES)

Helper function for inverting tuples of transformations. Used internally.
"""
function _inverse!_tuple(x::RealVector, ts::NTransformations{K},
                         ys::NTuple{K,Any}) where K
    index = firstindex(x)
    for (t, y) in zip(ts, ys)
        d = dimension(t)
        inverse!(index_into(x, index, d), t, y)
        index += d
    end
    x
end

transform_with(flag::LogJacFlag, tt::TransformationTuple, x::RealVector) =
    _transform_tuple(flag, tt.transformations, x)

inverse_eltype(tt::TransformationTuple{K}, y::NTuple{K,Any}) where K =
    _inverse_eltype_tuple(tt.transformations, y)

function inverse!(x::RealVector, tt::TransformationTuple{K},
                  y::NTuple{K,Any}) where K
    @argcheck length(x) == dimension(tt)
    _inverse!_tuple(x, tt.transformations, y)
end

struct TransformationNamedTuple{names, T <: NTransformations} <: VectorTransform
    transformations::T
    dimension::Int
    function TransformationNamedTuple(transformations::NamedTuple{names,T}) where
        {names, T <: NTransformations}
        new{names,T}(values(transformations), sum(dimension, transformations))
    end
end

"""
$(SIGNATURES)
"""
as(transformations::NamedTuple{T,<:NTransformations}) where T =
    TransformationNamedTuple(transformations)

dimension(tn::TransformationNamedTuple) = tn.dimension

function transform_with(flag::LogJacFlag, tt::TransformationNamedTuple{names},
                      x::RealVector) where {names}
    y, ℓ = _transform_tuple(flag, tt.transformations, x)
    NamedTuple{names}(y), ℓ
end

inverse_eltype(tt::TransformationNamedTuple{names},
               y::NamedTuple{names}) where names =
    _inverse_eltype_tuple(tt.transformations, values(y))

function inverse!(x::RealVector, tt::TransformationNamedTuple{names},
                  y::NamedTuple{names}) where names
    @argcheck length(x) == dimension(tt)
    _inverse!_tuple(x, tt.transformations, values(y))
end
