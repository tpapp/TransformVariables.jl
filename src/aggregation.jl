export to_array, to_tuple


# arrays

struct TransformationArray{T <: TransformReals,M} <: TransformReals
    transformation::T
    dims::NTuple{M, Int}
end

dimension(t::TransformationArray) = dimension(t.transformation) * prod(t.dims)

"""
$(SIGNATURES)

Return a transformation that applies `transformation` repeatedly to create an
array with the given `dims`.
"""
to_array(transformation::TransformReals, dims::Tuple{Vararg{Int}}) =
    TransformationArray(transformation, dims)

to_array(transformation::TransformReals, dims::Int...) =
    to_array(transformation, dims)

function transform_with(flag::LogJacFlag, t::TransformationArray, x::RealVector)
    @unpack transformation, dims = t
    d = dimension(transformation)
    yℓ = reshape([transform_with(flag, transformation, viewafter(x, i, d))
                  for i in 0:d:(d*(prod(dims)-1))],
                 dims)
    first.(yℓ), sum(last, yℓ)
end

function inverse(transformation_array::TransformationArray, y::Array)
    @unpack transformation, dims = transformation_array
    @argcheck size(y) == dims
    mapreduce(y -> inverse(transformation, y), vcat, vec(y))
end


# tuples

const NTransformations{N} = Tuple{Vararg{TransformReals,N}}

struct TransformationTuple{K, T <: NTransformations{K}} <: TransformReals
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
to_tuple(transformations::NTransformations) = TransformationTuple(transformations)

"""
$(SIGNATURES)
"""
to_tuple(transformations::TransformReals...) = to_tuple(transformations)

@inline function _transform_tuple(flag::LogJacFlag, tt::NTransformations,
                                  x::RealVector)
    index = 0
    yℓ = map(t -> begin
             d = dimension(t)
             result = transform_with(flag, t, viewafter(x, index, d))
             index += d
             result
             end, tt)
    first.(yℓ), sum(last, yℓ)
end

@inline _inverse_tuple(tt::NTransformations{K}, y::NTuple{K,Any}) where K =
    mapreduce(ty -> inverse(ty...), vcat, zip(tt, y))

transform_with(flag::LogJacFlag, tt::TransformationTuple, x::RealVector) =
    _transform_tuple(flag, tt.transformations, x)

inverse(tt::TransformationTuple{K}, y::NTuple{K,Any}) where K =
    _inverse_tuple(tt.transformations, y)

struct TransformationNamedTuple{names, T <: NTransformations} <: TransformReals
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
to_tuple(transformations::NamedTuple{T,<:NTransformations}) where T =
    TransformationNamedTuple(transformations)

"""
$(SIGNATURES)
"""
to_tuple(; transformations::NTransformations...) =
    to_tuple(collect(transformations))

dimension(tn::TransformationNamedTuple) = tn.dimension

function transform_with(flag::LogJacFlag, tt::TransformationNamedTuple{names},
                      x::RealVector) where {names}
    y, ℓ = _transform_tuple(flag, tt.transformations, x)
    NamedTuple{names}(y), ℓ
end

inverse(tn::TransformationNamedTuple{names}, y::NamedTuple{names}) where names =
    _inverse_tuple(tn.transformations, values(y))
