export to_array, to_tuple


# arrays

struct TransformationArray{T <: TransformReals,M} <: TransformReals
    transformation::T
    dims::NTuple{M, Int}
end

dimension(t::TransformationArray) = dimension(t.transformation) * prod(t.dims)

to_array(transformation::TransformReals, dims::Tuple{Vararg{Int}}) =
    TransformationArray(transformation, dims)

to_array(transformation::TransformReals, dims::Int...) =
    to_array(transformation, dims)

function transform_at(t::TransformationArray, flag::LogJacFlag, x::RealVector, index::Int)
    @unpack transformation, dims = t
    yℓ = reshape([transform_at(transformation, flag, x, i) for i in
                  StepRangeLen(index, dimension(transformation), prod(dims))],
                 dims)
    first.(yℓ), sum(last, yℓ)
end

function inverse(transformation_array::TransformationArray, y::Array)
    @unpack transformation, dims = transformation_array
    @argcheck size(y) == dims
    mapreduce(y -> inverse(transformation, y), vcat, vec(y))
end


# tuples

struct TransformationTuple{T <: Tuple{Vararg{TransformReals}}} <: TransformReals
    transformations::T
    total_dimension::Int
    function TransformationTuple(transformations::T) where {T <: Tuple{Vararg{TransformReals}}}
        new{T}(transformations, sum(dimension, transformations))
    end
end

dimension(transformation_tuple::TransformationTuple) =
    transformation_tuple.total_dimension

to_tuple(transformations::Tuple{Vararg{TransformReals}}) =
    TransformationTuple(transformations)

to_tuple(transformations::TransformReals...) = to_tuple(transformations)

function transform_at(transformation_tuple::TransformationTuple, flag::LogJacFlag,
                      x::RealVector, index::Int)
    @unpack transformations = transformation_tuple
    yℓ = map(t -> begin
             result = transform_at(t, flag, x, index)
             index += dimension(t)
             result
             end, transformations)
    first.(yℓ), sum(last, yℓ)
end

function inverse(t::TransformationTuple{<:Tuple{Vararg{<:TransformReals,K}}},
                 y::Tuple{Vararg{<:Any,K}}) where K
    mapreduce(ty -> inverse(ty...), vcat, zip(t.transformations, y))
end
