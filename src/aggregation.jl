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

function transform_at(t::TransformationArray, x::RealVector, index::Int)
    @unpack transformation, dims = t
    reshape([transform_at(transformation, x, i) for i in
             StepRangeLen(index, dimension(transformation), prod(dims))],
            dims)
end

function inverse(transformation_array::TransformationArray, y::Array)
    @unpack transformation, dims = transformation_array
    @argcheck size(y) == dims
    mapreduce(y -> inverse(transformation, y), vcat, vec(y))
end


# tuples

struct TransformationTuple{T <: Tuple{Vararg{TransformReals}}, N} <: TransformReals
    transformations::T
    function TransformationTuple(transformations::T) where {T <: Tuple{Vararg{TransformReals}}}
        N = sum(dimension, transformations)
        new{T, N}(transformations)
    end
end

dimension(::TransformationTuple{T, N}) where {T, N} = N

to_tuple(transformations::Tuple{Vararg{TransformReals}}) =
    TransformationTuple(transformations)

to_tuple(transformations::TransformReals...) = to_tuple(transformations)

function transform_at(transformation_tuple::TransformationTuple,
                      x::RealVector, index::Int)
    @unpack transformations = transformation_tuple
    map(t -> begin
        result = transform_at(t, x, index)
        index += dimension(t)
        result
        end, transformations)
end

function inverse(t::TransformationTuple{<:Tuple{Vararg{<:TransformReals,K}},K},
                 y::NTuple{K}) where K
    mapreduce(ty -> inverse(ty...), vcat, zip(t.transformations, y))
end
