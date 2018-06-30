export to_array, to_tuple


# arrays

struct TransformationArray{T <: TransformReals,M} <: TransformReals
    transformation::T
    dims::NTuple{M, Int}
end

length(t::TransformationArray) = length(t.transformation) * prod(t.dims)

to_array(transformation::TransformReals, dims::Tuple{Vararg{Int}}) =
    TransformationArray(transformation, dims)

to_array(transformation::TransformReals, dims::Int...) =
    to_array(transformation, dims)

function transform_at(t::TransformationArray, flag::LogJacFlag, x::RealVector, index::Int)
    @unpack transformation, dims = t
    yℓ = reshape([transform_at(transformation, flag, x, i) for i in
                  StepRangeLen(index, length(transformation), prod(dims))],
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
    length::Int
    function TransformationTuple(transformations::T) where {K, T <: NTransformations{K}}
        new{K,T}(transformations, sum(length, transformations))
    end
end

length(tt::TransformationTuple) = tt.length

to_tuple(transformations::NTransformations) = TransformationTuple(transformations)

to_tuple(transformations::TransformReals...) = to_tuple(transformations)

@inline function _transform_tuple(tt::NTransformations, flag::LogJacFlag,
                                  x::RealVector, index::Int)
    yℓ = map(t -> begin
             result = transform_at(t, flag, x, index)
             index += length(t)
             result
             end, tt)
    first.(yℓ), sum(last, yℓ)
end

@inline _inverse_tuple(tt::NTransformations{K}, y::NTuple{K,Any}) where K =
    mapreduce(ty -> inverse(ty...), vcat, zip(tt, y))

transform_at(tt::TransformationTuple, flag::LogJacFlag, x::RealVector,
             index::Int) = _transform_tuple(tt.transformations, flag, x, index)

inverse(tt::TransformationTuple{K}, y::NTuple{K,Any}) where K =
    _inverse_tuple(tt.transformations, y)

if VERSION ≥ v"0.7-"
    struct TransformationNamedTuple{names, T <: NTransformations} <: TransformReals
        transformations::T
        length::Int
        function TransformationNamedTuple(transformations::NamedTuple{names,T}) where
            {names, T <: NTransformations}
            new{names,T}(values(transformations), sum(length, transformations))
        end
    end

    to_tuple(transformations::NamedTuple{_,<:NTransformations}) where _ =
        TransformationNamedTuple(transformations)

    to_tuple(; transformations::NTransformations...) =
        to_tuple(collect(transformations))

    length(tn::TransformationNamedTuple) = tn.length

    function transform_at(tt::TransformationNamedTuple{names},
                          flag::LogJacFlag, x::RealVector,
                          index::Int) where {names}
        y, ℓ = _transform_tuple(tt.transformations, flag, x, index)
        NamedTuple{names}(y), ℓ
    end

    inverse(tn::TransformationNamedTuple{names}, y::NamedTuple{names}) where names =
        _inverse_tuple(tn.transformations, values(y))
end
