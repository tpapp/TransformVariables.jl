export to_array, to_tuple


# arrays

"""
$(TYPEDEF)

Apply `transformation` repeatedly to create an array with given `dims`.
"""
@calltrans struct ArrayTransform{T <: AbstractTransform,M} <: VectorTransform
    transformation::T
    dims::NTuple{M, Int}
end

dimension(t::ArrayTransform) = dimension(t.transformation) * prod(t.dims)

"""
$(SIGNATURES)

Return a transformation that applies `transformation` repeatedly to create an
array with the given `dims`.
"""
as(::Type{Array}, transformation::AbstractTransform, dims::Tuple{Vararg{Int}}) =
    ArrayTransform(transformation, dims)

as(::Type{Array}, dims::Tuple{Vararg{Int}}) = as(Array, Identity(), dims)

as(::Type{Array}, transformation::AbstractTransform, dims::Int...) =
    ArrayTransform(transformation, dims)

as(::Type{Array}, dims::Int...) = as(Array, Identity(), dims)

function as(::Type{Vector}, args...)
    t = as(Array, args...)
    @argcheck length(t.dims) == 1 "Vector should have 1 dimension."
    t
end

function as(::Type{Matrix}, args...)
    t = as(Array, args...)
    @argcheck length(t.dims) == 2 "Matrix should have 2 dimensions."
    t
end

function transform_with(flag::LogJacFlag, t::ArrayTransform, x::RealVector)
    @unpack transformation, dims = t
    d = dimension(transformation)
    index = firstindex(x)
    yℓ = reshape([(y = transform_with(flag, transformation, view_into(x, index, d));
                   index += d;
                   y)
                  for _ in Base.OneTo(prod(dims))],
                 dims)
    first.(yℓ), sum(last, yℓ)
end

function transform_with(flag::LogJacFlag, t::ArrayTransform{Identity}, x::RealVector)
    # TODO use version below when https://github.com/FluxML/Flux.jl/issues/416 is fixed
    # y = reshape(copy(x), t.dims)
    y = reshape(map(identity, x), t.dims)
    y, logjac_zero(flag, eltype(x))
end

inverse_eltype(t::ArrayTransform, x::AbstractArray) =
    inverse_eltype(t.transformation, first(x)) # FIXME shortcut

function inverse!(x::RealVector,
                  transformation_array::ArrayTransform,
                  y::Array)
    @unpack transformation, dims = transformation_array
    @argcheck size(y) == dims
    index = firstindex(x)
    d = dimension(transformation)
    for elt in vec(y)
        inverse!(view_into(x, index, d), transformation, elt)
        index += d;
    end
    x
end


# tuples

const NTransforms{N} = Tuple{Vararg{AbstractTransform,N}}

"""
$(TYPEDEF)

Transform consecutive groups of real numbers to a tuple, using the given transformations.
"""
@calltrans struct TransformTuple{K, T <: NTransforms{K}} <: VectorTransform
    transformations::T
    dimension::Int
    function TransformTuple(transformations::T) where {K, T <: NTransforms{K}}
        new{K,T}(transformations, sum(dimension, transformations))
    end
end

dimension(tt::TransformTuple) = tt.dimension

"""
$(SIGNATURES)

Return a transformation that transforms consecutive groups of real numbers to a
(named) tuple, using the given transformations.
"""
as(transformations::NTransforms) = TransformTuple(transformations)

"""
$(SIGNATURES)

Helper function for transforming tuples. Used internally, to help type inference. Use via
`transfom_tuple` only.
"""
_transform_tuple(flag::LogJacFlag, x::RealVector, index) = (), logjac_zero(flag, eltype(x))

function _transform_tuple(flag::LogJacFlag, x::RealVector, index, tfirst, trest...)
    d = dimension(tfirst)
    yfirst, ℓfirst = transform_with(flag, tfirst, view_into(x, index, d))
    yrest, ℓrest = _transform_tuple(flag, x, index + d, trest...)
    (yfirst, yrest...), ℓfirst + ℓrest
end

"""
$(SIGNATURES)

Helper function for tuple transformations.
"""
transform_tuple(flag::LogJacFlag, tt::NTransforms, x::RealVector) =
    _transform_tuple(flag, x, firstindex(x), tt...)

"""
$(SIGNATURES)

Helper function determining element type of inverses from tuples. Used
internally.
"""
_inverse_eltype_tuple(ts::NTransforms{K}, ys::NTuple{K,Any}) where K =
    mapreduce(((t, y),) -> inverse_eltype(t, y), promote_type, zip(ts, ys))

"""
$(SIGNATURES)

Helper function for inverting tuples of transformations. Used internally.
"""
function _inverse!_tuple(x::RealVector, ts::NTransforms{K},
                         ys::NTuple{K,Any}) where K
    index = firstindex(x)
    for (t, y) in zip(ts, ys)
        d = dimension(t)
        inverse!(view_into(x, index, d), t, y)
        index += d
    end
    x
end

transform_with(flag::LogJacFlag, tt::TransformTuple, x::RealVector) =
    transform_tuple(flag, tt.transformations, x)

inverse_eltype(tt::TransformTuple{K}, y::NTuple{K,Any}) where K =
    _inverse_eltype_tuple(tt.transformations, y)

function inverse!(x::RealVector, tt::TransformTuple{K},
                  y::NTuple{K,Any}) where K
    @argcheck length(x) == dimension(tt)
    _inverse!_tuple(x, tt.transformations, y)
end

"""
$(TYPEDEF)

Transform consecutive groups of real numbers to a named tuple, using the given
transformations.
"""
@calltrans struct TransformNamedTuple{names, T <: NTransforms} <: VectorTransform
    transformations::T
    dimension::Int
    function TransformNamedTuple(transformations::NamedTuple{names,T}) where
        {names, T <: NTransforms}
        new{names,T}(values(transformations), sum(dimension, transformations))
    end
end

"""
$(SIGNATURES)
"""
as(transformations::NamedTuple{T,<:NTransforms}) where T =
    TransformNamedTuple(transformations)

dimension(tn::TransformNamedTuple) = tn.dimension

function transform_with(flag::LogJacFlag, tt::TransformNamedTuple{names},
                      x::RealVector) where {names}
    y, ℓ = transform_tuple(flag, tt.transformations, x)
    NamedTuple{names}(y), ℓ
end

inverse_eltype(tt::TransformNamedTuple{names},
               y::NamedTuple{names}) where names =
    _inverse_eltype_tuple(tt.transformations, values(y))

function inverse!(x::RealVector, tt::TransformNamedTuple{names},
                  y::NamedTuple{names}) where names
    @argcheck length(x) == dimension(tt)
    _inverse!_tuple(x, tt.transformations, values(y))
end
