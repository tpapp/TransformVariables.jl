export to_array, to_tuple

####
#### array aggregator
####

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
    as(Array, [transformation], dims...)
    as(Array, [transformation], dims)

Return a transformation that applies `transformation` (which defaults to `asℝ`, the identity
transformation for scalars) repeatedly to create an array with the given `dims`.

`Matrix` or `Vector` can be used in place of `Array`, with conforming dimensions.

# Example

```julia
as(Array, asℝ₊, 2, 3)           # transform to a 2x3 matrix of positive numbers
as(Vector, 3)                   # ℝ³ → ℝ³, identity
```
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

function transform_with(flag::LogJacFlag, t::ArrayTransform, x, index::T) where {T}
    @unpack transformation, dims = t
    # NOTE not using index increments as that somehow breaks type inference
    d = dimension(transformation)
    𝐼 = reshape(range(index; length = prod(dims), step = d), dims)
    yℓ = map(index -> ((y, ℓ, _) = transform_with(flag, transformation, x, index); (y, ℓ)), 𝐼)
    ℓz = logjac_zero(flag, extended_eltype(x))
    first.(yℓ), isempty(yℓ) ? ℓz : ℓz + sum(last, yℓ), index
end

function transform_with(flag::LogJacFlag, t::ArrayTransform{Identity}, x, index)
    # TODO use version below when https://github.com/FluxML/Flux.jl/issues/416 is fixed
    # y = reshape(copy(x), t.dims)
    index′ = index+dimension(t)
    y = reshape(map(identity, x[index:(index′-1)]), t.dims)
    y, logjac_zero(flag, extended_eltype(x)), index′
end

function inverse_eltype(t::ArrayTransform, x::AbstractArray)
    inverse_eltype(t.transformation, first(x)) # FIXME shortcut
end

function inverse_at!(x::AbstractVector, index, transformation_array::ArrayTransform,
                     y::AbstractArray)
    @unpack transformation, dims = transformation_array
    @argcheck size(y) == dims
    for elt in vec(y)
        index = inverse_at!(x, index, transformation, elt)
    end
    index
end

####
#### Tuple and NamedTuple aggregators
####

"""
$(SIGNATURES)

Sum of the dimension of `transformations`. Utility function, *internal*.
"""
_sum_dimensions(transformations) = mapreduce(dimension, +, transformations; init = 0)

const NTransforms{N} = Tuple{Vararg{AbstractTransform,N}}

"""
$(TYPEDEF)

Transform consecutive groups of real numbers to a tuple, using the given transformations.
"""
@calltrans struct TransformTuple{T} <: VectorTransform
    transformations::T
    dimension::Int
    function TransformTuple(transformations::T) where {T <: NTransforms}
        new{T}(transformations, _sum_dimensions(transformations))
    end
    function TransformTuple(transformations::T
                            ) where {N, S <: NTransforms, T <: NamedTuple{N, S}}
        new{T}(transformations, _sum_dimensions(transformations))
    end
end

dimension(tt::TransformTuple) = tt.dimension

"""
    as(tuple)
    as(namedtuple)

Return a transformation that transforms consecutive groups of real numbers to a
(named) tuple, using the given transformations.

```jldoctest
julia> t = as((asℝ₊, UnitVector(3)));

julia> dimension(t)
3

julia> transform(t, zeros(dimension(t)))
(1.0, [0.0, 0.0, 1.0])

julia> t2 = as((σ = asℝ₊, u = UnitVector(3)));

julia> dimension(t2)
3

julia> transform(t2, zeros(dimension(t2)))
(σ = 1.0, u = [0.0, 0.0, 1.0])
```
"""
as(transformations::NTransforms) = TransformTuple(transformations)

"""
$(SIGNATURES)

Helper function for transforming tuples. Used internally, to help type inference. Use via
`transfom_tuple`.
"""
_transform_tuple(flag::LogJacFlag, x::AbstractVector, index, ::Tuple{}) =
    (), logjac_zero(flag, extended_eltype(x)), index

function _transform_tuple(flag::LogJacFlag, x::AbstractVector, index, ts)
    tfirst = first(ts)
    yfirst, ℓfirst, index′ = transform_with(flag, tfirst, x, index)
    yrest, ℓrest, index′′ = _transform_tuple(flag, x, index′, Base.tail(ts))
    (yfirst, yrest...), ℓfirst + ℓrest, index′′
end

"""
$(SIGNATURES)

Helper function for tuple transformations.
"""
function transform_tuple(flag::LogJacFlag, tt::NTransforms, x, index)
    _transform_tuple(flag, x, index, tt)
end

"""
$(SIGNATURES)

Helper function determining element type of inverses from tuples. Used
internally.

*Performs no argument validation, caller should do this.*
"""
_inverse_eltype_tuple(ts::NTransforms, ys::Tuple) =
    mapreduce(((t, y),) -> inverse_eltype(t, y), promote_type, zip(ts, ys))

"""
$(SIGNATURES)

Helper function for inverting tuples of transformations. Used internally.

*Performs no argument validation, caller should do this.*
"""
function _inverse!_tuple(x::AbstractVector, index, ts::NTransforms, ys::Tuple)
    for (t, y) in zip(ts, ys)
        index = inverse_at!(x, index, t, y)
    end
    index
end

function transform_with(flag::LogJacFlag, tt::TransformTuple{<:Tuple}, x, index)
    transform_tuple(flag, tt.transformations, x, index)
end

function inverse_eltype(tt::TransformTuple{<:Tuple}, y::Tuple)
    @unpack transformations = tt
    @argcheck length(transformations) == length(y)
    _inverse_eltype_tuple(transformations, y)
end

function inverse_at!(x::AbstractVector, index, tt::TransformTuple{<:Tuple}, y::Tuple)
    @unpack transformations = tt
    @argcheck length(transformations) == length(y)
    @argcheck length(x) == dimension(tt)
    _inverse!_tuple(x, index, tt.transformations, y)
end

as(transformations::NamedTuple{N,<:NTransforms}) where N =
    TransformTuple(transformations)

function transform_with(flag::LogJacFlag, tt::TransformTuple{<:NamedTuple}, x, index)
    @unpack transformations = tt
    y, ℓ, index′ = transform_tuple(flag, values(transformations), x, index)
    NamedTuple{keys(transformations)}(y), ℓ, index′
end

function inverse_eltype(tt::TransformTuple{<:NamedTuple}, y::NamedTuple)
    @unpack transformations = tt
    @argcheck keys(transformations) == keys(y)
    _inverse_eltype_tuple(values(transformations), values(y))
end

function inverse_at!(x::AbstractVector, index, tt::TransformTuple{<:NamedTuple}, y::NamedTuple)
    @unpack transformations = tt
    @argcheck keys(transformations) == keys(y)
    @argcheck length(x) == dimension(tt)
    _inverse!_tuple(x, index, values(transformations), values(y))
end
