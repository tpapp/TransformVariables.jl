export to_array, to_tuple

####
#### array aggregator
####

"""
$(TYPEDEF)

Apply `transformation` repeatedly to create an array with given `dims`.
"""
struct ArrayTransformation{T <: AbstractTransform,M} <: VectorTransform
    inner_transformation::T
    dims::NTuple{M, Int}
end

function dimension(transformation::ArrayTransformation)
    dimension(transformation.inner_transformation) * prod(transformation.dims)
end

result_size(transformation::ArrayTransformation) = transformation.dims

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
function as(::Type{Array}, transformation::AbstractTransform, dims::Tuple{Vararg{Int}})
    ArrayTransformation(transformation, dims)
end

as(::Type{Array}, dims::Tuple{Vararg{Int}}) = as(Array, Identity(), dims)

function as(::Type{Array}, transformation::AbstractTransform, dims::Int...)
    ArrayTransformation(transformation, dims)
end

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

function transform_with(flag::LogJacFlag, transformation::ArrayTransformation, x, index::T) where {T}
    @unpack inner_transformation, dims = transformation
    # NOTE not using index increments as that somehow breaks type inference
    d = dimension(inner_transformation) # length of an element transformation
    len = prod(dims)              # number of elements
    𝐼 = reshape(range(index; length = len, step = d), dims)
    yℓ = map(index -> ((y, ℓ, _) = transform_with(flag, inner_transformation, x, index); (y, ℓ)), 𝐼)
    ℓz = logjac_zero(flag, robust_eltype(x))
    index′ = index + d * len
    first.(yℓ), isempty(yℓ) ? ℓz : ℓz + sum(last, yℓ), index′
end

function transform_with(flag::LogJacFlag, t::ArrayTransformation{Identity}, x, index)
    # TODO use version below when https://github.com/FluxML/Flux.jl/issues/416 is fixed
    # y = reshape(copy(x), t.dims)
    index′ = index+dimension(t)
    y = reshape(map(identity, x[index:(index′-1)]), t.dims)
    y, logjac_zero(flag, robust_eltype(x)), index′
end

####
#### static array
####

"""
Transform into a static array.
"""
struct StaticArrayTransformation{D,S,T} <: VectorTransform
    inner_transformation::T
end

"""
    as(SArray{S}, [inner_transformation])

Return a transformation that applies `inner_transformation` (which defaults to `asℝ`, the
identity transformation for scalars) repeatedly to create an array with the given dimensions.

`SMatrix` or `SVector` can be used in place of `SArray`, with conforming dimensions.

# Example

```julia
as(SArray{2,3}, asℝ₊, 2, 3)     # transform to a 2x3 SMatrix of positive numbers
as(SVector{3})                   # ℝ³ → ℝ³, identity, but an SVector
```
"""
function as(::Type{<:SArray{S}}, inner_transformation = Identity()) where S
    dim = fieldtypes(S)
    @argcheck all(x -> x ≥ 1, dim)
    StaticArrayTransformation{prod(dim),S,typeof(inner_transformation)}(inner_transformation)
end

function dimension(transformation::StaticArrayTransformation{D}) where D
    D * dimension(transformation.inner_transformation)
end

result_size(::StaticArrayTransformation{D,S}) where {D,S} = fieldtypes(S)

function transform_with(flag::LogJacFlag, transformation::StaticArrayTransformation{D,S},
                        x::AbstractVector{T}, index) where {D,S,T}
    @unpack inner_transformation = transformation
    ℓ = logjac_zero(flag, robust_eltype(x))
    SArray{S}(begin
                  y, ℓΔ, index′ = transform_with(flag, inner_transformation, x, index)
                  index = index′
                  ℓ += ℓΔ
                  y
              end
              for _ in 1:D), ℓ
end

function inverse_eltype(transformation::Union{ArrayTransformation,StaticArrayTransformation},
                        x::AbstractArray)
    inverse_eltype(transformation.inner_transformation, first(x)) # FIXME shortcut
end

function inverse_at!(x::AbstractVector, index,
                     transformation::Union{ArrayTransformation,StaticArrayTransformation},
                     y::AbstractArray)
    @unpack inner_transformation = transformation
    dims = result_size(transformation)
    @argcheck size(y) == dims
    for elt in vec(y)
        index = inverse_at!(x, index, inner_transformation, elt)
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
_sum_dimensions(transformations) = reduce(+, map(dimension, transformations), init = 0)
# NOTE: See https://github.com/tpapp/TransformVariables.jl/pull/80
#       `map` and `reduce` both have specializations on `Tuple`s that make them type stable
#       even when the `Tuple` is heterogenous, but that is not currently the case with
#       `mapreduce`, therefore separate `reduce` and `map` are preferred as a workaround.

const NTransforms{N} = Tuple{Vararg{AbstractTransform,N}}

"""
$(TYPEDEF)

Transform consecutive groups of real numbers to a tuple, using the given transformations.
"""
struct TransformTuple{T} <: VectorTransform
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
    (), logjac_zero(flag, robust_eltype(x)), index

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
    reduce(promote_type, map(inverse_eltype, ts, ys))
# NOTE: See https://github.com/tpapp/TransformVariables.jl/pull/80
#       `map` and `reduce` both have specializations on `Tuple`s that make them type stable
#       even when the `Tuple` is heterogenous, but that is not currently the case with
#       `mapreduce`, therefore separate `reduce` and `map` are preferred as a workaround.

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
    _inverse!_tuple(x, index, values(transformations), values(y))
end
