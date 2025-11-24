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

function _summary_rows(transformation::ArrayTransformation, mime)
    (; inner_transformation, dims) = transformation
    _dims = foldr((a,b) -> "$(string(a))×$(string(b))", dims, init = "")
    rows = _summary_row(transformation, _dims)
    for row in _summary_rows(inner_transformation, mime)
        push!(rows, (level = row.level + 1, indices = nothing, repr = row.repr))
    end
    rows
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
    (; inner_transformation, dims) = transformation
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
    index′ = index+dimension(t)
    y = reshape(x[index:(index′-1)], t.dims)
    y, logjac_zero(flag, robust_eltype(x)), index′
end

"""
$(SIGNATURES)

Implementation of array domain labels, for reuse in the transformations that implement
variations. Internal, not exported.
"""
function _array_domain_label(inner_transformation, dims, index::Int)
    n, r = divrem(index, dimension(inner_transformation))
    (Tuple(CartesianIndices(dims)[n]), _domain_label(inner_transformation, r)...)
end

function _domain_label(transformation::ArrayTransformation, index::Int)
    (; inner_transformation, dims) = transformation
    _array_domain_label(inner_transformation, dims, index)
end

####
#### array view
####

"""
$(TYPEDEF)

View of an array with `dims`.

!!! note
    This feature is experimental, and not part of the stable API; it may disappear or change without
    relevant changes in SemVer or deprecations. Inner transformations are not supported.
"""
struct ViewTransformation{M} <: VectorTransform
    dims::NTuple{M, Int}
end

function as(::typeof(view), dims::Tuple{Vararg{Int}})
    @argcheck all(d -> d ≥ 0, dims) "All dimensions need to be non-negative."
    ViewTransformation(dims)
end

as(::typeof(view), dims::Int...) = as(view, dims)

dimension(transformation::ViewTransformation) = prod(transformation.dims)

function transform_with(flag::LogJacFlag, t::ViewTransformation, x, index)
    index′ = index + dimension(t)
    y = reshape(@view(x[index:(index′-1)]), t.dims)
    y, logjac_zero(flag, robust_eltype(x)), index′
end

function _domain_label(transformation::ViewTransformation, index::Int)
    (; dims) = transformation
    _array_domain_label(asℝ, dims, index)
end

function inverse_eltype(transformation::ViewTransformation,
                        ::Type{T}) where T <: AbstractArray
    _ensure_float(eltype(T))
end

function inverse_at!(x::AbstractVector, index, transformation::ViewTransformation,
                     y::AbstractArray)
    @argcheck size(y) == transformation.dims
    index′ = index + dimension(transformation)
    copy!(@view(x[index:(index′-1)]), vec(y))
    index′
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
                        x::AbstractVector{T}, index::Int) where {D,S,T}
    (; inner_transformation) = transformation
    # NOTE this is a fix for #112, enforcing types taken from the transformation of the
    # first element.
    y1, ℓ1, index1 = transform_with(flag, inner_transformation, x, index)
    D == 1 && return SArray{S}(y1), ℓ1, index1
    L = typeof(ℓ1)
    let ℓ::L = ℓ1, index::Int = index1
        function _f(_)
            y, ℓΔ, index′ = transform_with(flag, inner_transformation, x, index)
            index = index′
            ℓ = ℓ + ℓΔ
            y
        end
        yrest = SVector{D-1}(_f(i) for i in 2:D)
        SArray{S}(pushfirst(yrest, y1)), ℓ, index
    end
end

function inverse_eltype(transformation::Union{ArrayTransformation,StaticArrayTransformation},
                        ::Type{T}) where T <: AbstractArray
    inverse_eltype(transformation.inner_transformation,
                   _ensure_float(eltype(T)))
end

function inverse_at!(x::AbstractVector, index,
                     transformation::Union{ArrayTransformation,StaticArrayTransformation},
                     y::AbstractArray)
    (; inner_transformation) = transformation
    dims = result_size(transformation)
    @argcheck size(y) == dims
    for elt in vec(y)
        index = inverse_at!(x, index, inner_transformation, elt)
    end
    index
end

function _domain_label(transformation::StaticArrayTransformation{D,S}, index::Int) where {D,S}
    _array_domain_label(transformation.inner_transformation, fieldtypes(S), index)
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
    inner::T
    dimension::Int
    function TransformTuple(inner::T) where {T <: NTransforms}
        new{T}(inner, _sum_dimensions(inner))
    end
    function TransformTuple(inner::T
                            ) where {N, S <: NTransforms, T <: NamedTuple{N, S}}
        new{T}(inner, _sum_dimensions(inner))
    end
end


"""
$(SIGNATURES)

Helper function for accessing the `inner` field, as we define `getproperty` which masks
this. Internal.
"""
_inner(t) = getfield(t, :inner)

###
### expose inner tuple via indices and properties
###

Base.length(t::TransformTuple) = length(_inner(t))
Base.getindex(t::TransformTuple, i::Int) = getindex(_inner(t), i)
Base.propertynames(t::TransformTuple) = propertynames(_inner(t))
Base.getproperty(t::TransformTuple, i::Int) = getproperty(_inner(t), i)
Base.getproperty(t::TransformTuple{<:NamedTuple}, i::Symbol) = getproperty(_inner(t), i)

function _summary_rows(transformation::TransformTuple, mime)
    inner = _inner(transformation)
    repr1 = string(nameof(typeof(inner)), " of transformations")
    rows = _summary_row(transformation, repr1)
    _index = 0
    for (key, t) in pairs(inner)
        for row in _summary_rows(t, mime)
            _repr = row.level == 1 ? (repr(key) * " → " * row.repr) : row.repr
            push!(rows, (level = row.level + 1, indices = _offset(row.indices, _index),
                         repr = _repr))
        end
        _index += dimension(t)
    end
    rows
end

dimension(tt::TransformTuple) = getfield(tt, :dimension)

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

## Element access and modification

The resulting objects support `getindex` (`transformation[i]`), `getproperty`
(`transformation.key`)`, and `length`:

```jldoctest
julia> t = as((a = asℝ₊, b = asℝ))
[1:2] NamedTuple of transformations
  [1:1] :a → asℝ₊
  [2:2] :b → asℝ

julia> t.a
asℝ₊ (dimension 1)

julia> t[2]
asℝ (dimension 1)

julia> length(t)
2
```

You can also use the API from [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl):

```jldoctest
julia> using Accessors

julia> t = as((a = asℝ₊, b = asℝ))
[1:2] NamedTuple of transformations
  [1:1] :a → asℝ₊
  [2:2] :b → asℝ

julia> @set t.a = as𝕀
[1:2] NamedTuple of transformations
  [1:1] :a → as𝕀
  [2:2] :b → asℝ
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
function _inverse_eltype_tuple(ts::NTransforms{N}, ::Type{T}) where {N,T<:Tuple}
    @argcheck T <: NTuple{N,Any} "Incompatible input length."
    __inverse_eltype_tuple(ts, T)
end
function __inverse_eltype_tuple(ts::NTransforms, ::Type{Tuple{}})
    Union{}
end
function __inverse_eltype_tuple(ts::NTransforms, ::Type{T}) where {T<:Tuple}
    promote_type(inverse_eltype(Base.first(ts), fieldtype(T, 1)),
                 __inverse_eltype_tuple(Base.tail(ts), Tuple{Base.tail(fieldtypes(T))...}))
end

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
    transform_tuple(flag, _inner(tt), x, index)
end

function inverse_eltype(tt::TransformTuple{<:Tuple}, ::Type{T}) where T <: Tuple
    _inverse_eltype_tuple(_inner(tt), T)
end

function inverse_at!(x::AbstractVector, index, tt::TransformTuple{<:Tuple}, y::Tuple)
    inner = _inner(tt)
    @argcheck length(inner) == length(y)
    _inverse!_tuple(x, index, inner, y)
end

function as(inner::NamedTuple{N,<:NTransforms}) where N
    TransformTuple(inner)
end

function transform_with(flag::LogJacFlag, tt::TransformTuple{<:NamedTuple}, x, index)
    inner = _inner(tt)
    y, ℓ, index′ = transform_tuple(flag, values(inner), x, index)
    NamedTuple{keys(inner)}(y), ℓ, index′
end

function inverse_eltype(tt::TransformTuple{I},
                        ::Type{NT}) where {I<:NamedTuple,NT<:NamedTuple}
    inner = _inner(tt)
    _check_name_compatibility(NT,I)
    _inverse_eltype_tuple(values(inner), _reshuffle_namedtuple_fieldtypes(I, NT))
end

function inverse_at!(x::AbstractVector, index, tt::TransformTuple{I},
                     y::NamedTuple) where {I}
    inner = _inner(tt)
    _check_name_compatibility(typeof(y), I)
    _inverse!_tuple(x, index, values(inner), values(NamedTuple{keys(inner)}(y)))
end

function _domain_label(t::TransformTuple, index::Int)
    for (key, inner_transformation) in pairs(_inner(t))
        d = dimension(inner_transformation)
        if index ≤ d
            l = key isa Symbol ? key : (key, )
            return (l, _domain_label(inner_transformation, index)...)
        else
            index -= d
        end
    end
    error("internal error")
end
