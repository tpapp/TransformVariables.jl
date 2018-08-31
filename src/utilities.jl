
# logistic and logit

logistic(x::Real) = inv(one(x) + exp(-x))

function logistic_logjac(x::Real)
    mx = -abs(x)
    mx - 2*log1p(exp(mx))
end

logit(x::Real) = log(x / (one(x) - x))

"""
    $SIGNATURES

Number of elements (strictly) above the diagonal in an ``n×n`` matrix.
"""
unit_triangular_dimension(n::Int) = n * (n-1) ÷ 2

"""
Implement a view into a vector starting at a given element. Uses generalized
indexing.

!!! note

    Bounds are not (yet) checked, may be implemented later.
"""
struct IndexInto{T, S <: AbstractVector{T}} <: AbstractVector{T}
    i::Int
    parent::S
end

"""
$SIGNATURES

A wrapper functionally equivalent to `@view v[i:end]`.
"""
index_into(v::AbstractVector, i) = IndexInto(i, v)

index_into(v::IndexInto, i) = IndexInto(i, v.parent)

Base.axes(I::IndexInto) = (firstindex(I):lastindex(I), )

Base.IndexStyle(::Type{<:IndexInto}) = IndexLinear()

Base.length(I::IndexInto) = length(I.parent) - I.i + 1

Base.size(I::IndexInto) = (length(I), )

Base.firstindex(I::IndexInto) = I.i

Base.lastindex(I::IndexInto) = lastindex(I.parent)

Base.getindex(I::IndexInto, i::Int) = I.parent[i]

Base.first(I::IndexInto) = I.parent[I.i]

Base.view(I::IndexInto, i) = view(I.parent, i)
