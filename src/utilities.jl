
# logistic and logit

logistic(x::Real) = inv(one(x) + exp(-x))

function logistic_logjac(x::Real)
    mx = -abs(x)
    mx - 2*log1p(exp(mx))
end

logit(x::Real) = log(x / (one(x) - x))


# calculations

"""
    $SIGNATURES

Number of elements (strictly) above the diagonal in an ``n×n`` matrix.
"""
unit_triangular_dimension(n::Int) = n * (n-1) ÷ 2


# a lightweight view type

"""
Implement a view into a vector starting at a given element. Uses generalized
indexing.

!!! note

    Bounds are not (yet) checked, may be implemented later.

    This is an internal implementation detail, not meant to be exposed by the API.

    Use the `index_into` constructor internally.
"""
struct IndexInto{T, S <: AbstractVector{T}} <: AbstractVector{T}
    parent::S
    i::Int
    len::Int
end

"""
$SIGNATURES

A wrapper functionally equivalent to `@view v[i:end]`, no bounds checking.
"""
index_into(v::AbstractVector, i, len) = IndexInto(v, i, len)

index_into(v::IndexInto, i, len) = IndexInto(v.parent, i, len)

Base.axes(I::IndexInto) = (firstindex(I):lastindex(I), )

Base.IndexStyle(::Type{<:IndexInto}) = IndexLinear()

Base.length(I::IndexInto) = I.len

Base.size(I::IndexInto) = (length(I), )

Base.firstindex(I::IndexInto) = I.i

Base.lastindex(I::IndexInto) = I.i + I.len - 1

Base.getindex(I::IndexInto, i) = I.parent[i] # NO CHECKING

Base.setindex!(I::IndexInto, value, i) = setindex!(I.parent, value, i) # NO CHECkiNG

Base.first(I::IndexInto) = I.parent[I.i]

Base.view(I::IndexInto, i) = view(I.parent, i)

function Base.copy(I::IndexInto)
    @unpack i, len, parent = I
    I[i:(i + len - 1)]
end


# macros

"""
$(SIGNATURES)

Workaround for https://github.com/JuliaLang/julia/issues/14919 to make
transformation types callable.

TODO: remove when this issue is closed, also possibly remove MacroTools as a
dependency if not used elsewhere.
"""
macro calltrans(ex)
    if @capture(ex, struct T1_ fields__ end)
        @capture T1 (T2_ <: S_|T2_)
        @capture T2 (T3_{params__}|T3_)
        quote
            Base.@__doc__ $(esc(ex))
            (t::$(esc(T3)))(x) = transform(t, x)
        end
    else
        throw(ArgumentError("can't find anything to make callable in $(ex)"))
    end
end
