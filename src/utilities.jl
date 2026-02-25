###
### logistic and logit
###


"""
    $(SIGNATURES)

When `a <: Reactant.AnyTracedRArray` this adds a `@allowscalar` annotation so that the function can 
be compiled with Reactant. When `a` is not a `Reactant.AnyTracedRArray` it simply returns `a[i]`.

!!! warn
    This is necessary because by default Reactant does not allow scalar indexing of arrays unless you
    opt in. Note that often `Reactant` is able to raise the scalar indexing to the level of the whole
    array so this operation is not necessarily slow, but there are no guarantees. 
"""
Base.@propagate_inbounds function tv_getindex(a, i)
    return a[i]
end

function logistic_logjac(x::Number)
    mx = -abs(x)
    mx - 2*log1pexp(mx)
end

logit_logjac(y) = -log(y) - log1p(-y)

###
### calculations
###

"""
$(SIGNATURES)

Number of elements (strictly) above the diagonal in an ``n×n`` matrix.
"""
unit_triangular_dimension(n::Int) = n * (n-1) ÷ 2

# Adapted from LinearAlgebra.__normalize!
# MIT license
# Copyright (c) 2018-2024 LinearAlgebra.jl contributors: https://github.com/JuliaLang/LinearAlgebra.jl/contributors
@inline function __normalize!(a::AbstractArray, nrm)
    # The largest positive floating point number whose inverse is less than infinity
    δ = inv(prevfloat(typemax(nrm)))
    if nrm ≥ δ # Safe to multiply with inverse
        invnrm = inv(nrm)
        rmul!(a, invnrm)
    else # scale elements to avoid overflow
        εδ = eps(one(nrm))/δ
        rmul!(a, εδ)
        rmul!(a, inv(nrm*εδ))
    end
    return a
end

###
### type calculations
###

"""
$(SIGNATURES)

Regularize scalar (element) types to a floating point, falling back to `Float64`.

It serves two purposes:

1. broaden non-float type (eg `Int`) so that they can accommodate the algebraic results,
   eg mapping with `log`,

2. assign a sensible fallback type (currently `Float64`) for non-numerical element
   types; for example, if the input is `Vector{Any}`, `_ensure_float(Any)` will return
   `Float64`,

It is implicitly assumed that the input type is such that it can hold numerical values.
This is typically harmless, since containes for other types (eg `Union{}`, `Nothing`)
will fail anyway.

!!! NOTE
    Call this function *after* stripping units and similar, so that the input is a
    subtype of `Real` in most cases.
"""
_ensure_float(::Type) = Float64 # fallback for Any etc.

# heuristic: it is assumed that every `Real` type defines `float`.
# In case this does not hold, the package that defined `T` define `float(::Type{T})`.
_ensure_float(::Type{T}) where {T<:Real} = float(T)

"""
$(SIGNATURES)

Check that the *first* argument has all the names required to instantiate a `NamedTuple`
specified in the second argument.

Currently it only supports `NamedTuple`s for its first argument, and checks there are no
extra fields. Both requirements may be relaxed in the future.

If the requirements are not met, throw error with an informative message, otherwise
return `nothing`.
"""
function _check_name_compatibility(::Type{<:NamedTuple{A}},
                                   ::Type{<:NamedTuple{B}}) where {A,B}
    for b in B
        b ∈ A || throw(ArgumentError(LazyString("Property :", b, " not in ", A, ".")))
    end
    length(A) == length(B) || throw(ArgumentError(LazyString("Names ", A, " has extras compared to ", B, ".")))
    nothing
end

"""
$(SIGNATURES)

Return the reordered fieldtypes of the second argument (a `NamedTuple` type) according
to to the names in the first, for which specifying `NamedTuple{N}` is sufficient.

An error is thrown if an name is not found, but otherwise no comparison is done. Caller
should check with [`_check_name_compatibility`](@ref) first to throw an informative
error message.

```jldoctest
julia> TransformVariables._reshuffle_namedtuple_fieldtypes(NamedTuple{(:a,:b)},NamedTuple{(:b,:a),Tuple{Int64,Float64}})
Tuple{Float64, Int64}
```
"""
function _reshuffle_namedtuple_fieldtypes(::Type{<:NamedTuple{N}}, ::Type{NT}) where {N,NT<:NamedTuple}
    Tuple{map(n -> fieldtype(NT, n), N)...}
end
