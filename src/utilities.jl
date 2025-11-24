###
### logistic and logit
###

function logistic_logjac(x::Real)
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

Extend element type of argument so that it is closed under the algebra used by this package.

Pessimistic default for non-real types.
"""
function robust_eltype(::Type{S}) where S
    T = eltype(S)
    T <: Real ? typeof(√(one(T))) : Any
end

robust_eltype(x::T) where T = robust_eltype(T)

"""
$(SIGNATURES)

Regularize input type, preferring a floating point, falling back to `Float64`.

Internal, not exported.

# Motivation

Type calculations occasionally give types that are too narrow (eg `Union{}` for empty
vectors) or broad. Since this package is primarily intended for *numerical*
calculations, we fall back to something sensible. This function implements the
heuristics for this, and is currently used in inverse element type calculations.
"""
function _ensure_float(::Type{T}) where T
    if T <: Number # heuristic: it is assumed that every `Number` type defines `float`
        return float(T)
    else
        return Float64
    end
end

# pass through containers
_ensure_float(::Type{T}) where {T<:AbstractArray} = T

# special case Union{}
_ensure_float(::Type{Union{}}) = Float64

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
@generated function _reshuffle_namedtuple_fieldtypes(::Type{<:NamedTuple{N}},
                                                     ::Type{<:NamedTuple{M,K}}) where {N,M,K}
    _K = fieldtypes(K)
    S = map(N) do n
        i = findfirst(m -> m ≡ n, M)
        @assert i ≢ nothing
        _K[i]
    end
    :(Tuple{$(S...)})
end
