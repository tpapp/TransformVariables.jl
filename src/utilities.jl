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
