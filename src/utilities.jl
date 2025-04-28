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
    if T !== Union{} && T <: Number # heuristic: it is assumed that every `Number` type defines `float`
        return float(T)
    else
        return Float64
    end
end
