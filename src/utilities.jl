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
