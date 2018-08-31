
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
$(SIGNATURES)

A view of `len` elements, starting after the index `previndex`.
"""
@inline viewafter(x::AbstractVector, previndex, len) =
    @inbounds view(x, (previndex + 1):(previndex + len))
