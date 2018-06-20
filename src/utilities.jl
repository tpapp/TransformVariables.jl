
# logistic and logit

logistic(x::Real) = inv(one(x) + exp(-x))

function logistic(::LogJac, x::Real)
    mx = -abs(x)
    logistic(x), mx - 2*log1p(exp(mx))
end

logit(x::Real) = log(x / (one(x) - x))
