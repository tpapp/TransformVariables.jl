
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


# view management

"""
$SIGNATURES

A view of `v` starting from `i` for `len` elements, no bounds checking.
"""
view_into(v::AbstractVector, i, len) = @inbounds view(v, i:(i+len-1))


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
