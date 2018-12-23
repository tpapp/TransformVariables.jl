###
### logistic and logit
###

logistic(x::Real) = inv(one(x) + exp(-x))

function logistic_logjac(x::Real)
    mx = -abs(x)
    mx - 2*log1p(exp(mx))
end

logit(x::Real) = log(x / (one(x) - x))

logit_logjac(y) = -log(y) - log1p(-y)

###
### calculations
###

"""
    $SIGNATURES

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
function extended_eltype(::Type{S}) where S
    T = eltype(S)
    T <: Real ? typeof(√(one(T))) : Any
end

extended_eltype(x::T) where T = extended_eltype(T)

###
### view management
###

"""
$SIGNATURES

A view of `v` starting from `i` for `len` elements, no bounds checking.
"""
view_into(v::AbstractVector, i, len) = @inbounds view(v, i:(i+len-1))

###
### macros
###

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

####
#### random values
####

"Shared part of docstrings for keyword arguments of or passed to [`random_reals`](@ref)."
const _RANDOM_REALS_KWARGS_DOC = """
A standard multivaritate normal or Cauchy is used, depending on `cauchy`, then scaled with
`scale`. `rng` is the random number generator used.
"""

_random_reals_scale(rng::AbstractRNG, scale::Real, cauchy::Bool) =
    cauchy ? scale / abs2(randn(rng)) : scale * 1.0

"""
$(SIGNATURES)

Random real number.

$(_RANDOM_REALS_KWARGS_DOC)
"""
random_real(; scale::Real = 1, cauchy::Bool = false, rng::AbstractRNG = GLOBAL_RNG) =
    randn(rng) * _random_reals_scale(rng, scale, cauchy)

"""
$(SIGNATURES)

Random vector in ``ℝⁿ``.

$(_RANDOM_REALS_KWARGS_DOC)
"""
random_reals(n::Integer; scale::Real = 1, cauchy::Bool = false,
             rng::AbstractRNG = GLOBAL_RNG) =
                 randn(rng, n) .* _random_reals_scale(rng, scale, cauchy)
