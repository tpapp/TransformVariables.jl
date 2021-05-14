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
function extended_eltype(::Type{S}) where S
    T = eltype(S)
    T <: Real ? typeof(√(one(T))) : Any
end

extended_eltype(x::T) where T = extended_eltype(T)

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
