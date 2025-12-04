# Introduction

```@setup ex1
import Random
Random.seed!(1)
```

Some problems, especially in numerical integration and Markov Chain Monte Carlo, benefit from *transformation* of variables: for example, if ``σ > 0`` is a standard deviation parameter, it is usually better to work with `log(σ)` which can take any value on the real line. However, in general such transformations require correcting density functions by the determinant of their Jacobian matrix, usually referred to as "the Jacobian".

Also, is usually easier to code MCMC algorithms to work with vectors of real numbers, which may represent a "flattened" version of parameters, and would need to be decomposed into individual parameters, which themselves may be arrays, tuples, or special objects like lower triangular matrices.

This package is designed to help with both of these use cases. For example, consider the "8 schools" problem from Chapter 5.5 of Gelman et al (2013), in which SAT scores ``y_{ij}`` in ``J=8`` schools are modeled using a conditional normal

```math
y_{ij} ∼ N(θⱼ, σ²)
```
and the ``θⱼ`` are assume to have a hierarchical prior distribution

```math
θⱼ ∼ N(μ, τ²)
```

For this problem, one could define a transformation

```@example ex1
using TransformVariables
t = as((μ = asℝ, σ = asℝ₊, τ = asℝ₊, θs = as(Array, 8)))
dimension(t)
```

which would then yield a `NamedTuple` with the given names, with one of them being a `Vector`:

```@repl ex1
x = randn(dimension(t))
y = transform(t, x)
keys(y)
y.θs
```

Further worked examples of using this package can be found in the [DynamicHMCExamples.jl](https://github.com/tpapp/DynamicHMCExamples.jl/) repository. It is recommended that the user reads those first, and treats this documentation as a reference.

# General interface

## Transformations

```@docs
dimension
transform
transform_and_logjac
```

## Inverses

```@docs
inverse
inverse!
inverse_eltype
```

## Integration into Bayesian inference

```@docs
transform_logdensity
TransformVariables.logprior
TransformVariables.nonzero_logprior
```

## Miscellaneous

```@docs
domain_label
```

# Defining transformations

## The `as` constructor and aggregations

Some transformations, particularly *aggregations* use the function `as` as the constructor. Aggregating transformations are built from other transformations to transform consecutive (blocks of) real numbers into the desired domain.

It is recommended that you use `as(Array, ...)` and friends (`as(Vector, ...)`, `as(Matrix, ...)`) for repeating the *same* transformation, and named tuples such as `as((μ = ..., σ = ...))` for transforming into named parameters. For extracting parameters in log likelihoods, consider [Parameters.jl](https://github.com/mauro3/Parameters.jl).

See `methods(as)` for all the constructors, `?as` for their documentation.

```@docs
as
```

Transforms which produce `NamedTuple`s can be `merge`d, which internally calls `Base.merge`; name collisions will thus follow `Base` behavior, which is that the right-most instance will be kept.
When using e.g. [`ConstructionBase.setproperties`](https://juliaobjects.github.io/ConstructionBase.jl/stable/#ConstructionBase.setproperties) to map a vector onto a subset of parameters stored in a struct, this functionality allows transforms for different parameter subsets to be constructed for use separately or together:

```julia
t_a = as((;a = asℝ₊))
t_b = as((;b = as𝕀))
t_c = as((;c = TVShift(5) ∘ TVExp()))
t_ab = merge(t_a, t_b)
t_abc = merge(t_ab, t_c)
t_abc = merge(t_a, t_b, t_c)
t_collision = merge(t_a, as((;a = asℝ₋))) # Will have a = asℝ₋, from rightmost
```

## Scalar transforms

The symbol `∞` is a placeholder for infinity. It does not correspond to `Inf`, but acts as a placeholder for the correct dispatch. `-∞` is valid.

```@docs
∞
```

`as(Real, a, b)` defines transformations to finite and (semi-)infinite subsets of the real line, where `a` and `b` can be `-∞` and `∞`, respectively.

```@docs
as(::Type{Real}, Any, Any)
```

The following constants are defined for common cases.

```@docs
asℝ
asℝ₊
asℝ₋
as𝕀
```

For more granular control than the `as(Real, a, b)`, scalar transformations can be built from individual elements with the composition operator `∘` (typed as `\circ<tab>`):

```@docs
TVExp
TVLogistic
TVScale
TVShift
TVNeg
```

Consistent with common notation, transforms are applied right-to-left; for example, `as(Real, ∞, 3)` is equivalent to `TVShift(3) ∘ TVNeg() ∘ TVExp()`.
If you are working in an editor where typing Unicode is difficult, `TransformVariables.compose` is also available, as in `TransformVariables.compose(TVScale(5.0), TVNeg(), TVExp())`.

This composition works with any scalar transform in any order, so `TVScale(4) ∘ as(Real, 2, ∞) ∘ TVShift(1e3)` is a valid transform.
This is useful especially for making sure that values near 0, when transformed, yield usefully-scaled values for a given variable.

In addition, the `TVScale` transform accepts arbitrary types. It can be used as the outermost transform (so leftmost in the composition) to add, for example, `Unitful` units to a number (or to create other exotic number types which can be constructed by multiplying, such as a `ForwardDiff.Dual`).

However, note that calculating log Jacobian determinants may error for types that are not real numbers.
For example, 

```julia
using Unitful
t = TVScale(5u"m") ∘ TVExp()
```
produces positive quantities with the dimension of length. 
!!! note
    Because the log-Jacobian of a transform that adds units is not defined, `transform_and_logjac` and `inverse_and_logjac`
    only have methods defined for `TVScale{T} where {T<:Real}`. 
!!! note
    The inverse transform of `TVScale(scale)` divides by `scale`, which is the correct inverse for adding units to a number, but may be inappropriate for other custom number types. A transform that doesn't just multiply or an inverse that extracts a float from an exotic number type could be defined by adding methods to `transform` and `inverse` like the following:
    ```
    transform(t::TVScale{T}, x) where T<:MyCustomNumberType = MyCustomNumberType(x)
    inverse(t::TVScale{T}, x) where T<:MyCustomNumberType = get_the_float_part(x)```

## Special arrays

```@docs
unit_vector_norm
UnitVector
UnitSimplex
CorrCholeskyFactor
corr_cholesky_factor
```

## Miscellaneous transformations

```@docs
Constant
```

# Defining custom transformations

```@docs
logjac_forwarddiff
value_and_logjac_forwarddiff
CustomTransform
```
