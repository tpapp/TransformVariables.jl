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

```@docs
dimension
transform
transform_and_logjac
```

```@docs
inverse
inverse!
inverse_eltype
```

```@docs
transform_logdensity
```

```@docs
random_arg
random_value
```

# Defining transformations

## The `as` constructor and aggregations

Some transformations, particularly *aggregations* use the function `as` as the constructor. Aggregating transformations are built from other transformations to transform consecutive (blocks of) real numbers into the desired domain.

It is recommended that you use `as(Array, ...)` and friends (`as(Vector, ...)`, `as(Matrix, ...)`) for repeating the *same* transformation, and named tuples such as `as((μ = ..., σ = ...))` for transforming into named parameters. For extracting parameters in log likelihoods, consider [Parameters.jl](https://github.com/mauro3/Parameters.jl).

See `methods(as)` for all the constructors, `?as` for their documentation.

```@docs
as
```

## Scalar transforms

The symbol `∞` is a placeholder for infinity. It does not correspond to `Inf`, but acts as a placeholder for the correct dispatch. `-∞` is valid.

```@docs
∞
```

`as(Real, a, b)` defines transformations to finite and (semi-)infinite subsets of the real line, where `a` and `b` can be `-∞` and `∞`, respectively. The following constants are defined for common cases.

```@docs
asℝ
asℝ₊
asℝ₋
as𝕀
```

## Special arrays

```@docs
UnitVector
CorrCholeskyFactor
```

# Defining custom transformations

```@docs
logjac_forwarddiff
value_and_logjac_forwarddiff
CustomTransform
```

# Internals

## Types for various transformations

These are not part of the API, use the `as` constructor or one of the predefined constants.

### Scalar transformations

```@docs
TransformVariables.Identity
TransformVariables.ScaledShiftedLogistic
TransformVariables.ShiftedExp
```

### Aggregating transformations

```@docs
TransformVariables.ArrayTransform
TransformVariables.TransformTuple
TransformVariables.TransformNamedTuple
```

### Wrapper for inverse

```@docs
TransformVariables.InverseTransform
```

## Types and type aliases

```@docs
TransformVariables.RealVector
TransformVariables.AbstractTransform
TransformVariables.ScalarTransform
TransformVariables.VectorTransform
```

## Conditional calculation of log Jacobian determinant

```@docs
TransformVariables.LogJacFlag
TransformVariables.LogJac
TransformVariables.NoLogJac
TransformVariables.logjac_zero
```

## Macros

```@docs
TransformVariables.@calltrans
```

## Helper functions

```@docs
TransformVariables.transform_with
TransformVariables._transform_tuple
TransformVariables._inverse!_tuple
TransformVariables._inverse_eltype_tuple
TransformVariables.unit_triangular_dimension
```

## Building blocks for transformations

```@docs
TransformVariables.l2_remainder_transform
TransformVariables.l2_remainder_inverse
```
