# Introduction

Some problems, especially in numerical integration and Markov Chain Monte Carlo, benefit from *transformation* of variables: for example, if ``σ > 0`` is a standard deviation parameter, it is usually better to work with `log(σ)` which can take any value on the real line. However, in general such transformations require correcting density functions by the determinant of their Jacobian matrix, usually referred to as "the Jacobian". Also, is usually easier to code MCMC algorithms to work with vectors of real numbers, which may represent a "flattened" version of parameters, and would need to be decomposed into individual parameters, which themselves may be arrays, tuples, or special objects like lower triangular matrices. This package is designed to help with both of these use cases.

# General interface

```@docs
as
```

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

# Specific transformations

## Scalar transforms

```@docs
∞
```

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

## Aggregation of transformations

FIXME explain `as` syntax

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
TransformVariables.TupleTransform
TransformVariables.NamedTupleTransform
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

## Fast vector views

```@docs
TransformVariables.index_into
TransformVariables.IndexInto
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
