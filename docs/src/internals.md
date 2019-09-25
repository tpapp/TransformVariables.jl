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
```

### Wrapper for inverse

```@docs
TransformVariables.InverseTransform
```

## Types and type aliases

```@docs
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
