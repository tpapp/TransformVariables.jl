# Introduction

# General interface

```@docs
dimension
transform
transform_and_logjac
inverse
transform_logdensity
as
```

# Specific transformations

## Scalar transforms

```@docs
∞
```

```@docs
ℝ
ℝ₊
ℝ₋
𝕀
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
