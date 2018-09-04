# Introduction

# General interface

```@docs
dimension
transform
transform_and_logjac
inverse
transform_logdensity
```

# Specific transformations

## Scalar transforms

```@docs
to_interval
∞
```

```@docs
to_ℝ
to_ℝ₊
to_ℝ₋
to_𝕀
```

## Special arrays

```@docs
to_unitvec
to_corr_cholesky
```

## Aggregation of transformations

```@docs
to_array
to_tuple
```

# Defining custom transformations

```@docs
logjac_forwarddiff
CustomTransform
```
