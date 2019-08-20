# Unreleased

# 0.3.5

- use new LogDensityProblems interface

- rewrite internals to work better with AD (especially Zygote)

- fix broadcasting (`Ref(transformation)` no longer necessary)

# 0.3.4

- make `inverse(::ArrayTransform)` accept `AbstractArray`

# 0.3.3

- Add ASCII aliases for common scalar transforms

# 0.3.2

- Check length of inputs for CorrCholeskyFactor
- minor doc updates for CorrCholeskyFactor

# 0.3.1

- Catch wrong size UnitVector inputs (thanks @cscherrer)

# 0.3.0

- Type stability and inference fixes
- Allow empty Tuple and NamedTuple aggregators
- Fix element type calculations, especially for AD via Flux
- Argument checks for inverse of ScaledShiftedLogistic (thanks @andreasnoack)

# 0.2.0

Sorry, no changelog before this.
