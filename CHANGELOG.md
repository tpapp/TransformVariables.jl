# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `getindex`, `getproperty`, `length`, and support for [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl) API for tuple and named tuple transformations

### Changed
### Deprecated
### Removed
### Fixed
### Security

## Old changelogs (archive)

# 0.8.17

**Changelogs were not kept** between 0.3.11 and this version. PRs reconstructing them are welcome.

# 0.3.11

- remove dependency on MacroTools for abstract dispatch workaround, but require Julia 1.3
- deprecate t(x) interface

# 0.3.10

- compat bump

# 0.3.9

- follow up changes in for LogDensityProblems

# 0.3.8

- add UnitSimplex ([#56], thanks @scheidan)

# 0.3.7

- fix some tuple index bugs (cf [#58])

# 0.3.6

- Pretty print scalar transforms ([#54], thanks @tkf)

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
