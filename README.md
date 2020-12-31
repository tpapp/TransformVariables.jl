# TransformVariables.jl

![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
[![build](https://github.com/tpapp/TransformVariables.jl/workflows/CI/badge.svg)](https://github.com/tpapp/TransformVariables.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/TransformVariables.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/TransformVariables.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/TransformVariables.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tpapp.github.io/TransformVariables.jl/dev)

Successor of [ContinuousTransformations.jl](https://github.com/tpapp/ContinuousTransformations.jl).

## Features

- Simple interface.
- Fast implementation, unrolling when it makes sense.
- Targeted to applications in statistics, mainly [ML](https://en.wikipedia.org/wiki/Maximum_likelihood), [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation), and Bayesian inference.
- Take advantage of Julia v0.7 features, especially named tuples and compiler optimizations.

## Caveat

- **Work in progress.** API will change rapidly, without warning.
- Expect speed regressions until API is finalized.
