# TransformVariables

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/tpapp/TransformVariables.jl.svg?branch=master)](https://travis-ci.org/tpapp/TransformVariables.jl)
[![Coverage Status](https://coveralls.io/repos/tpapp/TransformVariables.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tpapp/TransformVariables.jl?branch=master)
[![codecov.io](http://codecov.io/github/tpapp/TransformVariables.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/TransformVariables.jl?branch=master)
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
