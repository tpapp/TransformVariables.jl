var documenterSearchIndex = {"docs":
[{"location":"#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"import Random\nRandom.seed!(1)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Some problems, especially in numerical integration and Markov Chain Monte Carlo, benefit from transformation of variables: for example, if σ  0 is a standard deviation parameter, it is usually better to work with log(σ) which can take any value on the real line. However, in general such transformations require correcting density functions by the determinant of their Jacobian matrix, usually referred to as \"the Jacobian\".","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Also, is usually easier to code MCMC algorithms to work with vectors of real numbers, which may represent a \"flattened\" version of parameters, and would need to be decomposed into individual parameters, which themselves may be arrays, tuples, or special objects like lower triangular matrices.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"This package is designed to help with both of these use cases. For example, consider the \"8 schools\" problem from Chapter 5.5 of Gelman et al (2013), in which SAT scores y_ij in J=8 schools are modeled using a conditional normal","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"y_ij  N(θⱼ σ²)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"and the θⱼ are assume to have a hierarchical prior distribution","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"θⱼ  N(μ τ²)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"For this problem, one could define a transformation","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using TransformVariables\nt = as((μ = asℝ, σ = asℝ₊, τ = asℝ₊, θs = as(Array, 8)))\ndimension(t)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"which would then yield a NamedTuple with the given names, with one of them being a Vector:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"x = randn(dimension(t))\ny = transform(t, x)\nkeys(y)\ny.θs","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Further worked examples of using this package can be found in the DynamicHMCExamples.jl repository. It is recommended that the user reads those first, and treats this documentation as a reference.","category":"page"},{"location":"#General-interface","page":"Introduction","title":"General interface","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"dimension\ntransform\ntransform_and_logjac","category":"page"},{"location":"#TransformVariables.dimension","page":"Introduction","title":"TransformVariables.dimension","text":"dimension(t::AbstractTransform)\n\nThe dimension (number of elements) that t transforms.\n\nTypes should implement this method.\n\n\n\n\n\n","category":"function"},{"location":"#TransformVariables.transform","page":"Introduction","title":"TransformVariables.transform","text":"transform(t, x)\n\nTransform x using t.\n\ntransform(t).\n\nReturn a callable equivalent to x -> transform(t, x) that transforms its argument:\n\ntransform(t, x) == transform(t)(x)\n\n\n\n\n\n","category":"function"},{"location":"#TransformVariables.transform_and_logjac","page":"Introduction","title":"TransformVariables.transform_and_logjac","text":"transform_and_logjac(t, x)\n\n\nTransform x using t; calculating the log Jacobian determinant, returned as the second value.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Introduction","title":"Introduction","text":"inverse\ninverse!\ninverse_eltype","category":"page"},{"location":"#TransformVariables.inverse","page":"Introduction","title":"TransformVariables.inverse","text":"inverse(t, y)\n\nReturn x so that transform(t, x) ≈ y.\n\ninverse(t)\n\nReturn a callable equivalent to y -> inverse(t, y). t can also be a callable created with transform, so the following holds:\n\ninverse(t)(y) == inverse(t, y) == inverse(transform(t))(y)\n\n\n\n\n\n","category":"function"},{"location":"#TransformVariables.inverse!","page":"Introduction","title":"TransformVariables.inverse!","text":"inverse!(x, transformation, y)\n\n\nPut inverse(t, y) into a preallocated vector x, returning x.\n\nGeneralized indexing should be assumed on x.\n\nSee inverse_eltype for determining the type of x.\n\n\n\n\n\n","category":"function"},{"location":"#TransformVariables.inverse_eltype","page":"Introduction","title":"TransformVariables.inverse_eltype","text":"inverse_eltype(t::AbstractTransform, y)\n\nThe element type for vector x so that inverse!(x, t, y) works.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Introduction","title":"Introduction","text":"transform_logdensity","category":"page"},{"location":"#TransformVariables.transform_logdensity","page":"Introduction","title":"TransformVariables.transform_logdensity","text":"transform_logdensity(t, f, x)\n\n\nLet y = t(x), and f(y) a log density at y. This function evaluates f ∘ t as a log density, taking care of the log Jacobian correction.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Introduction","title":"Introduction","text":"domain_label","category":"page"},{"location":"#TransformVariables.domain_label","page":"Introduction","title":"TransformVariables.domain_label","text":"domain_label(transformation, index)\n\n\nReturn a string that can be used to for identifying a coordinate. Mainly for debugging and generating graphs and data summaries.\n\nTransformations may provide a heuristic label.\n\nTransformations should implement _domain_label.\n\nExample\n\njulia> t = as((a = asℝ₊,\n            b = as(Array, asℝ₋, 1, 1),\n            c = corr_cholesky_factor(2)));\n\njulia> [domain_label(t, i) for i in 1:dimension(t)]\n3-element Vector{String}:\n \".a\"\n \".b[1,1]\"\n \".c[1]\"\n\n\n\n\n\n","category":"function"},{"location":"#Defining-transformations","page":"Introduction","title":"Defining transformations","text":"","category":"section"},{"location":"#The-as-constructor-and-aggregations","page":"Introduction","title":"The as constructor and aggregations","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Some transformations, particularly aggregations use the function as as the constructor. Aggregating transformations are built from other transformations to transform consecutive (blocks of) real numbers into the desired domain.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"It is recommended that you use as(Array, ...) and friends (as(Vector, ...), as(Matrix, ...)) for repeating the same transformation, and named tuples such as as((μ = ..., σ = ...)) for transforming into named parameters. For extracting parameters in log likelihoods, consider Parameters.jl.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"See methods(as) for all the constructors, ?as for their documentation.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"as","category":"page"},{"location":"#TransformVariables.as","page":"Introduction","title":"TransformVariables.as","text":"as(T, args...)\n\nShorthand for constructing transformations with image in T. args determines or modifies behavior, details depend on T.\n\nNot all transformations have an as method, some just have direct constructors. See methods(as) for a list.\n\nExamples\n\nas(Real, -∞, 1)          # transform a real number to (-∞, 1)\nas(Array, 10, 2)         # reshape 20 real numbers to a 10x2 matrix\nas(Array, as𝕀, 10)       # transform 10 real numbers to (0, 1)\nas((a = asℝ₊, b = as𝕀)) # transform 2 real numbers a NamedTuple, with a > 0, 0 < b < 1\nas(SArray{1,2,3}, as𝕀)  # transform to a static array of positive numbers\n\n\n\n\n\n","category":"function"},{"location":"#Scalar-transforms","page":"Introduction","title":"Scalar transforms","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The symbol ∞ is a placeholder for infinity. It does not correspond to Inf, but acts as a placeholder for the correct dispatch. -∞ is valid.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"∞","category":"page"},{"location":"#TransformVariables.∞","page":"Introduction","title":"TransformVariables.∞","text":"Placeholder representing of infinity for specifing interval boundaries. Supports the - operator, ie -∞.\n\n\n\n\n\n","category":"constant"},{"location":"","page":"Introduction","title":"Introduction","text":"as(Real, a, b) defines transformations to finite and (semi-)infinite subsets of the real line, where a and b can be -∞ and ∞, respectively.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"as(::Type{Real}, Any, Any)","category":"page"},{"location":"#TransformVariables.as-Tuple{Type{Real}, Any, Any}","page":"Introduction","title":"TransformVariables.as","text":"as(Real, left, right)\n\nReturn a transformation that transforms a single real number to the given (open) interval.\n\nleft < right is required, but may be -∞ or ∞, respectively, in which case the appropriate transformation is selected. See ∞.\n\nSome common transformations are predefined as constants, see asℝ, asℝ₋, asℝ₊, as𝕀.\n\nnote: Note\nThe finite arguments are promoted to a common type and affect promotion. Eg transform(as(0, ∞), 0f0) isa Float32, but transform(as(0.0, ∞), 0f0) isa Float64.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Introduction","title":"Introduction","text":"The following constants are defined for common cases.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"asℝ\nasℝ₊\nasℝ₋\nas𝕀","category":"page"},{"location":"#TransformVariables.asℝ","page":"Introduction","title":"TransformVariables.asℝ","text":"Transform to the real line (identity). See as.\n\nasℝ and as_real are equivalent alternatives.\n\n\n\n\n\n","category":"constant"},{"location":"#TransformVariables.asℝ₊","page":"Introduction","title":"TransformVariables.asℝ₊","text":"Transform to a positive real number. See as.\n\nasℝ₊ and as_positive_real are equivalent alternatives.\n\n\n\n\n\n","category":"constant"},{"location":"#TransformVariables.asℝ₋","page":"Introduction","title":"TransformVariables.asℝ₋","text":"Transform to a negative real number. See as.\n\nasℝ₋ and as_negative_real are equivalent alternatives.\n\n\n\n\n\n","category":"constant"},{"location":"#TransformVariables.as𝕀","page":"Introduction","title":"TransformVariables.as𝕀","text":"Transform to the unit interval (0, 1). See as.\n\nas𝕀 and as_unit_interval are equivalent alternatives.\n\n\n\n\n\n","category":"constant"},{"location":"#Special-arrays","page":"Introduction","title":"Special arrays","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"UnitVector\nUnitSimplex\nCorrCholeskyFactor\ncorr_cholesky_factor","category":"page"},{"location":"#TransformVariables.UnitVector","page":"Introduction","title":"TransformVariables.UnitVector","text":"UnitVector(n)\n\nTransform n-1 real numbers to a unit vector of length n, under the Euclidean norm.\n\n\n\n\n\n","category":"type"},{"location":"#TransformVariables.UnitSimplex","page":"Introduction","title":"TransformVariables.UnitSimplex","text":"UnitSimplex(n)\n\nTransform n-1 real numbers to a vector of length n whose elements are non-negative and sum to one.\n\n\n\n\n\n","category":"type"},{"location":"#TransformVariables.CorrCholeskyFactor","page":"Introduction","title":"TransformVariables.CorrCholeskyFactor","text":"CorrCholeskyFactor(n)\n\nnote: Note\nIt is better style to use corr_cholesky_factor, this will be deprecated.\n\nCholesky factor of a correlation matrix of size n.\n\nTransforms n(n-1)2 real numbers to an nn upper-triangular matrix U, such that U'*U is a correlation matrix (positive definite, with unit diagonal).\n\nNotes\n\nIf\n\nz is a vector of n IID standard normal variates,\nσ is an n-element vector of standard deviations,\nU is obtained from CorrCholeskyFactor(n),\n\nthen Diagonal(σ) * U' * z will be a multivariate normal with the given variances and correlation matrix U' * U.\n\n\n\n\n\n","category":"type"},{"location":"#TransformVariables.corr_cholesky_factor","page":"Introduction","title":"TransformVariables.corr_cholesky_factor","text":"corr_cholesky_factor(n)\n\n\nTransform into a Cholesky factor of a correlation matrix.\n\nIf the argument is a (positive) integer n, it determines the size of the output n × n, resulting in a Matrix.\n\nIf the argument is SMatrix{N,N}, an SMatrix is produced.\n\n\n\n\n\n","category":"function"},{"location":"#Miscellaneous-transformations","page":"Introduction","title":"Miscellaneous transformations","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Constant","category":"page"},{"location":"#TransformVariables.Constant","page":"Introduction","title":"TransformVariables.Constant","text":"Contant(value)\n\nPlaceholder for inserting a constant. Inverse checks equality with ==.\n\n\n\n\n\n","category":"type"},{"location":"#Defining-custom-transformations","page":"Introduction","title":"Defining custom transformations","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"logjac_forwarddiff\nvalue_and_logjac_forwarddiff\nCustomTransform","category":"page"},{"location":"#TransformVariables.logjac_forwarddiff","page":"Introduction","title":"TransformVariables.logjac_forwarddiff","text":"logjac_forwarddiff(f, x; handleNaN, chunk, cfg)\n\n\nCalculate the log Jacobian determinant of f at x using `ForwardDiff.\n\nNote\n\nf should be a bijection, mapping from vectors of real numbers to vectors of equal length.\n\nWhen handleNaN = true (the default), NaN log Jacobians are converted to -Inf.\n\n\n\n\n\n","category":"function"},{"location":"#TransformVariables.value_and_logjac_forwarddiff","page":"Introduction","title":"TransformVariables.value_and_logjac_forwarddiff","text":"value_and_logjac_forwarddiff(\n    f,\n    x;\n    flatten,\n    handleNaN,\n    chunk,\n    cfg\n)\n\n\nCalculate the value and the log Jacobian determinant of f at x. flatten is used to get a vector out of the result that makes f a bijection.\n\n\n\n\n\n","category":"function"},{"location":"#TransformVariables.CustomTransform","page":"Introduction","title":"TransformVariables.CustomTransform","text":"CustomTransform(g, f, flatten; chunk, cfg)\n\n\nWrap a custom transform y = f(transform(g, x))in a type that calculates the log Jacobian of∂y/∂xusingForwardDiff` when necessary.\n\nUsually, g::TransformReals, but when an integer is used, it amounts to the identity transformation with that dimension.\n\nflatten should take the result from f, and return a flat vector with no redundant elements, so that x  y is a bijection. For example, for a covariance matrix the elements below the diagonal should be removed.\n\nchunk and cfg can be used to configure ForwardDiff.JacobianConfig. cfg is used directly, while chunk = ForwardDiff.Chunk{N}() can be used to obtain a type-stable configuration.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Internals","page":"Internals","title":"Internals","text":"","category":"section"},{"location":"internals/#Types-for-various-transformations","page":"Internals","title":"Types for various transformations","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"These are not part of the API, use the as constructor or one of the predefined constants.","category":"page"},{"location":"internals/#Scalar-transformations","page":"Internals","title":"Scalar transformations","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"TransformVariables.Identity\nTransformVariables.ScaledShiftedLogistic\nTransformVariables.ShiftedExp","category":"page"},{"location":"internals/#TransformVariables.Identity","page":"Internals","title":"TransformVariables.Identity","text":"struct Identity <: TransformVariables.ScalarTransform\n\nIdentity x  x.\n\n\n\n\n\n","category":"type"},{"location":"internals/#TransformVariables.ScaledShiftedLogistic","page":"Internals","title":"TransformVariables.ScaledShiftedLogistic","text":"struct ScaledShiftedLogistic{T<:Real} <: TransformVariables.ScalarTransform\n\nMaps to (scale, shift + scale) using logistic(x) * scale + shift.\n\n\n\n\n\n","category":"type"},{"location":"internals/#TransformVariables.ShiftedExp","page":"Internals","title":"TransformVariables.ShiftedExp","text":"struct ShiftedExp{D, T<:Real} <: TransformVariables.ScalarTransform\n\nShifted exponential. When D::Bool == true, maps to (shift, ∞) using x ↦ shift + eˣ, otherwise to (-∞, shift) using x ↦ shift - eˣ.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Aggregating-transformations","page":"Internals","title":"Aggregating transformations","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"TransformVariables.ArrayTransformation\nTransformVariables.TransformTuple","category":"page"},{"location":"internals/#TransformVariables.ArrayTransformation","page":"Internals","title":"TransformVariables.ArrayTransformation","text":"struct ArrayTransformation{T<:TransformVariables.AbstractTransform, M} <: TransformVariables.VectorTransform\n\nApply transformation repeatedly to create an array with given dims.\n\n\n\n\n\n","category":"type"},{"location":"internals/#TransformVariables.TransformTuple","page":"Internals","title":"TransformVariables.TransformTuple","text":"struct TransformTuple{T} <: TransformVariables.VectorTransform\n\nTransform consecutive groups of real numbers to a tuple, using the given transformations.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Wrapper-for-inverse","page":"Internals","title":"Wrapper for inverse","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"TransformVariables.CallableInverse","category":"page"},{"location":"internals/#TransformVariables.CallableInverse","page":"Internals","title":"TransformVariables.CallableInverse","text":"struct Fix1{typeof(inverse), T<:TransformVariables.AbstractTransform} <: Function\n\nPartial application of inverse(t, y), callable with y. Use inverse(t) to construct.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Types-and-type-aliases","page":"Internals","title":"Types and type aliases","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"TransformVariables.AbstractTransform\nTransformVariables.ScalarTransform\nTransformVariables.VectorTransform","category":"page"},{"location":"internals/#TransformVariables.AbstractTransform","page":"Internals","title":"TransformVariables.AbstractTransform","text":"abstract type AbstractTransform\n\nSupertype for all transformations in this package.\n\nInterface\n\nThe user interface consists of\n\ndimension\ntransform\ntransform_and_logjac\n[inverse]@(ref), inverse!\ninverse_eltype.\n\n\n\n\n\n","category":"type"},{"location":"internals/#TransformVariables.ScalarTransform","page":"Internals","title":"TransformVariables.ScalarTransform","text":"abstract type ScalarTransform <: TransformVariables.AbstractTransform\n\nTransform a scalar (real number) to another scalar.\n\nSubtypes must define transform, transform_and_logjac, and inverse. Other methods of of the interface should have the right defaults.\n\nnote: Note\nThis type is for code organization within the package, and is not part of the public API.\n\n\n\n\n\n","category":"type"},{"location":"internals/#TransformVariables.VectorTransform","page":"Internals","title":"TransformVariables.VectorTransform","text":"abstract type VectorTransform <: TransformVariables.AbstractTransform\n\nTransformation that transforms <: AbstractVectors to other values.\n\nImplementation\n\nImplements transform and transform_and_logjac via transform_with, and inverse via inverse!.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Conditional-calculation-of-log-Jacobian-determinant","page":"Internals","title":"Conditional calculation of log Jacobian determinant","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"TransformVariables.LogJacFlag\nTransformVariables.LogJac\nTransformVariables.NoLogJac\nTransformVariables.logjac_zero","category":"page"},{"location":"internals/#TransformVariables.LogJacFlag","page":"Internals","title":"TransformVariables.LogJacFlag","text":"abstract type LogJacFlag\n\nFlag used internally by the implementation of transformations, as explained below.\n\nWhen calculating the log jacobian determinant for a matrix, initialize with\n\nlogjac_zero(flag, x)\n\nand then accumulate with log jacobians as needed with +.\n\nWhen flag is LogJac, methods should return the log Jacobian as the second argument, otherwise NoLogJac, which simply combines to itself with +, serving as an empty placeholder. This allows methods to share code of the two implementations.\n\n\n\n\n\n","category":"type"},{"location":"internals/#TransformVariables.LogJac","page":"Internals","title":"TransformVariables.LogJac","text":"Calculate log Jacobian as the second value.\n\n\n\n\n\n","category":"type"},{"location":"internals/#TransformVariables.NoLogJac","page":"Internals","title":"TransformVariables.NoLogJac","text":"Don't calculate log Jacobian, return NOLOGJAC as the second value.\n\n\n\n\n\n","category":"type"},{"location":"internals/#TransformVariables.logjac_zero","page":"Internals","title":"TransformVariables.logjac_zero","text":"logjac_zero(_, _)\n\n\nInitial value for log Jacobian calculations.\n\n\n\n\n\n","category":"function"},{"location":"internals/#Helper-functions","page":"Internals","title":"Helper functions","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"TransformVariables.transform_with\nTransformVariables._transform_tuple\nTransformVariables._inverse!_tuple\nTransformVariables._inverse_eltype_tuple\nTransformVariables.unit_triangular_dimension","category":"page"},{"location":"internals/#TransformVariables.transform_with","page":"Internals","title":"TransformVariables.transform_with","text":"transform_with(flag::LogJacFlag, transformation, x::AbstractVector, index)\n\nTransform elements of x from index, using transformation.\n\nReturn (y, logjac), index′, where\n\ny is the result of the transformation,\nlogjac is the the log Jacobian determinant or a placeholder, depending on flag,\nindex′ is the next index in x after the elements used for the transformation\n\nInternal function. Implementations\n\ncan assume that x has enough elements for transformation (ie @inbounds can be used),\nshould work with generalized indexing on x.\n\n\n\n\n\n","category":"function"},{"location":"internals/#TransformVariables._transform_tuple","page":"Internals","title":"TransformVariables._transform_tuple","text":"_transform_tuple(flag, x, index, _)\n\n\nHelper function for transforming tuples. Used internally, to help type inference. Use via transfom_tuple.\n\n\n\n\n\n","category":"function"},{"location":"internals/#TransformVariables._inverse!_tuple","page":"Internals","title":"TransformVariables._inverse!_tuple","text":"_inverse!_tuple(x, index, ts, ys)\n\n\nHelper function for inverting tuples of transformations. Used internally.\n\nPerforms no argument validation, caller should do this.\n\n\n\n\n\n","category":"function"},{"location":"internals/#TransformVariables._inverse_eltype_tuple","page":"Internals","title":"TransformVariables._inverse_eltype_tuple","text":"_inverse_eltype_tuple(ts, ys)\n\n\nHelper function determining element type of inverses from tuples. Used internally.\n\nPerforms no argument validation, caller should do this.\n\n\n\n\n\n","category":"function"},{"location":"internals/#TransformVariables.unit_triangular_dimension","page":"Internals","title":"TransformVariables.unit_triangular_dimension","text":"unit_triangular_dimension(n)\n\n\nNumber of elements (strictly) above the diagonal in an nn matrix.\n\n\n\n\n\n","category":"function"},{"location":"internals/#Building-blocks-for-transformations","page":"Internals","title":"Building blocks for transformations","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"TransformVariables.l2_remainder_transform\nTransformVariables.l2_remainder_inverse","category":"page"},{"location":"internals/#TransformVariables.l2_remainder_transform","page":"Internals","title":"TransformVariables.l2_remainder_transform","text":"(y, log_r, ℓ) =\n\nl2_remainder_transform(flag, x, log_r)\n\n\nGiven x  ℝ and 0  r  1, we define (y, r′) such that\n\ny² + (r)² = r²,\ny y  r is mapped with a bijection from x, with the sign depending on x,\n\nbut use log(r) for actual calculations so that large ys still give nonsingular results.\n\nℓ is the log Jacobian (whether it is evaluated depends on flag).\n\n\n\n\n\n","category":"function"},{"location":"internals/#TransformVariables.l2_remainder_inverse","page":"Internals","title":"TransformVariables.l2_remainder_inverse","text":"(x, r′) =\n\nl2_remainder_inverse(y, log_r)\n\n\nInverse of l2_remainder_transform in x and y.\n\n\n\n\n\n","category":"function"}]
}
