var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Introduction-1",
    "page": "Introduction",
    "title": "Introduction",
    "category": "section",
    "text": "import Random\nRandom.seed!(1)Some problems, especially in numerical integration and Markov Chain Monte Carlo, benefit from transformation of variables: for example, if σ  0 is a standard deviation parameter, it is usually better to work with log(σ) which can take any value on the real line. However, in general such transformations require correcting density functions by the determinant of their Jacobian matrix, usually referred to as \"the Jacobian\".Also, is usually easier to code MCMC algorithms to work with vectors of real numbers, which may represent a \"flattened\" version of parameters, and would need to be decomposed into individual parameters, which themselves may be arrays, tuples, or special objects like lower triangular matrices.This package is designed to help with both of these use cases. For example, consider the \"8 schools\" problem from Chapter 5.5 of Gelman et al (2013), in which SAT scores y_ij in J=8 schools are modeled using a conditional normaly_ij  N(θⱼ σ²)and the θⱼ are assume to have a hierarchical prior distributionθⱼ  N(μ τ²)For this problem, one could define a transformationusing TransformVariables\nt = as((μ = asℝ, σ = asℝ₊, τ = asℝ₊, θs = as(Array, 8)))\ndimension(t)which would then yield a NamedTuple with the given names, with one of them being a Vector:x = randn(dimension(t))\ny = transform(t, x)\nkeys(y)\ny.θsFurther worked examples of using this package can be found in the DynamicHMCExamples.jl repository. It is recommended that the user reads those first, and treats this documentation as a reference."
},

{
    "location": "index.html#TransformVariables.as",
    "page": "Introduction",
    "title": "TransformVariables.as",
    "category": "function",
    "text": "as(T, args...)\n\nShorthand for constructing transformations with image in T. args determines or modifies behavior, details depend on T.\n\nNot all transformations have an as method, some just have direct constructors. See methods(as) for a list.\n\nExamples\n\nas(Real, -∞, 1)          # transform a real number to (-∞, 1)\nas(Array, 10, 2)         # reshape 20 real numbers to a 10x2 matrix\nas((a = asℝ₊, b = as𝕀)) # transform 2 real numbers a NamedTuple, with a > 0, 0 < b < 1\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.dimension",
    "page": "Introduction",
    "title": "TransformVariables.dimension",
    "category": "function",
    "text": "dimension(t::AbstractTransform)\n\nThe dimension (number of elements) that t transforms.\n\nTypes should implement this method.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.transform",
    "page": "Introduction",
    "title": "TransformVariables.transform",
    "category": "function",
    "text": "transform(t, x)\n\n\nTransform x using t.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.transform_and_logjac",
    "page": "Introduction",
    "title": "TransformVariables.transform_and_logjac",
    "category": "function",
    "text": "transform_and_logjac(t, x)\n\n\nTransform x using t; calculating the log Jacobian determinant, returned as the second value.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.inverse",
    "page": "Introduction",
    "title": "TransformVariables.inverse",
    "category": "function",
    "text": "inverse(t::AbstractTransform, y)\n\nReturn x so that transform(t, x) ≈ y.\n\ninverse(t)\n\n\nReturn a callable equivalen to y -> inverse(t, y).\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.inverse!",
    "page": "Introduction",
    "title": "TransformVariables.inverse!",
    "category": "function",
    "text": "inverse!(x, t::AbstractTransform, y)\n\nPut inverse(t, y) into a preallocated vector x, returning x.\n\nGeneralized indexing should be assumed on x.\n\nSee inverse_eltype for determining the type of x.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.inverse_eltype",
    "page": "Introduction",
    "title": "TransformVariables.inverse_eltype",
    "category": "function",
    "text": "inverse_eltype(t::AbstractTransform, y)\n\nThe element type for vector x so that inverse!(x, t, y) works.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.transform_logdensity",
    "page": "Introduction",
    "title": "TransformVariables.transform_logdensity",
    "category": "function",
    "text": "transform_logdensity(t, f, x)\n\n\nLet y = t(x), and f(y) a log density at y. This function evaluates f ∘ t as a log density, taking care of the log Jacobian correction.\n\n\n\n\n\n"
},

{
    "location": "index.html#General-interface-1",
    "page": "Introduction",
    "title": "General interface",
    "category": "section",
    "text": "asdimension\ntransform\ntransform_and_logjacinverse\ninverse!\ninverse_eltypetransform_logdensity"
},

{
    "location": "index.html#Specific-transformations-1",
    "page": "Introduction",
    "title": "Specific transformations",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#TransformVariables.∞",
    "page": "Introduction",
    "title": "TransformVariables.∞",
    "category": "constant",
    "text": "Placeholder representing of infinity for specifing interval boundaries. Supports the - operator, ie -∞.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.asℝ",
    "page": "Introduction",
    "title": "TransformVariables.asℝ",
    "category": "constant",
    "text": "Transform to the real line (identity).\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.asℝ₊",
    "page": "Introduction",
    "title": "TransformVariables.asℝ₊",
    "category": "constant",
    "text": "Transform to a non-negative real number.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.asℝ₋",
    "page": "Introduction",
    "title": "TransformVariables.asℝ₋",
    "category": "constant",
    "text": "Transform to a non-positive real number.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.as𝕀",
    "page": "Introduction",
    "title": "TransformVariables.as𝕀",
    "category": "constant",
    "text": "Transform to the unit interval (0, 1).\n\n\n\n\n\n"
},

{
    "location": "index.html#Scalar-transforms-1",
    "page": "Introduction",
    "title": "Scalar transforms",
    "category": "section",
    "text": "∞asℝ\nasℝ₊\nasℝ₋\nas𝕀"
},

{
    "location": "index.html#TransformVariables.UnitVector",
    "page": "Introduction",
    "title": "TransformVariables.UnitVector",
    "category": "type",
    "text": "UnitVector(n)\n\nTransform n-1 real numbers to a unit vector of length n, under the Euclidean norm.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.CorrCholeskyFactor",
    "page": "Introduction",
    "title": "TransformVariables.CorrCholeskyFactor",
    "category": "type",
    "text": "CorrCholeskyFactor(n)\n\nCholesky factor of a correlation matrix of size n.\n\nTransforms n(n-1)2 real numbers to an nn upper-triangular matrix Ω, such that Ω\'*Ω is a correlation matrix (positive definite, with unit diagonal).\n\n\n\n\n\n"
},

{
    "location": "index.html#Special-arrays-1",
    "page": "Introduction",
    "title": "Special arrays",
    "category": "section",
    "text": "UnitVector\nCorrCholeskyFactor"
},

{
    "location": "index.html#Aggregation-of-transformations-1",
    "page": "Introduction",
    "title": "Aggregation of transformations",
    "category": "section",
    "text": "FIXME explain as syntax"
},

{
    "location": "index.html#TransformVariables.logjac_forwarddiff",
    "page": "Introduction",
    "title": "TransformVariables.logjac_forwarddiff",
    "category": "function",
    "text": "logjac_forwarddiff(f, x; handleNaN)\n\n\nCalculate the log Jacobian determinant of f at x using `ForwardDiff.\n\nNote\n\nf should be a bijection, mapping from vectors of real numbers to vectors of equal length.\n\nWhen handleNaN = true (the default), NaN log Jacobians are converted to -Inf.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.value_and_logjac_forwarddiff",
    "page": "Introduction",
    "title": "TransformVariables.value_and_logjac_forwarddiff",
    "category": "function",
    "text": "value_and_logjac_forwarddiff(f, x; flatten, handleNaN)\n\n\nCalculate the value and the log Jacobian determinant of f at x. flatten is used to get a vector out of the result that makes f a bijection.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.CustomTransform",
    "page": "Introduction",
    "title": "TransformVariables.CustomTransform",
    "category": "type",
    "text": "CustomTransform(g, f, flatten)\n\nWrap a custom transform y = f(transform(g, x))in a type that calculates the log Jacobian of∂y/∂xusingForwardDiff` when necessary.\n\nUsually, g::TransformReals, but when an integer is used, it amounts to the identity transformation with that dimension.\n\nflatten should take the result from f, and return a flat vector with no redundant elements, so that x  y is a bijection. For example, for a covariance matrix the elements below the diagonal should be removed.\n\n\n\n\n\n"
},

{
    "location": "index.html#Defining-custom-transformations-1",
    "page": "Introduction",
    "title": "Defining custom transformations",
    "category": "section",
    "text": "logjac_forwarddiff\nvalue_and_logjac_forwarddiff\nCustomTransform"
},

{
    "location": "index.html#Internals-1",
    "page": "Introduction",
    "title": "Internals",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Types-for-various-transformations-1",
    "page": "Introduction",
    "title": "Types for various transformations",
    "category": "section",
    "text": "These are not part of the API, use the as constructor or one of the predefined constants."
},

{
    "location": "index.html#TransformVariables.Identity",
    "page": "Introduction",
    "title": "TransformVariables.Identity",
    "category": "type",
    "text": "struct Identity <: TransformVariables.ScalarTransform\n\nIdentity x  x.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.ScaledShiftedLogistic",
    "page": "Introduction",
    "title": "TransformVariables.ScaledShiftedLogistic",
    "category": "type",
    "text": "struct ScaledShiftedLogistic{T<:Real} <: TransformVariables.ScalarTransform\n\nMaps to (scale, shift + scale) using x ↦ logistic(x)*scale + shift.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.ShiftedExp",
    "page": "Introduction",
    "title": "TransformVariables.ShiftedExp",
    "category": "type",
    "text": "struct ShiftedExp{D, T<:Real} <: TransformVariables.ScalarTransform\n\nShifted exponential. When D::Bool == true, maps to (shift, ∞) using x ↦ shift + eˣ, otherwise to (-∞, shift) using x ↦ shift - eˣ.\n\n\n\n\n\n"
},

{
    "location": "index.html#Scalar-transformations-1",
    "page": "Introduction",
    "title": "Scalar transformations",
    "category": "section",
    "text": "TransformVariables.Identity\nTransformVariables.ScaledShiftedLogistic\nTransformVariables.ShiftedExp"
},

{
    "location": "index.html#TransformVariables.ArrayTransform",
    "page": "Introduction",
    "title": "TransformVariables.ArrayTransform",
    "category": "type",
    "text": "struct ArrayTransform{T<:TransformVariables.AbstractTransform, M} <: TransformVariables.VectorTransform\n\nApply transformation repeatedly to create an array with given dims.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.TransformTuple",
    "page": "Introduction",
    "title": "TransformVariables.TransformTuple",
    "category": "type",
    "text": "struct TransformTuple{K, T<:Tuple{Vararg{TransformVariables.AbstractTransform,K}}} <: TransformVariables.VectorTransform\n\nTransform consecutive groups of real numbers to a tuple, using the given transformations.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.TransformNamedTuple",
    "page": "Introduction",
    "title": "TransformVariables.TransformNamedTuple",
    "category": "type",
    "text": "struct TransformNamedTuple{names, T<:(Tuple{Vararg{TransformVariables.AbstractTransform,N}} where N)} <: TransformVariables.VectorTransform\n\nTransform consecutive groups of real numbers to a named tuple, using the given transformations.\n\n\n\n\n\n"
},

{
    "location": "index.html#Aggregating-transformations-1",
    "page": "Introduction",
    "title": "Aggregating transformations",
    "category": "section",
    "text": "TransformVariables.ArrayTransform\nTransformVariables.TransformTuple\nTransformVariables.TransformNamedTuple"
},

{
    "location": "index.html#TransformVariables.InverseTransform",
    "page": "Introduction",
    "title": "TransformVariables.InverseTransform",
    "category": "type",
    "text": "struct InverseTransform{T}\n\nInverse of the wrapped transform. Use the 1-argument version of inverse to construct.\n\n\n\n\n\n"
},

{
    "location": "index.html#Wrapper-for-inverse-1",
    "page": "Introduction",
    "title": "Wrapper for inverse",
    "category": "section",
    "text": "TransformVariables.InverseTransform"
},

{
    "location": "index.html#TransformVariables.RealVector",
    "page": "Introduction",
    "title": "TransformVariables.RealVector",
    "category": "type",
    "text": "An AbstractVector of <:Real elements.\n\nUsed internally as a type for transformations from vectors.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.AbstractTransform",
    "page": "Introduction",
    "title": "TransformVariables.AbstractTransform",
    "category": "type",
    "text": "abstract type AbstractTransform\n\nSupertype for all transformations in this package.\n\nInterface\n\nThe user interface consists of\n\ndimension\ntransform\ntransform_and_logjac\n[inverse]@(ref), inverse!\ninverse_eltype.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.ScalarTransform",
    "page": "Introduction",
    "title": "TransformVariables.ScalarTransform",
    "category": "type",
    "text": "abstract type ScalarTransform <: TransformVariables.AbstractTransform\n\nTransform a scalar (real number) to another scalar.\n\nSubtypes mustdefine transform, transform_and_logjac, and inverse; other methods of of the interface should have the right defaults.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.VectorTransform",
    "page": "Introduction",
    "title": "TransformVariables.VectorTransform",
    "category": "type",
    "text": "abstract type VectorTransform <: TransformVariables.AbstractTransform\n\nTransformation that transforms <: RealVectors to other values.\n\nImplementation\n\nImplements transform and transform_and_logjac via transform_with, and inverse via inverse!.\n\n\n\n\n\n"
},

{
    "location": "index.html#Types-and-type-aliases-1",
    "page": "Introduction",
    "title": "Types and type aliases",
    "category": "section",
    "text": "TransformVariables.RealVector\nTransformVariables.AbstractTransform\nTransformVariables.ScalarTransform\nTransformVariables.VectorTransform"
},

{
    "location": "index.html#TransformVariables.LogJacFlag",
    "page": "Introduction",
    "title": "TransformVariables.LogJacFlag",
    "category": "type",
    "text": "abstract type LogJacFlag\n\nFlag used internally by the implementation of transformations, as explained below.\n\nWhen calculating the log jacobian determinant for a matrix, initialize with\n\nlogjac_zero(flag, x)\n\nand then accumulate with log jacobians as needed with +.\n\nWhen flag is LogJac, methods should return the log Jacobian as the second argument, otherwise NoLogJac, which simply combines to itself with +, serving as an empty placeholder. This allows methods to share code of the two implementations.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.LogJac",
    "page": "Introduction",
    "title": "TransformVariables.LogJac",
    "category": "type",
    "text": "Calculate log Jacobian as the second value.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.NoLogJac",
    "page": "Introduction",
    "title": "TransformVariables.NoLogJac",
    "category": "type",
    "text": "Don\'t calculate log Jacobian, return NOLOGJAC as the second value.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.logjac_zero",
    "page": "Introduction",
    "title": "TransformVariables.logjac_zero",
    "category": "function",
    "text": "logjac_zero(?, T)\n\n\nInitial value for log Jacobian calculations.\n\n\n\n\n\n"
},

{
    "location": "index.html#Conditional-calculation-of-log-Jacobian-determinant-1",
    "page": "Introduction",
    "title": "Conditional calculation of log Jacobian determinant",
    "category": "section",
    "text": "TransformVariables.LogJacFlag\nTransformVariables.LogJac\nTransformVariables.NoLogJac\nTransformVariables.logjac_zero"
},

{
    "location": "index.html#TransformVariables.@calltrans",
    "page": "Introduction",
    "title": "TransformVariables.@calltrans",
    "category": "macro",
    "text": "Workaround for https://github.com/JuliaLang/julia/issues/14919 to make transformation types callable.\n\nTODO: remove when this issue is closed, also possibly remove MacroTools as a dependency if not used elsewhere.\n\n\n\n\n\n"
},

{
    "location": "index.html#Macros-1",
    "page": "Introduction",
    "title": "Macros",
    "category": "section",
    "text": "TransformVariables.@calltrans"
},

{
    "location": "index.html#Fast-vector-views-1",
    "page": "Introduction",
    "title": "Fast vector views",
    "category": "section",
    "text": "TransformVariables.index_into\nTransformVariables.IndexInto"
},

{
    "location": "index.html#TransformVariables.transform_with",
    "page": "Introduction",
    "title": "TransformVariables.transform_with",
    "category": "function",
    "text": "transform_with(flag::LogJacFlag, t::AbstractTransform, x::RealVector)\n\nTransform elements of x, starting using transformation.\n\nThe first value returned is the transformed value, the second the log Jacobian determinant or a placeholder, depending on flag.\n\nIn contrast to [transform] and [transform_and_logjac], this method always assumes that x is a RealVector, for efficient traversal. Some types implement the latter two via this method.\n\nImplementations should assume generalized indexing on x.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables._transform_tuple",
    "page": "Introduction",
    "title": "TransformVariables._transform_tuple",
    "category": "function",
    "text": "_transform_tuple(flag, tt, x)\n\n\nHelper function for transforming tuples. Used internally.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables._inverse!_tuple",
    "page": "Introduction",
    "title": "TransformVariables._inverse!_tuple",
    "category": "function",
    "text": "_inverse!_tuple(x, ts, ys)\n\n\nHelper function for inverting tuples of transformations. Used internally.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables._inverse_eltype_tuple",
    "page": "Introduction",
    "title": "TransformVariables._inverse_eltype_tuple",
    "category": "function",
    "text": "_inverse_eltype_tuple(ts, ys)\n\n\nHelper function determining element type of inverses from tuples. Used internally.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.unit_triangular_dimension",
    "page": "Introduction",
    "title": "TransformVariables.unit_triangular_dimension",
    "category": "function",
    "text": "unit_triangular_dimension(n)\n\n\nNumber of elements (strictly) above the diagonal in an nn matrix.\n\n\n\n\n\n"
},

{
    "location": "index.html#Helper-functions-1",
    "page": "Introduction",
    "title": "Helper functions",
    "category": "section",
    "text": "TransformVariables.transform_with\nTransformVariables._transform_tuple\nTransformVariables._inverse!_tuple\nTransformVariables._inverse_eltype_tuple\nTransformVariables.unit_triangular_dimension"
},

{
    "location": "index.html#TransformVariables.l2_remainder_transform",
    "page": "Introduction",
    "title": "TransformVariables.l2_remainder_transform",
    "category": "function",
    "text": "(y, r, ℓ) =\n\nl2_remainder_transform(flag, x, r)\n\n\nGiven x  ℝ and 0  r  1, return (y, r′) such that\n\ny² + r² = r²,\ny y  r is mapped with a bijection from x.\n\nℓ is the log Jacobian (whether it is evaluated depends on flag).\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.l2_remainder_inverse",
    "page": "Introduction",
    "title": "TransformVariables.l2_remainder_inverse",
    "category": "function",
    "text": "(x, r′) =\n\nl2_remainder_inverse(y, r)\n\n\nInverse of l2_remainder_transform in x and y.\n\n\n\n\n\n"
},

{
    "location": "index.html#Building-blocks-for-transformations-1",
    "page": "Introduction",
    "title": "Building blocks for transformations",
    "category": "section",
    "text": "TransformVariables.l2_remainder_transform\nTransformVariables.l2_remainder_inverse"
},

]}
