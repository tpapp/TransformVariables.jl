var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Manual",
    "title": "Manual",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Introduction-1",
    "page": "Manual",
    "title": "Introduction",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#TransformVariables.dimension",
    "page": "Manual",
    "title": "TransformVariables.dimension",
    "category": "function",
    "text": "dimension(t::AbstractTransform)\n\nThe dimension (number of elements) that t transforms.\n\nTypes should implement this method.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.transform",
    "page": "Manual",
    "title": "TransformVariables.transform",
    "category": "function",
    "text": "transform(t, x)\n\n\nTransform x using t.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.transform_and_logjac",
    "page": "Manual",
    "title": "TransformVariables.transform_and_logjac",
    "category": "function",
    "text": "transform_and_logjac(t, x)\n\n\nTransform x using t; calculating the log Jacobian determinant, returned as the second value.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.inverse",
    "page": "Manual",
    "title": "TransformVariables.inverse",
    "category": "function",
    "text": "inverse(t::AbstractTransform, y)\n\nReturn x so that transform(t, x) ≈ y.\n\ninverse(t)\n\n\nReturn a callable equivalen to y -> inverse(t, y).\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.transform_logdensity",
    "page": "Manual",
    "title": "TransformVariables.transform_logdensity",
    "category": "function",
    "text": "transform_logdensity(t, f, x)\n\n\nLet y = t(x), and f(y) a log density at y. This function evaluates f ∘ t as a log density, taking care of the log Jacobian correction.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.as",
    "page": "Manual",
    "title": "TransformVariables.as",
    "category": "function",
    "text": "as(T, args...)\n\nShorthand for constructing transformations with image in T. args determines or modifies behavior, details depend on T.\n\nNot all transformations have an as method, some just have direct constructors. See methods(as) for a list.\n\nExamples\n\nas(Real, -∞, 1)     # transform a real number to (-∞, 1)\nas(Array, 10, 2)    # reshape 20 real numbers to a 10x2 matrix\nas((a = ℝ₊, b = 𝕀)) # transform 2 real numbers a NamedTuple, with a > 0, 0 < b < 1\n\n\n\n\n\n"
},

{
    "location": "index.html#General-interface-1",
    "page": "Manual",
    "title": "General interface",
    "category": "section",
    "text": "dimension\ntransform\ntransform_and_logjac\ninverse\ntransform_logdensity\nas"
},

{
    "location": "index.html#Specific-transformations-1",
    "page": "Manual",
    "title": "Specific transformations",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#TransformVariables.∞",
    "page": "Manual",
    "title": "TransformVariables.∞",
    "category": "constant",
    "text": "Placeholder representing of infinity for specifing interval boundaries. Supports the - operator, ie -∞.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.ℝ",
    "page": "Manual",
    "title": "TransformVariables.ℝ",
    "category": "constant",
    "text": "Transform to the real line (identity).\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.ℝ₊",
    "page": "Manual",
    "title": "TransformVariables.ℝ₊",
    "category": "constant",
    "text": "Transform to a non-negative real number.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.ℝ₋",
    "page": "Manual",
    "title": "TransformVariables.ℝ₋",
    "category": "constant",
    "text": "Transform to a non-positive real number.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.𝕀",
    "page": "Manual",
    "title": "TransformVariables.𝕀",
    "category": "constant",
    "text": "Transform to the unit interval (0, 1).\n\n\n\n\n\n"
},

{
    "location": "index.html#Scalar-transforms-1",
    "page": "Manual",
    "title": "Scalar transforms",
    "category": "section",
    "text": "∞ℝ\nℝ₊\nℝ₋\n𝕀"
},

{
    "location": "index.html#TransformVariables.UnitVector",
    "page": "Manual",
    "title": "TransformVariables.UnitVector",
    "category": "type",
    "text": "UnitVector(n)\n\nTransform n-1 real numbers to a unit vector of length n, under the Euclidean norm.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.CorrCholeskyFactor",
    "page": "Manual",
    "title": "TransformVariables.CorrCholeskyFactor",
    "category": "type",
    "text": "CorrCholeskyFactor(n)\n\nCholesky factor of a correlation matrix of size n.\n\n\n\n\n\n"
},

{
    "location": "index.html#Special-arrays-1",
    "page": "Manual",
    "title": "Special arrays",
    "category": "section",
    "text": "UnitVector\nCorrCholeskyFactor"
},

{
    "location": "index.html#Aggregation-of-transformations-1",
    "page": "Manual",
    "title": "Aggregation of transformations",
    "category": "section",
    "text": "FIXME explain as syntax"
},

{
    "location": "index.html#TransformVariables.logjac_forwarddiff",
    "page": "Manual",
    "title": "TransformVariables.logjac_forwarddiff",
    "category": "function",
    "text": "logjac_forwarddiff(f, x)\n\n\nCalculate the log Jacobian determinant of f at x using `ForwardDiff.\n\nNote\n\nf should be a bijection, mapping from vectors of real numbers to vectors of equal length.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.value_and_logjac_forwarddiff",
    "page": "Manual",
    "title": "TransformVariables.value_and_logjac_forwarddiff",
    "category": "function",
    "text": "value_and_logjac_forwarddiff(f, x)\nvalue_and_logjac_forwarddiff(f, x, flatten)\n\n\nCalculate the value and the log Jacobian determinant of f at x. flatten is used to get a vector out of the result that makes f a bijection.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.CustomTransform",
    "page": "Manual",
    "title": "TransformVariables.CustomTransform",
    "category": "type",
    "text": "CustomTransform(g, f, flatten)\n\nWrap a custom transform y = f(transform(g, x))in a type that calculates the log Jacobian of∂y/∂xusingForwardDiff` when necessary.\n\nUsually, g::TransformReals, but when an integer is used, it amounts to the identity transformation with that dimension.\n\nflatten should take the result from f, and return a flat vector with no redundant elements, so that x  y is a bijection. For example, for a covariance matrix the elements below the diagonal should be removed.\n\n\n\n\n\n"
},

{
    "location": "index.html#Defining-custom-transformations-1",
    "page": "Manual",
    "title": "Defining custom transformations",
    "category": "section",
    "text": "logjac_forwarddiff\nvalue_and_logjac_forwarddiff\nCustomTransform"
},

]}
