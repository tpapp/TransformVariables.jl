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
    "text": "dimension(t::TransformReals)\n\nThe dimension (number of elements) that t transforms.\n\n\n\n\n\n"
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
    "text": "inverse(t::TransformReals, y)\n\nReturn x so that transform(t, x) ≈ y.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.transform_logdensity",
    "page": "Manual",
    "title": "TransformVariables.transform_logdensity",
    "category": "function",
    "text": "transform_logdensity(t, f, x)\n\n\nLet y = t(x), and f(y) a log density at y. This function evaluates f ∘ t as a log density, taking care of the log Jacobian correction.\n\n\n\n\n\n"
},

{
    "location": "index.html#General-interface-1",
    "page": "Manual",
    "title": "General interface",
    "category": "section",
    "text": "dimension\ntransform\ntransform_and_logjac\ninverse\ntransform_logdensity"
},

{
    "location": "index.html#Specific-transformations-1",
    "page": "Manual",
    "title": "Specific transformations",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#TransformVariables.to_interval",
    "page": "Manual",
    "title": "TransformVariables.to_interval",
    "category": "function",
    "text": "to_interval(left, right)\n\n\nReturn a transformation that transforms a single real number to the given (open) interval.\n\nleft < right is required, but may be -∞ or ∞, respectively, in which case the appropriate transformation is selected. See ∞.\n\nSome common transformations are predefined as constants, see to_ℝ, to_ℝ₋, to_ℝ₊, to_𝕀.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.∞",
    "page": "Manual",
    "title": "TransformVariables.∞",
    "category": "constant",
    "text": "Placeholder representing of infinity for specifing interval boundaries. Supports the - operator, ie -∞.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.to_ℝ",
    "page": "Manual",
    "title": "TransformVariables.to_ℝ",
    "category": "constant",
    "text": "Transform to the real line (identity).\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.to_ℝ₊",
    "page": "Manual",
    "title": "TransformVariables.to_ℝ₊",
    "category": "constant",
    "text": "Transform to a non-negative real number.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.to_ℝ₋",
    "page": "Manual",
    "title": "TransformVariables.to_ℝ₋",
    "category": "constant",
    "text": "Transform to a non-positive real number.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.to_𝕀",
    "page": "Manual",
    "title": "TransformVariables.to_𝕀",
    "category": "constant",
    "text": "Transform to the unit interval (0, 1).\n\n\n\n\n\n"
},

{
    "location": "index.html#Scalar-transforms-1",
    "page": "Manual",
    "title": "Scalar transforms",
    "category": "section",
    "text": "to_interval\n∞to_ℝ\nto_ℝ₊\nto_ℝ₋\nto_𝕀"
},

{
    "location": "index.html#TransformVariables.to_unitvec",
    "page": "Manual",
    "title": "TransformVariables.to_unitvec",
    "category": "function",
    "text": "to_unitvec(n)\n\n\nReturn a transformation that transforms n - 1 real numbers to a unit vector (under Euclidean norm).\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.to_corr_cholesky",
    "page": "Manual",
    "title": "TransformVariables.to_corr_cholesky",
    "category": "function",
    "text": "to_corr_cholesky(n)\n\n\nReturn a transformation that transforms real numbers to an nn upper-triangular matrix Ω, such that Ω\'*Ω is a correlation matrix (positive definite, with unit diagonal).\n\n\n\n\n\n"
},

{
    "location": "index.html#Special-arrays-1",
    "page": "Manual",
    "title": "Special arrays",
    "category": "section",
    "text": "to_unitvec\nto_corr_cholesky"
},

{
    "location": "index.html#TransformVariables.to_array",
    "page": "Manual",
    "title": "TransformVariables.to_array",
    "category": "function",
    "text": "to_array(transformation, dims)\n\n\nReturn a transformation that applies transformation repeatedly to create an array with the given dims.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.to_tuple",
    "page": "Manual",
    "title": "TransformVariables.to_tuple",
    "category": "function",
    "text": "to_tuple(transformations)\n\n\nReturn a transformation that transforms consecutive groups of real numbers to a (named) tuple, using the given transformations.\n\n\n\n\n\nto_tuple(transformations)\n\n\n\n\n\n\nto_tuple(transformations)\n\n\n\n\n\n\nto_tuple(; (:: transformations NTransformations)...)\n\n\n\n\n\n\n"
},

{
    "location": "index.html#Aggregation-of-transformations-1",
    "page": "Manual",
    "title": "Aggregation of transformations",
    "category": "section",
    "text": "to_array\nto_tuple"
},

{
    "location": "index.html#TransformVariables.logjac_forwarddiff",
    "page": "Manual",
    "title": "TransformVariables.logjac_forwarddiff",
    "category": "function",
    "text": "logjac_forwarddiff(transformation, flatten, x)\n\n\nCalculate the log Jacobian of transformation at x using ForwardDiff.jacobian.\n\nflatten should be a bijection that maps the result of the transformation to a vector of reals. This means that elements which are redundant should not be part of the result; since f is continuous, this means that flatten should have the same number of elements as x.\n\n\n\n\n\n"
},

{
    "location": "index.html#TransformVariables.CustomTransform",
    "page": "Manual",
    "title": "TransformVariables.CustomTransform",
    "category": "type",
    "text": "CustomTransform(dimension, transformation, flatten)\n\nWrap a custom transform y = transformation(x) in a type that calculates the log Jacobian using ForwardDiff when necessary. See [logjac_forwarddiff] for the documentation of flatten.\n\n\n\n\n\n"
},

{
    "location": "index.html#Defining-custom-transformations-1",
    "page": "Manual",
    "title": "Defining custom transformations",
    "category": "section",
    "text": "logjac_forwarddiff\nCustomTransform"
},

]}
