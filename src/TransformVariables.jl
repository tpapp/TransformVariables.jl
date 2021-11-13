module TransformVariables

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
import ForwardDiff
using LinearAlgebra: UpperTriangular, logabsdet
using UnPack: @unpack
using Random: AbstractRNG, GLOBAL_RNG

import ChangesOfVariables
import InverseFunctions

include("utilities.jl")
include("generic.jl")
include("scalar.jl")
include("special_arrays.jl")
include("aggregation.jl")
include("custom.jl")

end # module
