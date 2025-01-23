module TransformVariables

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
import ForwardDiff
using LogExpFunctions
using LinearAlgebra: UpperTriangular, logabsdet
using Random: AbstractRNG, GLOBAL_RNG
using StaticArrays: MMatrix, SMatrix, SArray, SVector, pushfirst

import ChangesOfVariables
import InverseFunctions

include("utilities.jl")
include("generic.jl")
include("scalar.jl")
include("special_arrays.jl")
include("constant.jl")
include("aggregation.jl")
include("custom.jl")

end # module
