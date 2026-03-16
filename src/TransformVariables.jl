module TransformVariables

using ArgCheck: @argcheck
import Compat
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
import ForwardDiff
using LogExpFunctions
using LinearAlgebra: UpperTriangular, logabsdet, norm, rmul!
using Random: AbstractRNG, GLOBAL_RNG
using StaticArrays: MMatrix, SMatrix, SArray, SVector, pushfirst
using CompositionsBase

include("utilities.jl")
include("generic.jl")
include("scalar.jl")
include("special_arrays.jl")
include("constant.jl")
include("aggregation.jl")
include("custom.jl")
include("vector.jl")

end # module
