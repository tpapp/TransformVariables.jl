module TransformVariables

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
import ForwardDiff
using LinearAlgebra: UpperTriangular, logabsdet
using MacroTools: @capture
using Parameters: @unpack
using Random: AbstractRNG, GLOBAL_RNG

include("utilities.jl")
include("generic.jl")
include("scalar.jl")
include("special_arrays.jl")
include("aggregation.jl")
include("custom.jl")

end # module
