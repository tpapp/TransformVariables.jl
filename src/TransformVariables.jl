__precompile__()
module TransformVariables

using ArgCheck: @argcheck
using Compat: axes, undef
using Compat.LinearAlgebra: UpperTriangular
using DocStringExtensions: SIGNATURES
# import ForwardDiff
# import DiffResults: JacobianResult
using Parameters: @unpack


export TransformReals, dimension, transform, LOGJAC, inverse


# utilities

isoneindexed(v::AbstractVector) = axes(v, 1) isa Base.OneTo


# general

const RealVector{T <: Real} = AbstractVector{T}

struct LogJac end

const LOGJAC = LogJac()

abstract type TransformReals end

function dimension end

function transform_at end

function transform(t::TransformReals, x::RealVector)
    @argcheck dimension(t) == length(x)
    transform_at(t, isoneindexed(x) ? x : convert(Vector, x), 1)
end

function transform(t::TransformReals, ::LogJac, x::RealVector)
    @argcheck dimension(t) == length(x)
    transform_at(t, LOGJAC, isoneindexed(x) ? x : convert(Vector, x), 1)
end

include("utilities.jl")
include("scalar.jl")
include("special_arrays.jl")
include("aggregation.jl")

end # module
