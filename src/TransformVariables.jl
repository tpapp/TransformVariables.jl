__precompile__()
module TransformVariables

using ArgCheck: @argcheck
using Compat: axes
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

function result_vec end

function transform_at end

function transform(t::TransformReals, x::RealVector)
    @argcheck dimension(t) == length(x)
    transform_at(t, isoneindexed(x) ? x : convert(Vector, x), 1)
end

function transform(t::TransformReals, ::LogJac, x::RealVector)
    @argcheck dimension(t) == length(x)
    transform_at(t, LOGJAC, isoneindexed(x) ? x : convert(Vector, x), 1)
end

# function _value_and_logjac(t::TransformReals{N}, x::RealVector)
#     J = DiffResults.JacobianResult(x)
#     ForwardDiff.jacobian!(J, x -> result_vec(t, transform(t, x)), x)
#     DiffResults.value(J), logdet(DiffResults.jacobian(J))
# end

# logjac(t::TransformReals, x) = _value_and_logjac(t, x)[2]

# value_and_logjac(t::TransformReals, x) = _value_and_logjac(t, x)

include("utilities.jl")
include("scalar.jl")
include("aggregation.jl")

end # module
