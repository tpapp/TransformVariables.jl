__precompile__()
module TransformVariables

using ArgCheck: @argcheck
# import ForwardDiff
# import DiffResults: JacobianResult
using Parameters: @unpack
using StatsFuns: logit, logistic

export
    TransformReals, dimension, transform, inverse, # logjac, transform_and_logjac,
    ∞


# general

const RealVector = AbstractVector{<: Real}

abstract type TransformReals end

function dimension end

function result_vec end

function transform_at end

function transform(t::TransformReals, x::RealVector)
    @argcheck dimension(t) == length(x)
    transform_at(t, x, 1)
end

# function _value_and_logjac(t::TransformReals{N}, x::RealVector)
#     J = DiffResults.JacobianResult(x)
#     ForwardDiff.jacobian!(J, x -> result_vec(t, transform(t, x)), x)
#     DiffResults.value(J), logdet(DiffResults.jacobian(J))
# end

# logjac(t::TransformReals, x) = _value_and_logjac(t, x)[2]

# value_and_logjac(t::TransformReals, x) = _value_and_logjac(t, x)

include("scalar.jl")
include("aggregation.jl")

end # module
