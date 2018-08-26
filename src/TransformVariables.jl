__precompile__()
module TransformVariables

using ArgCheck: @argcheck
using LinearAlgebra: UpperTriangular
using DocStringExtensions: SIGNATURES
using Parameters: @unpack

export dimension, transform, transform_and_logjac, transform_logdensity, inverse


# utilities

@inline isoneindexed(v::AbstractVector) = axes(v, 1) isa Base.OneTo

@inline ensureoneindexed(v::AbstractVector) =
    isoneindexed(v) ? v : convert(Vector, v)


# log absolute Jacobian determinant

abstract type LogJacFlag end

struct LogJac <: LogJacFlag end

const LOGJAC = LogJac()

struct NoLogJac <: LogJacFlag end

const NOLOGJAC = NoLogJac()

Base.:+(::NoLogJac, ::NoLogJac) = NOLOGJAC

logjac_zero(::LogJac, x) = zero(x)

logjac_zero(::NoLogJac, _) = NOLOGJAC


# general

const RealVector{T <: Real} = AbstractVector{T}

abstract type TransformReals end

function transform_at end

@inline function _transform(t::TransformReals, flag::LogJacFlag, x::RealVector)
    @argcheck dimension(t) == length(x)
    transform_at(t, flag, ensureoneindexed(x), 1)
end

"""
    $SIGNATURES
"""
transform(t::TransformReals, x::RealVector) = first(_transform(t, NOLOGJAC, x))

"""
    $SIGNATURES
"""
transform_and_logjac(t::TransformReals, x::RealVector) = _transform(t, LOGJAC, x)

"""
    $SIGNATURES
"""
function transform_logdensity(t::TransformReals, f, x)
    y, ℓ = transform_and_logjac(t, x)
    ℓ + f(y)
end

"""
    dimension(t::TransformReals)

The dimension (number of elements) that `t` transforms.
"""
function dimension end

include("utilities.jl")
include("scalar.jl")
include("special_arrays.jl")
include("aggregation.jl")

end # module
