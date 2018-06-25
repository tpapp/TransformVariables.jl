__precompile__()
module TransformVariables

using ArgCheck: @argcheck
using Compat: axes, undef
using Compat.LinearAlgebra: UpperTriangular
using DocStringExtensions: SIGNATURES
# import ForwardDiff
# import DiffResults: JacobianResult
using Parameters: @unpack

export TransformReals, dimension, transform, transform_and_logjac, inverse


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

function dimension end

function transform_at end

@inline function _transform(t::TransformReals, flag::LogJacFlag, x::RealVector)
    @argcheck dimension(t) == length(x)
    transform_at(t, flag, ensureoneindexed(x), 1)
end

transform(t::TransformReals, x::RealVector) = first(_transform(t, NOLOGJAC, x))

transform_and_logjac(t::TransformReals, x::RealVector) = _transform(t, LOGJAC, x)

include("utilities.jl")
include("scalar.jl")
include("special_arrays.jl")
include("aggregation.jl")

end # module
