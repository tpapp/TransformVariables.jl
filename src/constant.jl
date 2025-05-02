####
#### placeholder for constants
####

export Constant

"""
`Constant(value)`

Placeholder for inserting a constant. Inverse checks equality with `==`.
"""
struct Constant{T} <: VectorTransform
    value::T
end

dimension(::Constant) = 0

function transform_with(logjac_flag::LogJacFlag, t::Constant, x::AbstractVector, index)
    t.value, logjac_zero(logjac_flag, eltype(x)), index
end

inverse_eltype(t::Constant, ::Type) = Union{}

function inverse_at!(x::AbstractVector, index, t::Constant, y)
    @argcheck t.value == y
    index
end
