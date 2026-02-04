module ReactantExt
using TransformVariables
using Reactant


Base.@propagate_inbounds function TransformVariables.tv_getindex(a::Reactant.AnyTracedRArray, i::Integer)
    @allowscalar a[i]
end

TransformVariables._ensure_float(x::Type{T}) where {T<:Reactant.TracedRNumber} = T

end