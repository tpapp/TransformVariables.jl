module ReactantExt
using TransformVariables: TransformVariables, ArrayTransformation, LogJacFlag,  
                          logjac_zero, transform_with, _ensure_float, dimension
using Reactant
using Reactant: TracedRNumber, AnyTracedRArray

RInt = Union{Int, TracedRNumber{Int}}
Base.@propagate_inbounds function TransformVariables.tv_getindex(a::AnyTracedRArray, i::RInt)
    @allowscalar a[i]
end

TransformVariables._ensure_float(x::Type{T}) where {T<:TracedRNumber} = T

@noinline function TransformVariables.transform_with(
    flag::TransformVariables.LogJacFlag, 
    transformation::TransformVariables.ArrayTransformation, 
    x::AnyTracedRArray, 
    index::T
) where {T}
    (; inner_transformation, dims) = transformation
    # NOTE not using index increments as that somehow breaks type inference
    d = dimension(inner_transformation) # length of an element transformation
    len = prod(dims)              # number of elements
    𝐼 = reshape(range(index; length = len, step = d), dims)
    @info 𝐼
    ℓa = logjac_zero(flag, _ensure_float(eltype(x)))
    tmp,_,_ = transform_with(flag, inner_transformation, x, first(𝐼))
    if typeof(tmp) <: Number
        yℓ = similar(x, typeof(tmp), length(𝐼))
    elseif typeof(tmp) <: AbstractArray
        yℓ = [similar(tmp) for _ in 1:length(𝐼)]
    else
        throw(ArgumentError("Number and AbstractArray transformations are only supported in Reactant compilation mode"))
    end
    @trace for i in eachindex(𝐼)
        idx = 𝐼[i]
        y, ℓ, _ = transform_with(flag, inner_transformation, x, idx)
        # if !isempty(y)
        ℓa += ℓ        
        # end
        @allowscalar yℓ[i] = y
        i += 1
    end
    index′ = index + d * len
    yℓ, ℓa, index′
end


end