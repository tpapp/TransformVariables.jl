module ReactantExt
using TransformVariables: TransformVariables, ArrayTransformation, LogJacFlag,  
                          logjac_zero, transform_with, _ensure_float, dimension
using Reactant
using Reactant: TracedRNumber, AnyTracedRArray

RInt = Union{Int, TracedRNumber{Int}}
Base.@propagate_inbounds function TransformVariables.tv_getindex(a::AnyTracedRArray, i::RInt)
    Reactant.@allowscalar a[i]
end

# The dims of the ArrayTransformation must be constant because Reactant can only deal with non-dynamic arrays.
Base.@nospecialize function Reactant.traced_type_inner(
    @nospecialize(prev::Type{ArrayTransformation{T, M}}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T, M}
    T_traced = Reactant.traced_type_inner(T, seen, mode, track_numbers, sharding, runtime)
    return TransformVariables.ArrayTransformation{T_traced, M}
end

TransformVariables._ensure_float(x::Type{T}) where {T<:TracedRNumber} = T

@noinline function TransformVariables.transform_with(
    flag::TransformVariables.LogJacFlag, 
    transformation::TransformVariables.ArrayTransformation, 
    x::AnyTracedRArray, 
    index
)
    (; inner_transformation, dims) = transformation
    # NOTE not using index increments as that somehow breaks type inference
    d = dimension(inner_transformation) # length of an element transformation
    len = prod(dims)              # number of elements
    𝐼 = range(index; length = len, step = d)

    # # Reactant can't easily handle the usual return type because the array eltype is not a Reactant primitive
    # # so we have to do a 2-pass algorithm
    # yℓ = map(index -> ((y, ℓ, _) = transform_with(flag, inner_transformation, x, index); y), 𝐼)
    # ℓa = map(index -> ((y, ℓ, _) = transform_with(flag, inner_transformation, x, index); ℓ), 𝐼)

    # index′ = index + d * len
    # y = reshape(yℓ, dims)
    # ℓa = sum(ℓa)
    # y, ℓa, index′
    
    tmp,_,_ = transform_with(flag, inner_transformation, x, first(𝐼))
    ℓa = logjac_zero(flag, _ensure_float(eltype(x)))
    if typeof(tmp) <: Number
        yℓ = similar(x, typeof(tmp), length(𝐼))
    elseif typeof(tmp) <: AbstractArray
        # convert to a larger array since reactant can't easily index into an julia array of Reactant arrays with a traced number.
        yℓ = similar(tmp, size(tmp)..., length(𝐼))
    else
        throw(ArgumentError("Number and AbstractArray transformations are only supported in Reactant compilation mode"))
    end
    @trace track_numbers=false for i in eachindex(𝐼)
        idx = 𝐼[i]
        y, ℓ, _ = transform_with(flag, inner_transformation, x, idx)
        if !isempty(y)
            ℓa += ℓ        
        end
        if tmp isa Number
            Reactant.@allowscalar yℓ[i] = y
        else
            idxs = ntuple(n-> Colon(), ndims(tmp)) 
            yℓ[idxs..., i] = y
        end
    end

    if tmp isa AbstractArray
        yℓ0 = collect(eachslice(yℓ, dims=ndims(yℓ)))
    else
        yℓ0 = yℓ
    end

    index′ = index + d * len
    reshape(yℓ0, dims), ℓa, index′
end

# To prevent ambiguities with regular method
function TransformVariables.transform_with(flag::LogJacFlag, t::ArrayTransformation{TransformVariables.Identity}, x::AnyTracedRArray, index)
    index′ = index+dimension(t)
    inds = Reactant.TracedUnitRange(index, index′-1, dimension(t))
    y = reshape(x[inds], t.dims)
    y, logjac_zero(flag, _ensure_float(eltype(x))), index′
end


end