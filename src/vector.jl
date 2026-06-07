export VectorIdentity, VecTVExp, VecTVLogistic, VecTVShift, VecTVScale, VecTVNeg
#######
####### identity
#######

"""
$(TYPEDEF)

Identity ``x ↦ x``.
"""
struct VectorIdentity <: VectorTransform
    d::Int
    function VectorIdentity(d)
        new(d)
    end
end

dimension(t::VectorIdentity) = t.d
transform_with(flag::LogJacFlag, t::VectorIdentity, x::AbstractVector{T}, index::Int) where {T} = x, logjac_zero(flag, T), index + dimension(t)
inverse_eltype(t::VectorIdentity, T::Type) = eltype(T)
function inverse_at!(x::AbstractVector, index::Integer, t::VectorIdentity, y::AbstractVector)
    newindex = index + dimension(t)
    x[index:newindex-1] .= y
    return newindex
end


#######
####### elementary vector transforms
#######

"""
$(TYPEDEF)

Exponential transformation `x ↦ eˣ`. Maps from all reals to the positive reals.
"""
struct VecTVExp <: VectorTransform
    d::Int
    function VecTVExp(d)
        new(d)
    end
end

dimension(t::VecTVExp) = t.d
transform_with(flag::LogJacFlag, t::VecTVExp, x::AbstractVector{T}, index::Int) where {T} = exp.(x), flag isa LogJac ? abs(prod(x)) : logjac_zero(flag, T), index + dimension(t)
inverse_eltype(t::VecTVExp, T::Type) = eltype(T)
function inverse_at!(x::AbstractVector, index::Integer, t::VecTVExp, y::AbstractVector)
    newindex = index + dimension(t)
    x[index:newindex-1] .= log.(y)
    return newindex
end

"""
$(TYPEDEF)

Logistic transformation `x ↦ logit(x)`. Maps from all reals to (0, 1).
"""
struct VecTVLogistic <: VectorTransform
    d::Int
    function VecTVLogistic(d)
        new(d)
    end
end
dimension(t::VecTVLogistic) = t.d
transform_with(flag::LogJacFlag, t::VecTVLogistic, x::AbstractVector{T}, index::Int) where {T} = logistic.(x), flag isa LogJac ? prod(logistic_logjac.(x)) : logjac_zero(flag, T), index + dimension(t)
inverse_eltype(t::VecTVLogistic, T::Type) = eltype(T)
function inverse_at!(x::AbstractVector, index::Integer, t::VecTVLogistic, y::AbstractVector)
    newindex = index + dimension(t)
    x[index:newindex-1] .= logit.(y)
    return newindex
end

"""
$(TYPEDEF)

Shift transformation `x ↦ x + shift`.
"""
struct VecTVShift{T<:Real} <: VectorTransform
    shift::AbstractVector
    function VecTVShift(shift::AbstractVector{T}) where {T}
        return new{T}(shift)
    end
end
function VecTVShift(val::Real, dim::Integer)
    return VecTVShift(repeat([val;], dim))
end

dimension(t::VecTVShift) = length(t.shift)
transform_with(flag::LogJacFlag, t::VecTVShift, x::AbstractVector{T}, index::Int) where {T} = x .+ t.shift, logjac_zero(flag, T), index + dimension(t)
inverse_eltype(t::VecTVShift, T::Type) = eltype(T)
function inverse_at!(x::AbstractVector, index::Integer, t::VecTVShift, y::AbstractVector)
    newindex = index + dimension(t)
    x[index:newindex-1] .= y .+ t.shift
    return newindex
end

"""
$(TYPEDEF)

Scale transformation `x ↦ scale * x`.
"""
struct VecTVScale{T<:Real} <: VectorTransform
    scale::AbstractVector
    function VecTVScale(scale::AbstractVector{T}) where {T}
        return new{T}(scale)
    end
end
function VecTVScale(val::Real, dim::Integer)
    return VecTVScale(repeat([val;], dim))
end

dimension(t::VecTVScale) = length(t.shift)
transform_with(flag::LogJacFlag, t::VecTVScale, x::AbstractVector{T}, index::Int) where {T} = x .* t.scale, flag isa LogJac ? log(abs(prod(x))) : logjac_zero(flag, T), index + dimension(t)
inverse_eltype(t::VecTVScale, T::Type) = eltype(T)
function inverse_at!(x::AbstractVector, index::Integer, t::VecTVScale, y::AbstractVector)
    newindex = index + dimension(t)
    x[index:newindex-1] .= y .* t.scale
    return newindex
end

"""
$(TYPEDEF)

Negative transformation `x ↦ -x`.
"""
struct VecTVNeg <: VectorTransform
    d::Int
    function VecTVNeg(d)
        new(d)
    end
end
dimension(t::VecTVNeg) = t.d
transform_with(flag::LogJacFlag, t::VecTVNeg, x::AbstractVector{T}, index::Int) where {T} = -x, logjac_zero(flag, T), index + dimension(t)
inverse_eltype(t::VecTVNeg, T::Type) = eltype(T)
function inverse_at!(x::AbstractVector, index::Integer, t::VecTVNeg, y::AbstractVector)
    newindex = index + dimension(t)
    x[index:newindex-1] .= -y
    return newindex
end


### TODO composition of vector transforms

#######
####### composite scalar transforms
#######
###"""
###$(TYPEDEF)
###
###A composite scalar transformation, i.e. a sequence of scalar transformations.
###"""
###struct CompositeScalarTransform{Ts <: Tuple} <: ScalarTransform
###    transforms::Ts
###    function CompositeScalarTransform(transforms::Ts) where {Ts <: Tuple{ScalarTransform,Vararg{ScalarTransform}}}
###        new{Ts}(transforms)
###    end
###end
###
###transform(t::CompositeScalarTransform, x) = foldr(transform, t.transforms, init=x)
###function transform_and_logjac(ts::CompositeScalarTransform, x)
###    foldr(ts.transforms, init=(x, logjac_zero(LogJac(), typeof(x)))) do t, (x, logjac)
###        nx, nlogjac = transform_and_logjac(t, x)
###        (nx, logjac + nlogjac)
###    end
###end
###
###inverse(ts::CompositeScalarTransform, x) = foldl((y, t) -> inverse(t, y), ts.transforms, init=x)
###function inverse_and_logjac(ts::CompositeScalarTransform, x)
###    foldl(ts.transforms, init=(x, logjac_zero(LogJac(), typeof(x)))) do (x, logjac), t
###        nx, nlogjac = inverse_and_logjac(t, x)
###        (nx, logjac + nlogjac)
###    end
###end
###
###Base.:∘(t::ScalarTransform, s::ScalarTransform) = CompositeScalarTransform((t, s))
###Base.:∘(t::ScalarTransform, ct::CompositeScalarTransform) = CompositeScalarTransform((t, ct.transforms...))
###Base.:∘(ct::CompositeScalarTransform, t::ScalarTransform) = CompositeScalarTransform((ct.transforms..., t))
###Base.:∘(ct1::CompositeScalarTransform, ct2::CompositeScalarTransform) = CompositeScalarTransform((ct1.transforms..., ct2.transforms...))
###Base.:∘(t::ScalarTransform, tt::Vararg{ScalarTransform}) = foldl(∘, tt; init=t)