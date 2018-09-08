export logjac_forwarddiff, value_and_logjac_forwarddiff, CustomTransform

"""
$(SIGNATURES)

Calculate the log Jacobian determinant of `f` at `x` using `ForwardDiff.

# Note

`f` should be a bijection, mapping from vectors of real numbers to vectors of
equal length.
"""
logjac_forwarddiff(f, x) = first(logabsdet(ForwardDiff.jacobian(f, x)))

"""
$(SIGNATURES)

Calculate the value and the log Jacobian determinant of `f` at `x`. `flatten` is
used to get a vector out of the result that makes `f` a bijection.
"""
value_and_logjac_forwarddiff(f, x, flatten = identity) =
    f(x), logjac_forwarddiff(flatten ∘ f, x)

"""
    CustomTransform(g, f, flatten)

Wrap a custom transform `y = f(transform(g, x))`` in a type that calculates the
log Jacobian of ``∂y/∂x`` using `ForwardDiff` when necessary.

Usually, `g::TransformReals`, but when an integer is used, it amounts to the
identity transformation with that dimension.

`flatten` should take the result from `f`, and return a flat vector with no
redundant elements, so that ``x ↦ y`` is a bijection. For example, for a
covariance matrix the elements below the diagonal should be removed.
"""
struct CustomTransform{G <: AbstractTransform, F, H} <: VectorTransform
    g::G
    f::F
    flatten::H
end

CustomTransform(n::Integer, f, flatten) =
    CustomTransform(as(Array, n), f, flatten)

dimension(t::CustomTransform) = dimension(t.g)

function transform_with(flag::NoLogJac, t::CustomTransform, x::RealVector)
    @unpack g, f = t
    f(first(transform_with(flag, g, x))), flag
end

function transform_with(flag::LogJac, t::CustomTransform, x::RealVector)
    @unpack g, f, flatten = t
    index = firstindex(x)
    xv = @view x[index:(index + dimension(g) - 1)]
    value_and_logjac_forwarddiff(x -> f(transform(g, x)), xv, flatten)
end
