"""
$SIGNATURES

Calculate the log Jacobian of `transformation` at `x` using
`ForwardDiff.jacobian`.

`flatten` should be a **bijection** that maps the result of the transformation
to a vector of reals. This means that elements which are redundant should not be
part of the result; since `f` is continuous, this means that `flatten` should
have the same number of elements as `x`.
"""
logjac_forwarddiff(transformation, flatten, x) =
    first(logabsdet(ForwardDiff.jacobian(flatten ∘ transformation, x)))

"""
    CustomTransform(dimension, transformation, flatten)

Wrap a custom transform ``y = transformation(x)`` in a type that calculates the
log Jacobian using `ForwardDiff` when necessary. See [`logjac_forwarddiff`] for
the documentation of `flatten`.
"""
struct CustomTransform{T,F} <: TransformReals
    dimension::Int
    transformation::T
    flatten::F
end

dimension(t::CustomTransform) = t.dimension

function transform_at(t::CustomTransform, flag::NoLogJac, x::RealVector,
                      index::Int)
    @unpack transformation, dimension = t
    transformation(@view x[index:(index + dimension - 1)]), flag
end

function transform_at(t::CustomTransform, flag::LogJac, x::RealVector,
                      index::Int)
    @unpack dimension, transformation, flatten = t
    xv = @view x[index:(index + dimension - 1)]
    transformation(xv), logjac_forwarddiff(transformation, flatten, xv)
end
