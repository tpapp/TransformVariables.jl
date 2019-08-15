export logjac_forwarddiff, value_and_logjac_forwarddiff, CustomTransform

"""
$(SIGNATURES)

Calculate the log Jacobian determinant of `f` at `x` using `ForwardDiff.

# Note

`f` should be a bijection, mapping from vectors of real numbers to vectors of
equal length.

When `handleNaN = true` (the default), NaN log Jacobians are converted to -Inf.
"""
function logjac_forwarddiff(f, x; handleNaN = true,
                            chunk = ForwardDiff.Chunk(x),
                            cfg = ForwardDiff.JacobianConfig(f, x, chunk))
    lj = first(logabsdet(ForwardDiff.jacobian(f, x, cfg)))
    isnan(lj) && handleNaN && return oftype(lj, -Inf)
    lj
end

"""
$(SIGNATURES)

Calculate the value and the log Jacobian determinant of `f` at `x`. `flatten` is
used to get a vector out of the result that makes `f` a bijection.
"""
function value_and_logjac_forwarddiff(f, x; flatten = identity, handleNaN = true,
                                      chunk = ForwardDiff.Chunk(x),
                                      cfg = ForwardDiff.JacobianConfig(flatten ∘ f, x, chunk))
    f(x), logjac_forwarddiff(flatten ∘ f, x; handleNaN = handleNaN, cfg = cfg)
end

@calltrans struct CustomTransform{G <: AbstractTransform, F, H, C} <: VectorTransform
    g::G
    f::F
    flatten::H
    cfg::C
end

_custom_f(g, f) = x -> f(transform(g, x))

_custom_chunk(g) = ForwardDiff.Chunk(zeros(dimension(g)))

_custom_cfg(g, f, flatten, chunk = _custom_chunk(g)) =
    ForwardDiff.JacobianConfig(flatten ∘ _custom_f(g, f), zeros(dimension(g)), chunk)

"""
$(SIGNATURES)

Wrap a custom transform `y = f(transform(g, x))`` in a type that calculates the log Jacobian
of ``∂y/∂x`` using `ForwardDiff` when necessary.

Usually, `g::TransformReals`, but when an integer is used, it amounts to the identity
transformation with that dimension.

`flatten` should take the result from `f`, and return a flat vector with no redundant
elements, so that ``x ↦ y`` is a bijection. For example, for a covariance matrix the
elements below the diagonal should be removed.

`chunk` and `cfg` can be used to configure `ForwardDiff.JacobianConfig`. `cfg` is used
directly, while `chunk = ForwardDiff.Chunk{N}()` can be used to obtain a type-stable
configuration.
"""
function CustomTransform(g::AbstractTransform, f, flatten;
                         chunk = _custom_chunk(g),
                         cfg = _custom_cfg(g, f, flatten, chunk))
    CustomTransform(g, f, flatten, cfg)
end

CustomTransform(n::Integer, f, flatten; kwargs...) =
    CustomTransform(as(Array, n), f, flatten; kwargs...)

dimension(t::CustomTransform) = dimension(t.g)

function transform_with(flag::NoLogJac, t::CustomTransform, x::AbstractVector, index)
    @unpack g, f = t
    f(first(transform_with(flag, g, x, index))), flag, index + dimension(t)
end

function transform_with(flag::LogJac, t::CustomTransform, x::AbstractVector, index)
    @unpack g, f, flatten, cfg = t
    index = firstindex(x)
    index′ = index + dimension(g)
    y, ℓ = value_and_logjac_forwarddiff(_custom_f(g, f), x[index:(index′ - 1)];
                                        flatten = flatten, cfg = cfg)
    y, ℓ, index′
end
