Base.@deprecate random_arg(t::ScalarTransform; kwargs...) randn()
Base.@deprecate random_arg(t::VectorTransform; kwargs...) randn(dimension(t))
Base.@deprecate random_value(t::AbstractTransform; kwargs) transform(t, randn(dimension(t)))
