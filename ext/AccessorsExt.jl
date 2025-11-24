module AccessorsExt

import Accessors: set, insert, delete
using Accessors: IndexLens, PropertyLens
using TransformVariables: TransformTuple, _inner

set(t::TransformTuple, ::typeof(_inner), inner) = TransformTuple(inner)
set(t::TransformTuple, lens::IndexLens, val) = set(t, lens ∘ _inner, val)
insert(t::TransformTuple{<:Tuple}, lens::IndexLens, val) = insert(t, lens ∘ _inner, val)
delete(t::TransformTuple, lens::IndexLens) = delete(t, lens ∘ _inner)
set(t::TransformTuple{<:NamedTuple}, lens::PropertyLens, val) = set(t, lens ∘ _inner, val)
insert(t::TransformTuple{<:NamedTuple}, lens::PropertyLens, val) = insert(t, lens ∘ _inner, val)
delete(t::TransformTuple{<:NamedTuple}, lens::PropertyLens) = delete(t, lens ∘ _inner)

end
