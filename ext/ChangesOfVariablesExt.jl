module ChangesOfVariablesExt

import TransformVariables
import ChangesOfVariables

function ChangesOfVariables.with_logabsdet_jacobian(f::TransformVariables.CallableTransform, x)
    return TransformVariables.transform_and_logjac(f.t, x)
end
function ChangesOfVariables.with_logabsdet_jacobian(f::TransformVariables.CallableInverse, x)
    return TransformVariables.inverse_and_logjac(f.t, x)
end

end # module
