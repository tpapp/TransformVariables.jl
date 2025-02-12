module InverseFunctionsExt

import TransformVariables
import InverseFunctions

function InverseFunctions.inverse(f::TransformVariables.CallableTransform)
    return TransformVariables.inverse(f)
end
function InverseFunctions.inverse(f::TransformVariables.CallableInverse)
    return TransformVariables.inverse(f)
end

end # module
