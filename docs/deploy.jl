import Pkg
pkg"add Documenter"
using Documenter

deploydocs(
    repo = "github.com/tpapp/TransformVariables.jl.git",
    julia = "1.0"
)
