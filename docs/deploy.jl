get(ENV, "TRAVIS_BRANCH", "") == "master" || exit()
using Pkg
pkg"add Documenter"
using Documenter

deploydocs(
    repo = "github.com/tpapp/TransformVariables.jl.git",
    julia = "1.0",
    target = "build",
    deps = nothing,
    make = nothing
)
