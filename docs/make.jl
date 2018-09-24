using Documenter, TransformVariables

makedocs(
    modules = [TransformVariables],
    format = :html,
    sitename = "$TransformVariables.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/tpapp/TransformVariables.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
