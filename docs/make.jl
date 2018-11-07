using Documenter, TransformVariables

makedocs(
    modules = [TransformVariables],
    sitename = "TransformVariables.jl",
    format = :html,
    checkdocs = :exports,
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/tpapp/TransformVariables.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
