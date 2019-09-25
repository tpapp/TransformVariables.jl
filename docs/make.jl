using Documenter, TransformVariables

DocMeta.setdocmeta!(TransformVariables, :DocTestSetup, :(using TransformVariables); recursive=true)
makedocs(
    modules = [TransformVariables],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "TransformVariables.jl",
    clean = true,
    checkdocs = :exports,
    pages = Any["index.md", "internals.md"]
)

deploydocs(repo = "github.com/tpapp/TransformVariables.jl.git")
