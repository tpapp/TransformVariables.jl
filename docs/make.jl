using Documenter, TransformVariables

DocMeta.setdocmeta!(TransformVariables, :DocTestSetup,
                    :(using TransformVariables); recursive=true)

makedocs(
    modules = [TransformVariables],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    clean = true,
    sitename = "TransformVariables.jl",
    authors = "Tamás K. Papp",
    checkdocs = :exports,
    pages = Any["index.md", "internals.md"]
)

deploydocs(
    repo = "github.com/tpapp/TransformVariables.jl.git",
    push_preview = true
)
