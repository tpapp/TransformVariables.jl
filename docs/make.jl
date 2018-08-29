using Documenter, TransformVariables

makedocs(;
         sitename = "TransformVariables.jl",
         modules = [TransformVariables],
         format = :html,
         clean = true,
         checkdocs = :exports,
         pages = Any["Manual" => "index.md"]
)

deploydocs(
    repo = "github.com/tpapp/TransformVariables.jl.git",
    julia = "1.0"
)
