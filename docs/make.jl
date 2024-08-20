using Documenter, BcdiStrain

makedocs(
    sitename="BcdiStrain.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BcdiStrain"=>"index.md",
        "Usage"=>"use.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiStrain.jl.git",
)
