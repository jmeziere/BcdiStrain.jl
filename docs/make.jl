using Documenter, BcdiStrain

makedocs(
    sitename="BcdiStrain.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BCDI"=>"index.md",
        "BcdiStrain"=>"main.md",
        "Usage"=>"use.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiStrain.jl.git",
)
