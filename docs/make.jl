using Documenter, DocumenterCitations, BcdiStrain

makedocs(
    sitename="BcdiStrain.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BcdiStrain"=>"index.md",
        "Usage"=>"use.md",
        "References"=>"references.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiStrain.jl.git",
)
