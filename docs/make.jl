using Documenter, DocumenterCitations, BcdiStrain

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename="BcdiStrain.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BcdiStrain"=>"index.md",
        "Usage"=>"use.md",
        "References"=>"refs.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiStrain.jl.git",
)
