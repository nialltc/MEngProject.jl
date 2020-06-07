using MEngProject
using Documenter

makedocs(;
    modules=[MEngProject],
    authors="nialltc <39237476+nialltc@users.noreply.github.com> and contributors",
    repo="https://github.com/nialltc/MEngProject.jl/blob/{commit}{path}#L{line}",
    sitename="MEngProject.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nialltc.github.io/MEngProject.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nialltc/MEngProject.jl",
)
