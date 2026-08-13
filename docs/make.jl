using Documenter
using Funicular
using Literate

# docs/demo/minirsvd.jl is a program that runs, not a listing that was typed
# into a page. Literate turns the one file into the demo page so the two cannot
# drift apart, and test/test_demo.jl runs it on the reference backend. The plain
# code fence is deliberate: Literate's default is an `@example` block, which
# Documenter would run at build time, and this program needs a GPU and several
# gigabytes of scratch.
Literate.markdown(joinpath(@__DIR__, "demo", "minirsvd.jl"),
                  joinpath(@__DIR__, "src");
                  name="demo", credit=false,
                  codefence="```julia" => "```")

makedocs(sitename="Funicular.jl",
         authors="Paul Virally",
         modules=[Funicular],
         repo=Documenter.Remotes.GitHub("PaulVirally", "Funicular.jl"),
         checkdocs=:exports,
         pages=["Home" => "index.md",
                "When to use this" => "when-to-use.md",
                "Residency" => "residency.md",
                "Operators" => "operators.md",
                "Demo" => "demo.md",
                "Porting a randomized method" => "porting.md",
                "Reference" => "reference.md"],
         format=Documenter.HTML(canonical="https://paulvirally.github.io/Funicular.jl/stable/",
                                edit_link="main",
                                prettyurls=get(ENV, "CI", nothing) == "true"))

deploydocs(repo="github.com/PaulVirally/Funicular.jl", devbranch="main", push_preview=true)
