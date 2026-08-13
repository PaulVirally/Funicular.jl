# Turn everything in benchmark/results into figures.
#
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark --startup-file=no benchmark/plot.jl
#
# This is the only script here that needs a plotting package, so it has an
# environment of its own. The benchmarks themselves run from test/cuda, so that
# a machine set up to run the suite can run them too.
#
# Every file is a TSV with a header row. A numeric first column becomes the x
# axis of a line plot, one line per remaining column; a text first column
# becomes the labels of a bar chart.

using DelimitedFiles
using Plots

const RESULTS = joinpath(@__DIR__, "results")

isdir(RESULTS) || error("there is nothing in $RESULTS yet. Run the benchmarks first, for instance julia --project=test/cuda --startup-file=no benchmark/overlap.jl")

function readtsv(path)
    rows = readdlm(path, '\t')
    String.(rows[1, :]), rows[2:end, :]
end

axislabel(name) = replace(name, "_" => " ")

function figure(name, header, body)
    labels = reshape(axislabel.(header[2:end]), 1, :)
    if all(x -> x isa Number, body[:, 1])
        plot(Float64.(body[:, 1]), Float64.(body[:, 2:end]);
             label=labels, xlabel=axislabel(header[1]), marker=:circle,
             title=name, legend=:outertopright)
    else
        bar(String.(body[:, 1]), Float64.(body[:, 2]);
            label=axislabel(header[2]), xlabel=axislabel(header[1]),
            title=name, legend=false)
    end
end

for file in sort(readdir(RESULTS))
    endswith(file, ".tsv") || continue
    name = first(splitext(file))
    header, body = readtsv(joinpath(RESULTS, file))
    isempty(body) && continue
    out = joinpath(RESULTS, name * ".png")
    savefig(figure(name, header, body), out)
    println("wrote ", out)
end
