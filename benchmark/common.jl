# Shared by every benchmark in this directory. Each one prints a table and
# writes the same numbers to benchmark/results/<name>.tsv, which plot.jl turns
# into figures.

using Printf

const RESULTS = joinpath(@__DIR__, "results")

# Minimum of a few samples after a warmup run. The minimum rather than the mean
# because the quantity of interest is how fast the machine can do this, and
# every source of noise here only adds time.
function best(f, samples=5)
    f()
    minimum(@elapsed(f()) for _ in 1:samples)
end

function record(name::AbstractString, header, rows)
    mkpath(RESULTS)
    path = joinpath(RESULTS, name * ".tsv")
    open(path, "w") do io
        println(io, join(header, "\t"))
        for row in rows
            println(io, join(row, "\t"))
        end
    end
    println("wrote ", path)
    path
end

function banner(title::AbstractString, subtitle::AbstractString="")
    println("\n", title)
    println("=" ^ length(title))
    isempty(subtitle) || println(subtitle)
    println()
end

gbps(bytes, seconds) = bytes / seconds / 2^30

verdict(ok::Bool) = ok ? "ok" : "OVER"
