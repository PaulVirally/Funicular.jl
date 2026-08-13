# Every benchmark in this directory, in the order they are worth reading.
#
#     julia --project=test/cuda --startup-file=no benchmark/runall.jl
#
# Each one runs in a process of its own, so a failure in one leaves the rest to
# run and none of them inherits another's device state. Results land in
# benchmark/results as TSV; benchmark/plot.jl turns them into figures.
#
# The whole suite takes tens of minutes, most of it in unified.jl, which
# allocates more than the device's memory twice over. Run the scripts
# individually while chasing something specific.

const SCRIPTS = ("overlap.jl", "resident.jl", "pinned.jl", "cholqr.jl", "disk.jl", "unified.jl")

failed = String[]
for script in SCRIPTS
    path = joinpath(@__DIR__, script)
    println("\n", "─" ^ 72, "\n", path, "\n")
    try
        run(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no $path`)
    catch
        push!(failed, script)
        println("\n$script failed; carrying on with the rest")
    end
end

if isempty(failed)
    println("\nall ", length(SCRIPTS), " benchmarks ran")
else
    println("\nfailed: ", join(failed, ", "))
end
println("results are in ", joinpath(@__DIR__, "results"))
