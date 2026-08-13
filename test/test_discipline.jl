const SYNCHRONIZATION = (:sync_all, :sync_queue, :sync_event, :sync_quietly)

function definition_name(expr::Expr)
    expr.head in (:function, :(=)) || return nothing
    signature = expr.args[1]
    signature isa Expr && signature.head === :where && (signature = signature.args[1])
    signature isa Expr && signature.head === :call || return nothing
    signature.args[1] isa Symbol ? signature.args[1] : nothing
end

# The synchronization helpers may loop over queues. A sweep may not synchronize
# per panel.
function syncs_inside_loops(expr, inloop::Bool=false)
    found = Symbol[]
    expr isa Expr || return found
    name = definition_name(expr)
    name !== nothing && startswith(String(name), "sync") && return found
    if inloop && expr.head === :call && expr.args[1] isa Symbol && expr.args[1] in SYNCHRONIZATION
        push!(found, expr.args[1])
    end
    nested = inloop || expr.head in (:for, :while)
    for arg in expr.args
        append!(found, syncs_inside_loops(arg, nested))
    end
    found
end

function sourcefiles()
    paths = String[]
    for directory in ("src", "ext")
        top = joinpath(@__DIR__, "..", directory)
        isdir(top) || continue
        for (root, _, files) in walkdir(top)
            append!(paths, joinpath(root, file) for file in files if endswith(file, ".jl"))
        end
    end
    paths
end

@testset "no synchronization inside a sweep loop" begin
    for path in sourcefiles()
        parsed = Meta.parseall(read(path, String); filename=path)
        @test syncs_inside_loops(parsed) == Symbol[]
    end
end

@testset "the loop detector has teeth" begin
    offender = Meta.parseall("function sweep(); for j in 1:3; sync_queue(b, q); end; end")
    @test syncs_inside_loops(offender) == [:sync_queue]
    allowed = Meta.parseall("function sweep(); for j in 1:3; end; sync_queue(b, q); end")
    @test syncs_inside_loops(allowed) == Symbol[]
    whitelisted = Meta.parseall("function sync_them(qs); for q in qs; sync_queue(b, q); end; end")
    @test syncs_inside_loops(whitelisted) == Symbol[]
end
