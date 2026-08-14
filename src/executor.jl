# One panel matrix's part in a sweep: whether the sweep reads its panels into
# the staging buffer, writes the buffer back into them, or both.
struct SweepOperand{T,P}
    pm::PanelMatrix{T,P}
    read::Bool
    write::Bool
end

readpanels(pm::PanelMatrix) = SweepOperand(pm, true, false)
writepanels(pm::PanelMatrix) = SweepOperand(pm, false, true)
updatepanels(pm::PanelMatrix) = SweepOperand(pm, true, true)

# How a sweep cuts the matrix into steps. Column panels are the traversal for
# work that acts on whole vectors, that is, for everything the operator touches.
# Row blocks are the traversal for work that multiplies by a k×k matrix on the
# right. Such a product mixes columns, so a column panel on its own is not
# enough, but a block of rows spanning every panel is.
abstract type Traversal end

# A panel may appear more than once. That is how project sweeps the whole matrix
# once per output block. Repeats are only allowed while nothing writes.
struct PanelSteps{L<:AbstractVector} <: Traversal
    panels::Vector{Int}
    labels::L
end

PanelSteps(count::Integer) = PanelSteps(collect(1:Int(count)), Base.OneTo(Int(count)))

struct RowSteps <: Traversal
    blocks::Vector{UnitRange{Int}}
end

function RowSteps(N::Integer, height::Integer)
    h = clamp(Int(height), 1, Int(N))
    RowSteps([top:min(top + h - 1, Int(N)) for top in 1:h:Int(N)])
end

rowtraversal(pm::PanelMatrix) = RowSteps(pm.N, row_block_height(pm.N, pm.k, pm.w))

nsteps(traversal::PanelSteps) = length(traversal.panels)
nsteps(traversal::RowSteps) = length(traversal.blocks)

steplabel(traversal::PanelSteps, step::Int) = traversal.labels[step]
steplabel(::RowSteps, step::Int) = step

repeats_a_panel(traversal::PanelSteps) = !allunique(traversal.panels)
repeats_a_panel(::RowSteps) = false

buffer_dims(::PanelSteps, pm::PanelMatrix) = (pm.N, pm.w)
buffer_dims(traversal::RowSteps, pm::PanelMatrix) = (length(first(traversal.blocks)), pm.k)

assert_sweepable(::PanelSteps, X::PanelMatrix, Y::PanelMatrix) = assert_same_panels(X, Y)
assert_sweepable(::RowSteps, X::PanelMatrix, Y::PanelMatrix) = assert_same_rows(X, Y)

step_view(traversal::PanelSteps, buffer, step::Int, pm::PanelMatrix) = view(buffer, :, 1:pm.panels[traversal.panels[step]].width)
step_view(traversal::RowSteps, buffer, step::Int, ::PanelMatrix) = view(buffer, 1:length(traversal.blocks[step]), :)

stage_in!(traversal::PanelSteps, dst, pm::PanelMatrix, step::Int; queue, via=nothing) = materialize!(dst, pm.panels[traversal.panels[step]], pm.plan; queue=queue, via=via)
stage_in!(traversal::RowSteps, dst, pm::PanelMatrix, step::Int; queue, via=nothing) = materialize_rows!(dst, pm, traversal.blocks[step]; queue=queue, via=via)

stage_out!(traversal::PanelSteps, pm::PanelMatrix, src, step::Int; queue, via=nothing) = writeback!(pm.panels[traversal.panels[step]], src, pm.plan; queue=queue, via=via)
stage_out!(traversal::RowSteps, pm::PanelMatrix, src, step::Int; queue, via=nothing) = writeback_rows!(pm, src, traversal.blocks[step]; queue=queue, via=via)

# Which panels one step reads or writes. A row block is a slice of every panel,
# so a row sweep touches the whole matrix at every step.
steppanels(traversal::PanelSteps, pm::PanelMatrix, step::Int) = (pm.panels[traversal.panels[step]],)
steppanels(::RowSteps, pm::PanelMatrix, ::Int) = pm.panels

# The producer's half of the disk tier: bring every panel a step needs into host
# memory and hold it there. This is where the read of panel j+1 happens while
# the pipeline works on panel j, and it runs on the channel's own task so that
# neither the disk nor the lock around it reaches the loop issuing the pipeline.
function stage_step!(traversal::Traversal, pm::PanelMatrix, step::Int)
    for panel in steppanels(traversal, pm, step)
        stage_host!(panel, pm.plan)
    end
    nothing
end

# The pins the step took are replaced by the event its copies finish on, so an
# eviction of one of these panels waits for that event.
function release_step!(traversal::Traversal, operands, step::Int, hold)
    for operand in operands, panel in steppanels(traversal, operand.pm, step)
        unpin!(operand.pm.plan, panel, hold)
    end
    nothing
end

# The producer pins each step and the loop releases it, so a sweep that ran to
# the end leaves nothing pinned. A sweep that gave up part way through leaves
# the steps it never issued pinned, and only that path reaches this function.
function clear_pins!(operands)
    plan = first(operands).pm.plan
    lock(plan.hostpool.lock) do
        for operand in operands, panel in operand.pm.panels
            panel.pins = 0
        end
    end
    nothing
end

function stage_channel(traversal::Traversal, operands; depth::Integer=FEED_DEPTH)
    Channel{Int}(depth; spawn=true) do feed
        for step in 1:nsteps(traversal)
            for operand in operands
                stage_step!(traversal, operand.pm, step)
            end
            put!(feed, step)
        end
    end
end

# Called from the failure path of a sweep, where the exception has already been
# raised. Draining here only keeps pending work off the staging buffers once
# they go back to the pool.
function sync_quietly(backend::DeviceBackend, queues)
    for queue in queues
        try
            sync_queue(backend, queue)
        catch
        end
    end
    nothing
end

"""
    sweep!(f!, operands...; nbuffers=nothing, traversal=nothing)

Visits a set of panel matrices step by step, calling `f!(label, buffers...)` with
one device resident staging buffer per operand, in the order the operands were
given. Build the operands with [`readpanels`](@ref), [`writepanels`](@ref) and
[`updatepanels`](@ref). A read operand is copied into its buffer before `f!`
runs, a write operand is copied back out of it after.

A step is one column panel by default, labelled with the panel index. Passing
[`RowSteps`](@ref) as the traversal instead makes a step a block of rows
spanning every panel, with an `nrows × k` buffer.

The operands have to agree on whatever dimension a step names, and on nothing
more. Under column panel steps that means the same `k` and the same panel width;
the row counts are free, so one sweep can read an `n × k` matrix and write an
`m × k` one. Under row block steps it means the same `N`, and the column counts
and the widths are free. Every operand still belongs to one plan, and each gets
staging buffers sized from its own matrix.

Uploads go on one queue, `f!` on a second, writebacks on a third, ordered by
events so that with `nbuffers=2` step `s+1` uploads while step `s` computes.
A staging buffer goes to a new step only after the step that held it `nbuffers`
steps earlier is done with it. The loop itself never synchronizes and never
allocates: the buffers come from the plan's pool and the host waits once, at the
end.

A separate task runs a couple of steps ahead of that loop, bringing each step's
panels up from the disk tier if that is where they are, so a disk read overlaps
the pipeline rather than stalling it. Those panels stay in host memory until the
loop has issued their copies, so `host_budget` cannot be made arbitrarily small:
see [`ResidencyPlan`](@ref).

# Arguments
- `f!`: Called once per step as `f!(label, buffers...)`
- `operands::SweepOperand...`: The panel matrices, each wrapped as a read, write or update operand
- `nbuffers=nothing`: Staging buffers per operand, defaulting to the plan's `nbuffers`
- `traversal=nothing`: What a step is, defaulting to one step per column panel
"""
function sweep!(f!, operands::SweepOperand...; nbuffers=nothing, traversal=nothing)
    isempty(operands) && throw(ArgumentError("a sweep needs at least one panel matrix, wrapped as readpanels(pm), writepanels(pm) or updatepanels(pm)"))
    reference = first(operands).pm
    plan = reference.plan
    steps = traversal === nothing ? PanelSteps(npanels(reference)) : traversal
    for operand in operands
        assert_sweepable(steps, reference, operand.pm)
        operand.pm.plan === plan || throw(ArgumentError("every panel matrix in one sweep must belong to the same ResidencyPlan, since the plan is what owns the staging buffers the sweep runs through. Build both matrices from one plan"))
        operand.write && assert_writable(operand.pm)
    end
    for i in 2:length(operands), l in 1:(i - 1)
        operands[i].pm === operands[l].pm && throw(ArgumentError("the same panel matrix takes part in one sweep twice. Pass it once as updatepanels(pm) if the sweep both reads and writes it"))
    end
    requested = nbuffers === nothing ? plan.nbuffers : Int(nbuffers)
    requested >= 1 || throw(ArgumentError("nbuffers must be at least 1, got $requested"))
    writing = any(operand -> operand.write, operands)
    writing && repeats_a_panel(steps) && throw(ArgumentError("this sweep visits a panel more than once and writes back, so the second visit would overwrite the first. Repeated panels are only allowed in a read-only sweep"))

    backend = plan.backend
    count = nsteps(steps)
    nb = min(requested, count)
    # The dimension a step leaves free may differ between operands, so each set
    # of buffers is sized from its own matrix. The device pool is keyed by
    # eltype and shape, so two sets of different sizes come out of it without
    # colliding.
    dims = map(operand -> buffer_dims(steps, operand.pm), operands)
    buffers = map((operand, d) -> checkout_device_buffers!(plan, eltype(operand.pm), d, nb), operands, dims)
    staging = map((operand, d) -> checkout_staging_buffers!(plan, operand.pm, d, nb), operands, dims)
    up, compute, down = make_sweep_queues(backend)
    slotfree = Vector{Any}(nothing, nb)
    feed = stage_channel(steps, operands)
    try
        for _ in 1:count
            step = take!(feed)
            slot = mod1(step, nb)
            targets = map((operand, buffer) -> step_view(steps, buffer[slot], step, operand.pm), operands, buffers)
            vias = map((operand, set) -> set === nothing ? nothing : step_view(steps, set[slot], step, operand.pm), operands, staging)
            label = steplabel(steps, step)

            free = slotfree[slot]
            free === nothing || wait_event(backend, up, free)
            for (operand, target, via) in zip(operands, targets, vias)
                operand.read && stage_in!(steps, target, operand.pm, step; queue=up, via=via)
            end
            wait_event(backend, compute, record_event(backend, up))
            done = submit!(backend, compute, () -> f!(label, targets...))
            if writing
                wait_event(backend, down, done)
                for (operand, target, via) in zip(operands, targets, vias)
                    operand.write && stage_out!(steps, operand.pm, target, step; queue=down, via=via)
                end
                done = record_event(backend, down)
            end
            slotfree[slot] = done
            release_step!(steps, operands, step, done)
        end
        # Compute first so the caller sees an exception thrown by the panel function.
        sync_queue(backend, compute)
        sync_queue(backend, up)
        sync_queue(backend, down)
    catch
        clear_pins!(operands)
        rethrow()
    finally
        close(feed)
        sync_quietly(backend, (up, compute, down))
        for buffer in buffers
            checkin_device_buffers!(plan, buffer)
        end
        for set in staging
            set === nothing || checkin_device_buffers!(plan, set)
        end
    end
    nothing
end

"""
    foreachpanel(f!, pm; write=true, nbuffers=nothing)

Runs `f!(j, panel)` on every panel of `pm` in order, with `panel` a device
resident view of panel `j`. This is the escape hatch for device work that
Funicular does not provide itself: the sweep handles residency and overlap, and
`f!` does the work on the panel.

`panel` is a view of a staging buffer, `panelwidth(pm, j)` columns wide, and is
valid only for the duration of the call. Keep `f!` to work it issues on the
device. Its results reach the host when `foreachpanel` returns, not when `f!`
does.

Two staging buffers hide the copies behind the compute, while one buffer
serializes the pipeline and is the correctness oracle. Panels are visited in
order under both, so an accumulation over panels sums in the same order either
way.

# Arguments
- `f!`: Called as `f!(j, panel)` once per panel
- `pm::PanelMatrix`: The panel matrix to sweep
- `write::Bool=true`: Whether to copy each buffer back into panel `j` and mark the panel dirty. Pass `false` for a read-only sweep
- `nbuffers=nothing`: Number of staging buffers, defaulting to the plan's setting
"""
function foreachpanel(f!, pm::PanelMatrix; write::Bool=true, nbuffers=nothing)
    sweep!(f!, write ? updatepanels(pm) : readpanels(pm); nbuffers=nbuffers)
end
