const GiB = 1024^3

# How far the executor's producer task may run ahead of the pipeline, in steps.
# It lives here rather than with the executor because it decides how many panels
# a sweep holds in host memory at once, and so how small host_budget is allowed
# to be.
const FEED_DEPTH = 2

mutable struct DevicePool
    budget::Int
    allocated::Int
    available::Dict{Tuple{DataType,Dims{2}},Vector{Any}}
end

DevicePool(budget::Integer) = DevicePool(Int(budget), 0, Dict{Tuple{DataType,Dims{2}},Vector{Any}}())

mutable struct HostPool
    budget::Int
    reserved::Int
    slabs::Vector{HostSlab}
    cursors::Vector{Int}
    free::Dict{Int,Vector{HostBlock}}
    residents::Vector{Any}
    lock::ReentrantLock
end

HostPool(budget::Integer) = HostPool(Int(budget), 0, HostSlab[], Int[], Dict{Int,Vector{HostBlock}}(), Any[], ReentrantLock())

"""
    ResidencyPlan(; device_budget, host_budget, backend=CPUBackend(), kwargs...)

Owner of all data Funicular allocates: the device buffer pool through which the
pipeline stages panels, and the host slabs where cold panels are stored. Budgets
are in bytes and are checked before allocating, so a plan that is too small for
a given panel width throws an error.

Giving the plan a `scratch_dir` means `host_budget` no longer has to cover the
whole matrix. It does still have to cover what a sweep holds at once: a panel
sweep stages up to $(FEED_DEPTH + 2) steps ahead of the pipeline, one panel per
matrix per step, and `gram` and `cholqr2!` traverse rows and so hold every panel
of their matrix.

# Arguments
- `backend::DeviceBackend=CPUBackend()`: The backend the plan allocates through
- `device_budget::Real`: Bytes the plan may allocate on the device. Take it from the device's total memory minus a reserve
- `host_budget::Real`: Bytes the plan may allocate for the host tier, page-locked on a real device backend. Slabs are allocated on demand up to this cap, not upfront, since page-locking is slow
- `workspace_bytes::Real=0`: Device memory the operator needs for itself, held back from the buffer pool
- `nbuffers::Integer=2`: Staging buffers per sweep the budget arithmetic assumes, 2 for the double-buffered pipeline and 1 for the serialized oracle
- `target_panel_bytes::Real=2 * GiB`: Panel size aimed for when choosing the panel width, clamped to what the budget allows
- `panel_width=nothing`: Fixed panel width, overriding the choice above
- `host_eltype=nothing`, `disk_eltype=nothing`: Storage eltype of the cold tiers, e.g. `ComplexF32` under a `ComplexF64` compute eltype, with the conversion fused into the transfers. Defaults to the compute eltype, since narrowing must be asked for
- `scratch_dir=nothing`: Directory for the disk tier. A plan that has one gives every `PanelMatrix` a file of its own and spills panels to it when the host tier fills up. Needs HDF5.jl loaded
"""
struct ResidencyPlan{B<:DeviceBackend}
    backend::B
    device_budget::Int
    host_budget::Int
    scratch_dir::Union{Nothing,String}
    workspace_bytes::Int
    nbuffers::Int
    target_panel_bytes::Int
    panel_width::Union{Nothing,Int}
    host_eltype::Union{Nothing,DataType}
    disk_eltype::Union{Nothing,DataType}
    devicepool::DevicePool
    hostpool::HostPool
end

function ResidencyPlan(; backend::DeviceBackend=CPUBackend(), device_budget::Real, host_budget::Real, scratch_dir=nothing, workspace_bytes::Real=0, nbuffers::Integer=2, target_panel_bytes::Real=2 * GiB, panel_width=nothing, host_eltype=nothing, disk_eltype=nothing)
    dev = floor(Int, device_budget)
    host = floor(Int, host_budget)
    work = floor(Int, workspace_bytes)
    work >= 0 || throw(ArgumentError("workspace_bytes must be nonnegative, got $work"))
    nbuffers >= 1 || throw(ArgumentError("nbuffers must be at least 1, got $nbuffers"))
    target_panel_bytes >= 1 || throw(ArgumentError("target_panel_bytes must be positive, got $target_panel_bytes"))
    host > 0 || throw(ArgumentError("host_budget must be positive, got $host"))
    dev > work || throw(ArgumentError("device_budget of $(human_readable_bytes(dev)) does not cover the operator workspace of $(human_readable_bytes(work)), leaving nothing for panel buffers. Raise device_budget or report a smaller workspace_bytes"))
    width = panel_width === nothing ? nothing : Int(panel_width)
    width === nothing || width >= 1 || throw(ArgumentError("panel_width must be positive, got $width"))
    for (name, S) in (("host_eltype", host_eltype), ("disk_eltype", disk_eltype))
        S === nothing && continue
        S isa DataType && isconcretetype(S) || throw(ArgumentError("$name must be a concrete type, got $S"))
    end
    dir = scratch_dir === nothing ? nothing : String(scratch_dir)
    ResidencyPlan(backend, dev, host, dir, work, Int(nbuffers), floor(Int, target_panel_bytes),
                  width, host_eltype, disk_eltype, DevicePool(dev - work), HostPool(host))
end

function Base.show(io::IO, plan::ResidencyPlan)
    print(io, "ResidencyPlan(", nameof(typeof(plan.backend)),
          ", device ", human_readable_bytes(plan.device_budget),
          ", host ", human_readable_bytes(plan.host_budget))
    plan.scratch_dir === nothing || print(io, ", scratch ", repr(plan.scratch_dir))
    print(io, ")")
end

panel_bytes(N::Integer, w::Integer, ::Type{T}) where {T} = Int(N) * Int(w) * sizeof(T)

function device_bytes_required(N::Integer, w::Integer, ::Type{T}; nbuffers::Integer=2, extra_buffers::Integer=0, workspace_bytes::Integer=0) where {T}
    (Int(nbuffers) + Int(extra_buffers)) * panel_bytes(N, w, T) + Int(workspace_bytes)
end

"""
    choose_panel_width(N, k, T; device_budget, kwargs...) -> Int

Largest panel width that keeps `nbuffers + extra_buffers` device panel buffers
plus `workspace_bytes` inside `device_budget`, capped at `target_panel_bytes`
per panel and at `k`. The width is then evened out over the panels it implies,
so a `k = 1000` matrix whose budget allows `w = 999` is cut into two panels of
500 rather than one of 999 and one of 1.
"""
function choose_panel_width(N::Integer, k::Integer, ::Type{T}; device_budget::Real, nbuffers::Integer=2, extra_buffers::Integer=0, workspace_bytes::Real=0, target_panel_bytes::Real=2 * GiB) where {T}
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    buffers = Int(nbuffers) + Int(extra_buffers)
    buffers >= 1 || throw(ArgumentError("nbuffers + extra_buffers must be at least 1, got $buffers"))
    column = Int(N) * sizeof(T)
    available = floor(Int, device_budget) - floor(Int, workspace_bytes)
    per_buffer = fld(available, buffers)
    per_buffer >= column || throw(ArgumentError("device budget of $(human_readable_bytes(floor(Int, device_budget))) leaves $(human_readable_bytes(available)) after a $(human_readable_bytes(floor(Int, workspace_bytes))) operator workspace, so $(human_readable_bytes(per_buffer)) for each of $buffers panel buffers, but a single column of $N $T entries needs $(human_readable_bytes(column)). Raise device_budget, lower nbuffers, or narrow the compute eltype"))
    w = min(Int(k), max(1, fld(min(floor(Int, target_panel_bytes), per_buffer), column)))
    cld(Int(k), cld(Int(k), w))
end

"""
    row_block_height(N, k, w) -> Int

Rows in one block of the row traversal, chosen so that an `nrows × k` block
buffer holds as much as an `N × w` panel buffer does. A sweep over row blocks
therefore takes the same number of steps and moves the same bytes as a sweep
over panels.
"""
row_block_height(N::Integer, k::Integer, w::Integer) = clamp(fld(Int(N) * Int(w), Int(k)), 1, Int(N))

function resolve_panel_width(plan::ResidencyPlan, N::Integer, k::Integer, ::Type{T}; override=nothing, extra_buffers::Integer=0) where {T}
    requested = override === nothing ? plan.panel_width : override
    if requested === nothing
        return choose_panel_width(N, k, T; device_budget=plan.device_budget,
                                  nbuffers=plan.nbuffers, extra_buffers=extra_buffers,
                                  workspace_bytes=plan.workspace_bytes,
                                  target_panel_bytes=plan.target_panel_bytes)
    end
    w = Int(requested)
    1 <= w <= k || throw(ArgumentError("panel width $w must lie in 1:$k"))
    needed = device_bytes_required(N, w, T; nbuffers=plan.nbuffers,
                                   extra_buffers=extra_buffers,
                                   workspace_bytes=plan.workspace_bytes)
    if needed > plan.device_budget
        buffers = plan.nbuffers + Int(extra_buffers)
        fits = choose_panel_width(N, k, T; device_budget=plan.device_budget,
                                  nbuffers=plan.nbuffers, extra_buffers=extra_buffers,
                                  workspace_bytes=plan.workspace_bytes,
                                  target_panel_bytes=plan.device_budget)
        throw(ArgumentError("panel width $w needs $(human_readable_bytes(needed)) on the device ($buffers buffers of $(human_readable_bytes(panel_bytes(N, w, T))) plus $(human_readable_bytes(plan.workspace_bytes)) of operator workspace) but device_budget is $(human_readable_bytes(plan.device_budget)). Use panel_width ≤ $fits or raise device_budget"))
    end
    w
end

function checkout_device_buffers!(plan::ResidencyPlan, ::Type{T}, dims::Dims{2}, count::Integer) where {T}
    supports_eltype(plan.backend, T) || throw(ArgumentError("backend $(nameof(typeof(plan.backend))) cannot compute in $T. Apple GPUs have no double precision, so a Metal plan works in ComplexF32 or Float32; build the panel matrices in one of those"))
    pool = plan.devicepool
    cache = get!(() -> Any[], pool.available, (T, dims))
    buffers = Any[]
    nbytes = prod(dims) * sizeof(T)
    for _ in 1:count
        if isempty(cache)
            pool.allocated + nbytes <= pool.budget || throw(ArgumentError("device pool is full: a $(dims[1])×$(dims[2]) $T buffer needs $(human_readable_bytes(nbytes)), $(human_readable_bytes(pool.allocated)) is already allocated and the pool caps at $(human_readable_bytes(pool.budget)) (device_budget $(human_readable_bytes(plan.device_budget)) minus $(human_readable_bytes(plan.workspace_bytes)) of operator workspace). Reduce the panel width, lower nbuffers, or raise device_budget"))
            push!(buffers, alloc_device(plan.backend, T, dims))
            pool.allocated += nbytes
        else
            push!(buffers, pop!(cache))
        end
    end
    buffers
end

function checkin_device_buffers!(plan::ResidencyPlan, buffers)
    for buffer in buffers
        cache = get!(() -> Any[], plan.devicepool.available, (eltype(buffer), size(buffer)))
        push!(cache, buffer)
    end
    nothing
end

device_bytes_allocated(plan::ResidencyPlan) = plan.devicepool.allocated

# Makes room for one panel block, spilling panels that have somewhere colder to
# go until there is some. The spill itself happens with the pool unlocked, since
# it waits on the copy that last touched the victim and then writes a panel to
# disk, and holding the lock through that would stall the task issuing the
# pipeline.
function acquire_host_block!(plan::ResidencyPlan, nbytes::Integer)
    n = align_up(nbytes, SLAB_ALIGN)
    while true
        block = take_host_block!(plan, n)
        block === nothing || return block
        victim = claim_victim!(plan, n)
        victim === nothing && throw(host_full_error(plan, n))
        try
            spill!(plan, victim)
        catch
            register_resident!(plan, victim)
            rethrow()
        end
    end
end

function take_host_block!(plan::ResidencyPlan, n::Integer)
    pool = plan.hostpool
    lock(pool.lock) do
        reusable = get!(() -> HostBlock[], pool.free, n)
        isempty(reusable) || return pop!(reusable)
        for i in eachindex(pool.slabs)
            pool.cursors[i] + n <= sizeof(pool.slabs[i]) || continue
            block = HostBlock(i, pool.cursors[i], n)
            pool.cursors[i] += n
            return block
        end
        grow_host_pool!(plan, n) || return nothing
        i = lastindex(pool.slabs)
        block = HostBlock(i, pool.cursors[i], n)
        pool.cursors[i] += n
        block
    end
end

# Slabs double in size so that page-locking cost is amortized without reserving
# the whole host_budget up front.
function grow_host_pool!(plan::ResidencyPlan, nbytes::Integer)
    pool = plan.hostpool
    remaining = pool.budget - pool.reserved
    remaining < nbytes && return false
    slab = alloc_host_slab(plan.backend, min(remaining, max(Int(nbytes), pool.reserved)))
    push!(pool.slabs, slab)
    push!(pool.cursors, 0)
    pool.reserved += sizeof(slab)
    true
end

# The panel that has been in host memory longest, is not staged for a step of a
# running sweep, and holds a block of exactly the size being asked for. The pool
# hands blocks back by size and cannot split them, so evicting a panel whose
# block is a different size would not help.
function claim_victim!(plan::ResidencyPlan, n::Integer)
    pool = plan.hostpool
    lock(pool.lock) do
        for (i, panel) in enumerate(pool.residents)
            panel.pins == 0 || continue
            panel.data isa HostStorage && panel.data.block.nbytes == n || continue
            deleteat!(pool.residents, i)
            return panel
        end
        nothing
    end
end

function host_full_error(plan::ResidencyPlan, nbytes::Integer)
    pool = plan.hostpool
    remaining = pool.budget - pool.reserved
    stranded = host_bytes_free(pool)
    detail = stranded == 0 ? "" : ", and the $(human_readable_bytes(stranded)) already released is only reusable by blocks of the size it was released at"
    if plan.scratch_dir === nothing
        return ArgumentError("host tier is full: a $(human_readable_bytes(nbytes)) panel block does not fit in the $(human_readable_bytes(remaining)) left of the $(human_readable_bytes(pool.budget)) host_budget$detail. Raise host_budget, reduce k or the panel width, or give the plan a scratch_dir so panels can live on disk")
    end
    ArgumentError("host tier is full and no panel of that size is free to spill: a $(human_readable_bytes(nbytes)) panel block does not fit in the $(human_readable_bytes(remaining)) left of the $(human_readable_bytes(pool.budget)) host_budget$detail. A panel sweep holds up to $(FEED_DEPTH + 2) steps at once, one panel per matrix per step, and gram and cholqr2! traverse rows and so hold every panel of their matrix. Raise host_budget to cover that many panels")
end

function register_resident!(plan::ResidencyPlan, panel)
    panel.home === nothing && return nothing
    pool = plan.hostpool
    lock(pool.lock) do
        push!(pool.residents, panel)
    end
    nothing
end

function unregister_resident!(plan::ResidencyPlan, panel)
    pool = plan.hostpool
    lock(pool.lock) do
        i = findfirst(resident -> resident === panel, pool.residents)
        i === nothing || deleteat!(pool.residents, i)
    end
    nothing
end

# A pinned panel is one a sweep has staged and not yet issued its copies for, so
# its host memory is spoken for. The hold that replaces the pin is the event
# those copies finish on, and a spill waits on that event.
function pin!(plan::ResidencyPlan, panel)
    lock(plan.hostpool.lock) do
        panel.pins += 1
    end
    nothing
end

function unpin!(plan::ResidencyPlan, panel, hold=nothing)
    lock(plan.hostpool.lock) do
        panel.pins -= 1
        hold === nothing || (panel.hold = hold)
    end
    nothing
end

function release_host_block!(plan::ResidencyPlan, block::HostBlock)
    pool = plan.hostpool
    lock(pool.lock) do
        push!(get!(() -> HostBlock[], pool.free, block.nbytes), block)
    end
    nothing
end

host_bytes_reserved(pool::HostPool) = pool.reserved
host_bytes_in_use(pool::HostPool) = sum(pool.cursors; init=0) - host_bytes_free(pool)
host_bytes_free(pool::HostPool) = sum(block.nbytes for blocks in values(pool.free) for block in blocks; init=0)

function alloc_host_storage!(plan::ResidencyPlan, ::Type{T}, dims::Dims{2}, blockbytes::Integer=prod(dims) * sizeof(T)) where {T}
    block = acquire_host_block!(plan, blockbytes)
    slab = plan.hostpool.slabs[block.slab]
    HostStorage(slab_matrix(slab, T, dims, block.offset), slab, block)
end
