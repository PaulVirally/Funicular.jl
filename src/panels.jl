mutable struct Panel{T}
    data::Any
    tier::Tier
    # A DiskHome, a GhostHome, or nothing for a panel whose only copy is the
    # host one. Untyped for the same reason `data` is: it is read once per
    # panel-sized transfer rather than once per element.
    home::Any
    width::Int
    dirty::Bool
    epoch::Int
    pins::Int
    hold::Any
end

panelstorage(panel::Panel) = storage(panel.data)
isresident(panel::Panel) = panel.data !== nothing
ondisk(panel::Panel) = panel.tier isa DiskTier
iscold(panel::Panel) = panel.tier isa Union{DiskTier,GhostTier}
isreleased(panel::Panel) = panel.data === nothing && !iscold(panel)
storage_eltype(panel::Panel) = eltype(panelstorage(panel))

"""
    PanelMatrix{T}(undef, N, k; plan, w=nothing)
    PanelMatrix(A::AbstractMatrix; plan, w=nothing)

Tall-skinny `N × k` matrix of compute eltype `T`, partitioned into column panels
of width `w` with a ragged last panel when `w` does not divide `k`. Each panel is
resident on one tier at a time and moves between tiers under the plan's control.
Panels start out on the host tier.

This is deliberately not an `AbstractMatrix` and has no `getindex`: a scalar
read of a cold panel would cost a whole panel of traffic. Collect it with
`Matrix(pm)` when the result fits in host memory, and otherwise work one panel
at a time.

A plan with a `scratch_dir` gives the matrix a file of its own at construction,
and panels spill to it whenever the host tier runs out of room. Reload a saved
matrix with [`load`](@ref) instead of constructing one.

# Arguments
- `N::Integer`: Number of rows
- `k::Integer`: Number of columns
- `A::AbstractMatrix`: Dense matrix whose columns are copied into the new matrix panel by panel
- `plan::ResidencyPlan`: The plan that owns the tiers the panels live on
- `w=nothing`: Panel width for this matrix, or `nothing` to take the plan's choice from the device budget (see [`ResidencyPlan`](@ref))
"""
struct PanelMatrix{T,P<:ResidencyPlan}
    N::Int
    k::Int
    w::Int
    panels::Vector{Panel{T}}
    plan::P
    store::Union{Nothing,PanelStore}
end

function PanelMatrix{T}(::UndefInitializer, N::Integer, k::Integer; plan::ResidencyPlan, w=nothing) where {T}
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    isconcretetype(T) || throw(ArgumentError("PanelMatrix eltype must be concrete, got $T"))
    width = resolve_panel_width(plan, N, k, T; override=w)
    S = host_storage_eltype(plan, T)
    D = disk_storage_eltype(plan, T)
    store = plan.scratch_dir === nothing ? nothing : scratch_store(plan, Int(N), Int(k), width, D, T)
    # Every panel takes a block sized for the nominal width, including the ragged
    # one, so that a spilled panel frees a block the next one can use. The pool
    # hands blocks back by size and cannot split them.
    blockbytes = panel_bytes(N, width, S)
    panels = Panel{T}[]
    try
        for j in 1:cld(Int(k), width)
            cols = ((j - 1) * width + 1):min(j * width, Int(k))
            home = store === nothing ? nothing : DiskHome(store, cols)
            panel = Panel{T}(nothing, HostTier(), home, length(cols), false, 0, 0, nothing)
            push!(panels, panel)
            panel.data = alloc_host_storage!(plan, S, (Int(N), length(cols)), blockbytes)
            register_resident!(plan, panel)
        end
    catch
        for panel in panels
            release!(plan, panel)
        end
        store === nothing || close_store!(store)
        rethrow()
    end
    PanelMatrix{T,typeof(plan)}(Int(N), Int(k), width, panels, plan, store)
end

function PanelMatrix(A::AbstractMatrix{T}; plan::ResidencyPlan, w=nothing) where {T}
    pm = PanelMatrix{T}(undef, size(A, 1), size(A, 2); plan=plan, w=w)
    copyto!(pm, A)
    pm
end

function host_storage_eltype(plan::ResidencyPlan, ::Type{T}) where {T}
    plan.host_eltype === nothing && return T
    S = plan.host_eltype
    if T <: Complex && !(S <: Complex)
        throw(ArgumentError("host_eltype $S cannot hold the imaginary part of a $T panel. Narrowing a complex compute eltype means a narrower complex one, so use host_eltype = Complex{$S}"))
    end
    S
end

# A panel crosses between the host and the disk tier unconverted, so the two
# tiers store the same eltype. Narrowing one and not the other would put a
# conversion on every spill, and it would not save anything that the narrower
# tier does not already save.
function disk_storage_eltype(plan::ResidencyPlan, ::Type{T}) where {T}
    S = host_storage_eltype(plan, T)
    plan.disk_eltype === nothing && return S
    plan.disk_eltype === S || throw(ArgumentError("disk_eltype $(plan.disk_eltype) and the host tier's $S disagree, and a panel moves between the two without conversion. Set host_eltype to $(plan.disk_eltype) as well, or leave disk_eltype unset"))
    S
end

# A matrix whose host tier stores a narrower eltype than it computes in needs a
# second set of device buffers, in the storage eltype, for the narrow copy to
# land in before the device widens it. When the two eltypes agree, which is the
# default, there is nothing to check out.
function checkout_staging_buffers!(plan::ResidencyPlan, pm::PanelMatrix{T}, dims::Dims{2}, count::Integer) where {T}
    S = host_storage_eltype(plan, T)
    S === T && return nothing
    checkout_device_buffers!(plan, S, dims, count)
end

Base.size(pm::PanelMatrix) = (pm.N, pm.k)
function Base.size(pm::PanelMatrix, d::Integer)
    d >= 1 || throw(ArgumentError("dimension must be positive, got $d"))
    d == 1 ? pm.N : d == 2 ? pm.k : 1
end
Base.eltype(::PanelMatrix{T}) where {T} = T
Base.eltype(::Type{<:PanelMatrix{T}}) where {T} = T

"""
    npanels(pm) -> Int

Number of column panels, `cld(k, panelwidth(pm))`.
"""
npanels(pm::PanelMatrix) = length(pm.panels)

"""
    panelwidth(pm) -> Int
    panelwidth(pm, j) -> Int

Nominal panel width of `pm`, or the width of panel `j`. The last panel is
narrower than the nominal width when `panelwidth(pm)` does not divide `k`.
"""
panelwidth(pm::PanelMatrix) = pm.w

function panelwidth(pm::PanelMatrix, j::Integer)
    checkpanel(pm, j)
    pm.panels[j].width
end

"""
    panelrange(pm, j) -> UnitRange{Int}

Columns of the full `N × k` matrix that panel `j` holds.
"""
function panelrange(pm::PanelMatrix, j::Integer)
    checkpanel(pm, j)
    firstcol = (Int(j) - 1) * pm.w + 1
    firstcol:(firstcol + pm.panels[j].width - 1)
end

function checkpanel(pm::PanelMatrix, j::Integer)
    1 <= j <= npanels(pm) || throw(BoundsError(pm.panels, j))
    nothing
end

function Base.show(io::IO, pm::PanelMatrix{T}) where {T}
    print(io, "PanelMatrix{", T, "}(", pm.N, "×", pm.k, ", w = ", pm.w, ", ", npanels(pm), " panels")
    if all(isreleased, pm.panels)
        print(io, ", released")
    else
        isghost(pm) && print(io, ", ghost")
        cold = count(ondisk, pm.panels)
        cold == 0 || print(io, ", ", cold, " on disk")
        S = host_storage_eltype(pm.plan, T)
        S === T || print(io, ", host storage ", S)
    end
    print(io, ")")
end

function assert_conformal(X::PanelMatrix, Y::PanelMatrix)
    (X.N, X.k, X.w) == (Y.N, Y.k, Y.w) && return nothing
    throw(ArgumentError("panel matrices must share their (N, k, w) partitioning, got ($(X.N), $(X.k), $(X.w)) and ($(Y.N), $(Y.k), $(Y.w)). Repartitioning is not supported"))
end

function Base.copyto!(pm::PanelMatrix, A::AbstractMatrix)
    size(A) == size(pm) || throw(DimensionMismatch("source is $(size(A)) but the PanelMatrix is $(size(pm))"))
    assert_writable(pm)
    for (j, panel) in enumerate(pm.panels)
        stage_host!(panel, pm.plan)
        try
            copyto!(panelstorage(panel), view(A, :, panelrange(pm, j)))
            panel.dirty = true
            panel.epoch += 1
        finally
            unpin!(pm.plan, panel)
        end
    end
    pm
end

function Base.Matrix(pm::PanelMatrix{T}; max_bytes::Real=Sys.free_memory() ÷ 2) where {T}
    nbytes = pm.N * pm.k * sizeof(T)
    nbytes <= max_bytes || throw(ArgumentError("collecting a $(pm.N)×$(pm.k) $T PanelMatrix needs $(human_readable_bytes(nbytes)) of host memory but max_bytes is $(human_readable_bytes(floor(Int, max_bytes))). Work panel by panel instead, or raise max_bytes if the memory really is there"))
    A = Matrix{T}(undef, pm.N, pm.k)
    for (j, panel) in enumerate(pm.panels)
        stage_host!(panel, pm.plan)
        try
            copyto!(view(A, :, panelrange(pm, j)), panelstorage(panel))
        finally
            unpin!(pm.plan, panel)
        end
    end
    A
end

# Precision fusion: when the host tier stores a narrower eltype than the panels
# compute in, only the narrow bytes cross the bus and the conversion runs as a
# device kernel, into or out of a staging buffer the sweep checked out of the
# pool. Both halves are submitted to the same queue, so they run in order.
function convert_staged!(dst, src, plan::ResidencyPlan, queue, event)
    backend = plan.backend
    queue === nothing || return submit!(backend, queue, () -> convert_copy!(backend, dst, src))
    convert_copy!(backend, dst, src)
    KernelAbstractions.synchronize(ka_backend(backend))
    event
end

materialize!(dst::AbstractMatrix, panel::Panel, plan::ResidencyPlan; queue=nothing, via=nothing) = materialize!(dst, panel, panel.tier, plan; queue=queue, via=via)

function materialize!(dst::AbstractMatrix, panel::Panel, ::HostTier, plan::ResidencyPlan; queue=nothing, via=nothing)
    isresident(panel) || throw(ArgumentError("this panel has no host memory behind it: it was released by free!, or it went to a colder tier and nothing staged it back. Funicular.prefetch! brings one panel up, and a sweep brings up every panel of a step before it copies any of them"))
    src = panelstorage(panel)
    size(dst) == size(src) || throw(DimensionMismatch("device buffer is $(size(dst)) but the host panel is $(size(src)); a ragged last panel needs a view of the staging buffer"))
    via === nothing && return h2d!(dst, src, plan.backend; queue=queue)
    event = h2d!(via, src, plan.backend; queue=queue)
    convert_staged!(dst, via, plan, queue, event)
end

writeback!(panel::Panel, src::AbstractMatrix, plan::ResidencyPlan; queue=nothing, via=nothing) = writeback!(panel, src, panel.tier, plan; queue=queue, via=via)

function writeback!(panel::Panel, src::AbstractMatrix, ::HostTier, plan::ResidencyPlan; queue=nothing, via=nothing)
    isresident(panel) || throw(ArgumentError("this panel has no host memory behind it: it was released by free!, or it went to a colder tier and nothing staged it back. Funicular.prefetch! brings one panel up, and a sweep brings up every panel of a step before it copies any of them"))
    dst = panelstorage(panel)
    size(dst) == size(src) || throw(DimensionMismatch("device buffer is $(size(src)) but the host panel is $(size(dst)). A ragged last panel needs a view of the staging buffer"))
    if via !== nothing
        convert_staged!(via, src, plan, queue, nothing)
        src = via
    end
    event = d2h!(dst, src, plan.backend; queue=queue)
    panel.dirty = true
    panel.epoch += 1
    event
end

materialize!(dst::AbstractMatrix, panel::Panel, rows::AbstractUnitRange, plan::ResidencyPlan; queue=nothing) = materialize!(dst, panel, rows, panel.tier, plan; queue=queue)

function materialize!(dst::AbstractMatrix, panel::Panel, rows::AbstractUnitRange, ::HostTier, plan::ResidencyPlan; queue=nothing)
    isresident(panel) || throw(ArgumentError("this panel has no host memory behind it: it was released by free!, or it went to a colder tier and nothing staged it back. Funicular.prefetch! brings one panel up, and a sweep brings up every panel of a step before it copies any of them"))
    src = view(panelstorage(panel), rows, :)
    size(dst) == size(src) || throw(DimensionMismatch("device buffer block is $(size(dst)) but rows $rows of the host panel are $(size(src))"))
    h2d!(dst, src, plan.backend; queue=queue)
end

writeback!(panel::Panel, src::AbstractMatrix, rows::AbstractUnitRange, plan::ResidencyPlan; queue=nothing) = writeback!(panel, src, rows, panel.tier, plan; queue=queue)

function writeback!(panel::Panel, src::AbstractMatrix, rows::AbstractUnitRange, ::HostTier, plan::ResidencyPlan; queue=nothing)
    isresident(panel) || throw(ArgumentError("this panel has no host memory behind it: it was released by free!, or it went to a colder tier and nothing staged it back. Funicular.prefetch! brings one panel up, and a sweep brings up every panel of a step before it copies any of them"))
    dst = view(panelstorage(panel), rows, :)
    size(dst) == size(src) || throw(DimensionMismatch("device buffer block is $(size(src)) but rows $rows of the host panel are $(size(dst))"))
    event = d2h!(dst, src, plan.backend; queue=queue)
    panel.dirty = true
    panel.epoch += 1
    event
end

# A sweep never reaches these: its producer task brings every panel of a step to
# the host tier before the pipeline issues a copy for it. The producer exists so
# that reading a cold panel does not put a disk read and a host allocation on
# whichever task is issuing the pipeline.
cold_panel_error() = throw(ArgumentError("this copy reached a panel that is not in host memory: it is still on disk, or it is a ghost panel that has not been generated yet. Bring it up with Funicular.prefetch! first, or let a sweep do it: a sweep stages every panel of a step before it issues a copy for it"))

materialize!(::AbstractMatrix, ::Panel, ::DiskTier, ::ResidencyPlan; kwargs...) = cold_panel_error()
writeback!(::Panel, ::AbstractMatrix, ::DiskTier, ::ResidencyPlan; kwargs...) = cold_panel_error()
materialize!(::AbstractMatrix, ::Panel, ::AbstractUnitRange, ::DiskTier, ::ResidencyPlan; kwargs...) = cold_panel_error()
writeback!(::Panel, ::AbstractMatrix, ::AbstractUnitRange, ::DiskTier, ::ResidencyPlan; kwargs...) = cold_panel_error()

# A row block of the whole matrix is a slice of every panel, gathered into one
# buffer whose columns are laid out as the matrix's are. Every panel has to be
# host resident for this. That comes for free on the host tier, but not on the
# disk tier: a row sweep over a matrix backed by disk needs the whole matrix to
# fit in host_budget at once.
function materialize_rows!(dst::AbstractMatrix, pm::PanelMatrix, rows::AbstractUnitRange; queue=nothing, via=nothing)
    landing = via === nothing ? dst : via
    event = nothing
    for j in 1:npanels(pm)
        event = materialize!(view(landing, :, panelrange(pm, j)), pm.panels[j], rows, pm.plan; queue=queue)
    end
    via === nothing && return event
    convert_staged!(dst, via, pm.plan, queue, event)
end

function writeback_rows!(pm::PanelMatrix, src::AbstractMatrix, rows::AbstractUnitRange; queue=nothing, via=nothing)
    leaving = src
    if via !== nothing
        convert_staged!(via, src, pm.plan, queue, nothing)
        leaving = via
    end
    event = nothing
    for j in 1:npanels(pm)
        event = writeback!(pm.panels[j], view(leaving, :, panelrange(pm, j)), rows, pm.plan; queue=queue)
    end
    event
end

"""
    Funicular.prefetch!(pm, j) -> Panel

Brings panel `j` of `pm` to the host tier and leaves it there, reading it from
disk if that is where it was. A sweep does this for itself, one step ahead of the
pipeline, so `prefetch!` is for code that works outside one.
"""
function prefetch!(pm::PanelMatrix, j::Integer)
    checkpanel(pm, j)
    panel = pm.panels[j]
    stage_host!(panel, pm.plan)
    unpin!(pm.plan, panel)
    panel
end

# Staging a panel means getting its data into host memory and holding it there
# until the caller says otherwise. A host tier panel is already there. The disk
# tier reads panel j+1 here, on the producer task, while the pipeline works on
# panel j.
stage_host!(panel::Panel, plan::ResidencyPlan) = stage_host!(panel, panel.tier, plan)

function stage_host!(panel::Panel, ::HostTier, plan::ResidencyPlan)
    isresident(panel) || throw(ArgumentError("panel has no storage: it was released"))
    pin!(plan, panel)
    panel
end

function stage_host!(panel::Panel{T}, ::DiskTier, plan::ResidencyPlan) where {T}
    home = panel.home
    pin!(plan, panel)
    store = home.store
    S = host_storage_eltype(plan, T)
    dims = (store.N, length(home.cols))
    storage = alloc_host_storage!(plan, S, dims, panel_bytes(store.N, store.w, S))
    try
        read_panel!(home, storage.array)
    catch
        release_host_block!(plan, storage.block)
        unpin!(plan, panel)
        rethrow()
    end
    panel.data = storage
    panel.tier = HostTier()
    register_resident!(plan, panel)
    panel
end

# Give the host block back to the pool and put the panel back on the tier it
# came from, writing it out first if it has changed and has somewhere to go. The
# wait is for whatever copy last read or wrote this panel's host memory; it runs
# on the task that needed the room, which is the producer rather than the task
# issuing the pipeline.
function spill!(plan::ResidencyPlan, panel::Panel)
    hold = panel.hold
    hold === nothing || sync_event(plan.backend, hold)
    panel.hold = nothing
    cool!(panel.home, panel)
    release!(plan, panel)
    panel.tier = coldtier(panel.home)
    nothing
end

function cool!(home::DiskHome, panel::Panel)
    panel.dirty || return nothing
    write_panel!(home, panelstorage(panel))
    panel.dirty = false
    nothing
end

coldtier(::DiskHome) = DiskTier()

function evict!(panel::Panel, plan::ResidencyPlan)
    iscold(panel) && return nothing
    panel.home === nothing && throw(ArgumentError("panel lives on the host tier and the plan has no colder tier (scratch_dir = $(repr(plan.scratch_dir))), so there is nowhere to evict it to. Give the plan a scratch_dir, or save the matrix and free! it if you want its memory back"))
    panel.pins == 0 || throw(ArgumentError("panel is staged for a sweep that is still running, so its host memory is not free to give back. Evict it once the sweep has returned"))
    spill!(plan, panel)
end

function release!(plan::ResidencyPlan, panel::Panel)
    unregister_resident!(plan, panel)
    panel.data isa HostStorage && release_host_block!(plan, panel.data.block)
    panel.data = nothing
    nothing
end

function free!(pm::PanelMatrix)
    for panel in pm.panels
        release!(pm.plan, panel)
        panel.tier = HostTier()
        panel.home = nothing
        panel.hold = nothing
        panel.pins = 0
    end
    pm.store === nothing || close_store!(pm.store)
    nothing
end
