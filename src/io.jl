# The disk tier's surface. This file holds only layout and bookkeeping. The file
# format and the calls that touch it belong to the HDF5 extension, and are
# reached through the five functions below.

"""
    open_panel_store(path, mode; N, k, w, stored, computed) -> handle

Opens or creates the file behind a store and returns the extension's handle
for it.

Mode `"w"` creates a file laid out for an `N × k` matrix of eltype `stored`,
chunked one panel of `w` columns at a time. Modes `"r"` and `"r+"` open an
existing file, in which case the keywords are ignored and the layout is whatever
the file says it is.
"""
function open_panel_store(args...; kwargs...)
    throw(ArgumentError("the disk tier arrives with the HDF5.jl package, which has to be loaded first. Without it a ResidencyPlan cannot take a scratch_dir and panels have nowhere colder than the host tier to go"))
end

"""
    close_panel_store(handle)

Releases the file behind a store.
"""
function close_panel_store end

"""
    panel_store_layout(handle) -> (N, k, w, stored, computed)

Returns what an opened file records about the matrix it holds.
"""
function panel_store_layout end

"""
    read_panel_region!(handle, dst, cols)

Reads columns `cols` of the stored matrix into `dst`. The destination is a host
array of the store's own eltype whose shape is exactly that of those columns.
"""
function read_panel_region! end

"""
    write_panel_region!(handle, src, cols)

Writes `src` into columns `cols` of the stored matrix.
"""
function write_panel_region! end

"""
    Funicular.diskarray(path) -> AbstractDiskArray

Returns a file Funicular wrote as an `N × k` DiskArrays.jl array with one chunk
per panel.

This is for interop with the JuliaIO ecosystem rather than a path Funicular
itself takes: reading a panel this way goes through generic chunked-array code
instead of through the pipeline. Both DiskArrays.jl and HDF5.jl have to be
loaded, and the array should be closed when you are done with it.
"""
function diskarray(args...; kwargs...)
    throw(ArgumentError("Funicular.diskarray needs both the DiskArrays and HDF5 packages loaded"))
end

# HDF5's thread safety is a build detail this package does not depend on, so
# every call into it in this process is serialized through this lock. A producer
# task holds the lock only for the length of one panel-sized read, and never
# while it waits on the pipeline.
const DISK_LOCK = ReentrantLock()

const SCRATCH_COUNTER = Threads.Atomic{Int}(0)

# What the cold tier can hold. Half precision is missing because HDF5 has no
# datatype for it, so a plan that narrows its storage that far has nowhere to
# put a panel.
const STORAGE_ELTYPES = Dict{String,DataType}(string(T) => T for T in
    (Float32, Float64, ComplexF32, ComplexF64))

function storage_eltype_named(name::AbstractString)
    haskey(STORAGE_ELTYPES, name) && return STORAGE_ELTYPES[name]
    throw(ArgumentError("this file stores $name, which is not an eltype Funicular reads. Known eltypes are $(join(sort(collect(keys(STORAGE_ELTYPES))), ", "))"))
end

function open_store(path::AbstractString, mode::AbstractString; N::Integer=0, k::Integer=0, w::Integer=0, stored::DataType=ComplexF64, computed::DataType=ComplexF64, scratch::Bool=false)
    mode == "w" && !haskey(STORAGE_ELTYPES, string(stored)) && throw(ArgumentError("the disk tier cannot store $stored. It holds $(join(sort(collect(keys(STORAGE_ELTYPES))), ", ")); narrow the cold tiers to one of those or leave them at the compute eltype"))
    handle = lock(DISK_LOCK) do
        open_panel_store(path, mode; N=N, k=k, w=w, stored=stored, computed=computed)
    end
    layout = lock(DISK_LOCK) do
        panel_store_layout(handle)
    end
    PanelStore(String(path), handle, layout..., mode == "r", scratch, 0, 0)
end

function scratch_store(plan::ResidencyPlan, N::Integer, k::Integer, w::Integer, ::Type{S}, ::Type{T}) where {S,T}
    mkpath(plan.scratch_dir)
    serial = Threads.atomic_add!(SCRATCH_COUNTER, 1)
    path = joinpath(plan.scratch_dir, string("funicular-", getpid(), "-", serial, ".h5"))
    open_store(path, "w"; N=N, k=k, w=w, stored=S, computed=T, scratch=true)
end

function close_store!(store::PanelStore)
    store.handle === nothing && return nothing
    handle = store.handle
    store.handle = nothing
    lock(DISK_LOCK) do
        close_panel_store(handle)
    end
    store.scratch && rm(store.path; force=true)
    nothing
end

function read_panel!(home::DiskHome, dst::AbstractMatrix)
    store = home.store
    store.handle === nothing && throw(ArgumentError("the store at $(repr(store.path)) is closed, so the panel that lived there cannot be read back. It was closed by free! or by save"))
    lock(DISK_LOCK) do
        read_panel_region!(store.handle, dst, home.cols)
        store.reads += 1
    end
    dst
end

function write_panel!(home::DiskHome, src::AbstractMatrix)
    store = home.store
    store.handle === nothing && throw(ArgumentError("the store at $(repr(store.path)) is closed, so the panel cannot be written back to it"))
    store.readonly && throw(ArgumentError("the store at $(repr(store.path)) was loaded read only and this panel has been modified, so there is nowhere to put it. Load the matrix without readonly = true if the operation is meant to reach the file"))
    lock(DISK_LOCK) do
        write_panel_region!(store.handle, src, home.cols)
        store.writes += 1
    end
    src
end

"""
    disk_reads(pm) -> Int
    disk_writes(pm) -> Int

Returns how many panel reads and writes the matrix's store has served since it
was opened. Both are zero for a matrix that has never been colder than the
host tier.
"""
disk_reads(pm::PanelMatrix) = pm.store === nothing ? 0 : pm.store.reads
disk_writes(pm::PanelMatrix) = pm.store === nothing ? 0 : pm.store.writes

"""
    save(pm::PanelMatrix, path) -> path

Writes `pm` to `path` as one HDF5 file holding a single chunked dataset with one
chunk per panel, in the storage eltype the plan's cold tiers use. The matrix is
left where it was: its panels keep whatever residency they had, and the file is
closed before this returns.

Reload the file with [`load`](@ref), or read it from generic chunked-array code
with `Funicular.diskarray(path)`.
"""
function save(pm::PanelMatrix{T}, path::AbstractString) where {T}
    S = disk_storage_eltype(pm.plan, T)
    store = open_store(path, "w"; N=pm.N, k=pm.k, w=pm.w, stored=S, computed=T)
    try
        for (j, panel) in enumerate(pm.panels)
            stage_host!(panel, pm.plan)
            try
                write_panel!(DiskHome(store, panelrange(pm, j)), panelstorage(panel))
            finally
                unpin!(pm.plan, panel)
            end
        end
    finally
        close_store!(store)
    end
    path
end

"""
    load(PanelMatrix, path; plan, readonly=false) -> PanelMatrix
    load(PanelMatrix{T}, path; plan, readonly=false) -> PanelMatrix{T}

Opens a file written by [`save`](@ref) as a `PanelMatrix` whose panels start out
on disk and stream up as they are needed. The compute eltype comes from the file
unless the first form is given one. The panel width always comes from the file,
since it is the file's chunking.

The file is the matrix's cold tier rather than a copy of it: a panel that an
operation modifies is written back into it. Pass `readonly=true` to open the
file for reading only. Any operation that would modify a panel then raises an
error instead of silently editing the file.

The store is closed by `Funicular.free!(pm)`.
"""
load(::Type{PanelMatrix}, path::AbstractString; plan::ResidencyPlan, readonly::Bool=false) = load_store(nothing, path, plan, readonly)
load(::Type{PanelMatrix{T}}, path::AbstractString; plan::ResidencyPlan, readonly::Bool=false) where {T} = load_store(T, path, plan, readonly)

function load_store(compute, path::AbstractString, plan::ResidencyPlan, readonly::Bool)
    isfile(path) || throw(ArgumentError("there is no file at $(repr(path)) to load a PanelMatrix from"))
    store = open_store(path, readonly ? "r" : "r+")
    try
        T = compute === nothing ? store.computed : compute
        S = host_storage_eltype(plan, T)
        S === store.stored || throw(ArgumentError("the file at $(repr(path)) stores $(store.stored) but this plan's cold tiers hold $S. Give the plan host_eltype = $(store.stored), or load the file with a plan whose storage eltype matches it"))
        # The file's chunking fixes the panel width, so this is only here to
        # refuse a plan whose device budget cannot stage a panel that wide.
        resolve_panel_width(plan, store.N, store.k, T; override=store.w)
        panels = Panel{T}[]
        for j in 1:cld(store.k, store.w)
            cols = ((j - 1) * store.w + 1):min(j * store.w, store.k)
            push!(panels, Panel{T}(nothing, DiskTier(), DiskHome(store, cols), length(cols), false, 0, 0, nothing))
        end
        PanelMatrix{T,typeof(plan)}(store.N, store.k, store.w, panels, plan, store)
    catch
        close_store!(store)
        rethrow()
    end
end
