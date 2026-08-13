# Internal vendor API. The CPU reference backend defined here is the only
# implementation that every user will use. CUDA, Metal, and other vendors are
# package extensions found in ext/*Ext.jl

abstract type DeviceBackend end

"""
    devicetype(backend, T) -> Type

Matrix type for a panel of eltype `T` when the panel is resident on `backend`.
"""
function devicetype end

"""
    alloc_device(backend, T, dims) -> array

Allocates device memory for a `dims`-shaped array of eltype `T`.
"""
function alloc_device end

"""
    alloc_host_slab(backend, nbytes) -> HostSlab

Allocates one host slab with `nbytes` usable bytes (page-locked if the backend
benefits from that). A slab must stay alive for as long as any of its views is
alive.
"""
function alloc_host_slab end

"""
    h2d!(dst, src, backend; queue=nothing) -> event

Copies host `src` into device `dst`, converting the eltype when the two differ.
When `queue=nothing` the copy is complete when the call returns, otherwise it
is enqueued on `queue` and is complete once the returned event is signalled.
"""
function h2d! end

"""
    d2h!(dst, src, backend; queue=nothing) -> event

Copies device `src` into host `dst`, converting the eltype of `dst` when it
disagrees with `src`. See [`h2d!`](@ref) for queue semantics.
"""
function d2h! end

"""
    make_queue(backend) -> queue

Creates an execution queue. Work enqueued on one queue runs in issue order. Work
on different queues is ordered through events.
"""
function make_queue end

"""
    make_sweep_queues(backend) -> (upload, compute, writeback)

Creates the three queues a sweep runs on, in one call. A sweep uses them on one
set of staging buffers and orders them with events alone, so a backend that
tracks which queue last touched a buffer can give the three queues whatever
shared state that tracking needs. The default is three independent queues.
"""
make_sweep_queues(backend::DeviceBackend) = (make_queue(backend), make_queue(backend), make_queue(backend))

"""
    submit!(backend, queue, work) -> event

Runs `work()` as part of `queue`. The reference backend defers the closure to a
task, while a real device backend runs it at once on the host and whatever it
launches lands on `queue`. Either way the work is ordered against everything
else on `queue`, and the returned event is signalled once the work has finished.
"""
function submit! end

"""
    record_event(backend, queue) -> event

Records an event that becomes signalled when everything enqueued on `queue` has
finished.
"""
function record_event end

"""
    wait_event(backend, queue, event)

Makes work on `queue` wait for `event` without blocking the caller.
"""
function wait_event end

"""
    sync_queue(backend, queue)

Blocks the calling task until `queue` has drained.
"""
function sync_queue end

"""
    sync_event(backend, event)

Blocks the calling task until `event` is signalled.
"""
function sync_event end

"""
    sync_all(backend)

Blocks the calling task until every queue of `backend` has drained.
"""
function sync_all end

"""
    ka_backend(backend) -> KernelAbstractions.Backend

The KernelAbstractions backend that goes with `backend`. The kernels Funicular
shares between vendors, in `kernels.jl`, reach the device through this rather
than through a vendor package.
"""
function ka_backend end

"""
    supports_eltype(backend, T) -> Bool

Whether `backend` can compute in eltype `T`.

Note that this returns false for Float64 and ComplexF64 on Metal, since those
types are not natively supported on Apple GPUs.
"""
supports_eltype(::DeviceBackend, ::Type) = true

"""
    transfers_are_real(backend) -> Bool

Whether host/device copies move data across an actual bus.

Note that this returns false for the CPU reference backend and for Metal's
unified memory, since there the CPU and GPU share the same memory.
"""
transfers_are_real(::DeviceBackend) = true

const SLAB_ALIGN = 64

align_up(n::Integer, a::Integer) = cld(n, a) * a

struct HostSlab{O}
    owner::O
    ptr::Ptr{UInt8}
    nbytes::Int
end

# Backends allocate nbytes + SLAB_ALIGN and hand the raw base pointer here
function aligned_slab(owner, base::Ptr{UInt8}, nbytes::Integer)
    pad = Int(mod(-reinterpret(UInt, base), UInt(SLAB_ALIGN)))
    HostSlab(owner, base + pad, Int(nbytes))
end

Base.sizeof(slab::HostSlab) = slab.nbytes

function slab_matrix(slab::HostSlab, ::Type{T}, dims::Dims{2}, offset::Integer) where {T}
    nbytes = prod(dims) * sizeof(T)
    offset >= 0 || throw(ArgumentError("slab offset must be nonnegative, got $offset"))
    offset + nbytes <= slab.nbytes || throw(ArgumentError("a $(human_readable_bytes(nbytes)) view at offset $offset does not fit in a $(human_readable_bytes(slab.nbytes)) host slab"))
    unsafe_wrap(Array, Ptr{T}(slab.ptr + offset), dims)
end

"""
    Funicular.cuda_backend()

The CUDA device backend, available once the CUDA package has been loaded. It is
the only backend where the tiers, the streams and the events are real: host
slabs are page-locked, copies are asynchronous on explicit streams, and the
pipeline overlaps them against compute.
"""
function cuda_backend(args...; kwargs...)
    throw(ArgumentError("the CUDA backend arrives with the CUDA package, which has to be loaded first"))
end

"""
    Funicular.metal_backend(; jitter=0.0)

The Metal device backend, available once the Metal.jl package has been loaded.
It exists to run the logic layer against a real GPU array type: it computes in
ComplexF32 and Float32 only, and its unified memory makes host/device copies a
formality rather than a transfer, so `transfers_are_real` is false and no timing
conclusion follows from it.

# Arguments
- `jitter=0.0`: Maximum sleep in seconds injected before each queued operation
"""
function metal_backend(args...; kwargs...)
    throw(ArgumentError("the Metal backend arrives with the Metal.jl package, which has to be loaded first. It is available on Apple silicon only"))
end

# A queue whose ordering comes from a chain of tasks rather than from a device
# stream. The reference backend has no asynchrony to gain, and a chain of tasks
# turns a missing wait_event into a schedule-dependent failure. The event token
# is shared with any backend that orders its work through tasks. The Metal
# extension does that on top of a queue of its own.
mutable struct TaskQueue
    tail::Task
end

struct TaskEvent
    task::Union{Task,Nothing}
end

const TASK_QUEUES = WeakRef[]
const TASK_QUEUES_LOCK = ReentrantLock()

function make_task_queue()
    queue = TaskQueue(Threads.@spawn nothing)
    lock(TASK_QUEUES_LOCK) do
        filter!(ref -> ref.value !== nothing, TASK_QUEUES)
        push!(TASK_QUEUES, WeakRef(queue))
    end
    queue
end

function submit_task!(queue::TaskQueue, jitter::Real, work)
    previous = queue.tail
    task = Threads.@spawn begin
        wait(previous)
        yield()
        jitter > 0 && sleep(jitter * rand())
        work()
    end
    queue.tail = task
    TaskEvent(task)
end

record_event(::DeviceBackend, queue::TaskQueue) = TaskEvent(queue.tail)

function wait_event(backend::DeviceBackend, queue::TaskQueue, event::TaskEvent)
    event.task === nothing && return nothing
    submit!(backend, queue, () -> wait(event.task))
    nothing
end

function sync_queue(::DeviceBackend, queue::TaskQueue)
    wait(queue.tail)
    nothing
end

function sync_event(::DeviceBackend, event::TaskEvent)
    event.task === nothing || wait(event.task)
    nothing
end

function sync_task_queues(backend::DeviceBackend)
    queues = lock(TASK_QUEUES_LOCK) do
        TaskQueue[ref.value for ref in TASK_QUEUES if ref.value !== nothing]
    end
    for queue in queues
        sync_queue(backend, queue)
    end
    nothing
end

"""
    CPUBackend(; jitter=0.0)

Reference backend where the "device" is a plain `Array`. Queues are chains of
tasks instead of streams so that ordering mistakes show up as failures.
`jitter` is the maximum sleep in seconds injected before each queued operation,
so a positive value makes the schedule vary between runs and helps test
scheduling. Run such tests multiple times with multiple threads to increase the
chance of catching race conditions.
"""
struct CPUBackend <: DeviceBackend
    jitter::Float64
end

function CPUBackend(; jitter::Real=0.0)
    jitter >= 0 || throw(ArgumentError("jitter must be nonnegative, got $jitter"))
    CPUBackend(Float64(jitter))
end

devicetype(::CPUBackend, ::Type{T}) where {T} = Matrix{T}
alloc_device(::CPUBackend, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)
ka_backend(::CPUBackend) = KernelAbstractions.CPU()
transfers_are_real(::CPUBackend) = false

function alloc_host_slab(::CPUBackend, nbytes::Integer)
    owner = Vector{UInt8}(undef, nbytes + SLAB_ALIGN)
    aligned_slab(owner, pointer(owner), nbytes)
end

make_queue(::CPUBackend) = make_task_queue()
submit!(backend::CPUBackend, queue::TaskQueue, work) = submit_task!(queue, backend.jitter, work)
sync_all(backend::CPUBackend) = sync_task_queues(backend)

h2d!(dst, src, backend::CPUBackend; queue=nothing) = cpu_copy!(dst, src, backend, queue)
d2h!(dst, src, backend::CPUBackend; queue=nothing) = cpu_copy!(dst, src, backend, queue)

function cpu_copy!(dst, src, backend::CPUBackend, queue)
    size(dst) == size(src) || throw(DimensionMismatch("copy between a $(size(dst)) destination and a $(size(src)) source"))
    queue === nothing || return submit!(backend, queue, () -> copyto!(dst, src))
    copyto!(dst, src)
    TaskEvent(nothing)
end

function human_readable_bytes(n::Integer)
    n < 1024 && return "$n B"
    units = ("KiB", "MiB", "GiB", "TiB")
    x = float(n) / 1024
    i = 1
    while x >= 1024 && i < length(units)
        x /= 1024
        i += 1
    end
    string(round(x, digits=2), " ", units[i])
end
