module FunicularMetalExt

# Metal.jl exports a MetalBackend of its own, the KernelAbstractions one, so it
# is imported rather than brought into scope. The MetalBackend defined below is
# Funicular's own, and Metal's is written out in full wherever it is needed.
import Metal

import Funicular: alloc_device, alloc_host_slab, d2h!, devicetype,
                  gram_accumulate!, h2d!, ka_backend, make_queue, metal_backend,
                  rdiv_upper!, record_event, submit!, sync_all, sync_queue,
                  supports_eltype, transfers_are_real, wait_event

using Funicular: DeviceBackend, HostSlab, TaskEvent, convert_copy!,
                 gram_accumulate_ka!, rdiv_upper_ka!

"""
    MetalBackend(; jitter=0.0)

Device backend on an Apple GPU. Panels live in `MtlArray`s of private storage,
so anything that falls back to reading device memory an element at a time fails
loudly rather than quietly running on the CPU.

Memory is unified. A host slab is not page locked but wrapped as a shared
`MtlArray`, so a transfer is a kernel over memory the two processors already
share. That makes the backend a check on the logic and on the array types rather
than on the pipeline, so `transfers_are_real` is false. Apple GPUs have no
double precision either, so `supports_eltype` refuses Float64 and ComplexF64.

`jitter` means what it does on [`CPUBackend`](@ref): seconds of random delay
before each queued operation, to vary the schedule between runs.
"""
struct MetalBackend <: DeviceBackend
    jitter::Float64
end

function MetalBackend(; jitter::Real=0.0)
    jitter >= 0 || throw(ArgumentError("jitter must be nonnegative, got $jitter"))
    MetalBackend(Float64(jitter))
end

metal_backend(; kwargs...) = MetalBackend(; kwargs...)

# The host slabs, weakly held so that a plan going out of scope takes its memory
# with it. Lookup is by address: a transfer sees only the panel's host array and
# has to find the shared buffer that array sits in.
mutable struct MetalSlab
    owner::Vector{UInt8}
    gpu::Metal.MtlArray{UInt8,1}
    base::UInt
    nbytes::Int
end

const SLABS = WeakRef[]
const SLABS_LOCK = ReentrantLock()

function register_slab!(slab::MetalSlab)
    lock(SLABS_LOCK) do
        filter!(ref -> ref.value !== nothing, SLABS)
        push!(SLABS, WeakRef(slab))
    end
    slab
end

function slab_containing(ptr::Ptr)
    address = reinterpret(UInt, ptr)
    lock(SLABS_LOCK) do
        for ref in SLABS
            slab = ref.value
            slab isa MetalSlab || continue
            slab.base <= address < slab.base + slab.nbytes && return slab
        end
        nothing
    end
end

devicetype(::MetalBackend, ::Type{T}) where {T} = Metal.MtlMatrix{T}
ka_backend(::MetalBackend) = Metal.MetalBackend()
transfers_are_real(::MetalBackend) = false
supports_eltype(::MetalBackend, ::Type{T}) where {T<:Number} = real(T) !== Float64
supports_eltype(::MetalBackend, ::Type) = false

function alloc_device(backend::MetalBackend, ::Type{T}, dims::Dims) where {T}
    supports_eltype(backend, T) || throw(ArgumentError("the Metal backend cannot hold $T: Apple GPUs have no double precision, so compute in ComplexF32 or Float32 and keep the double precision work on the host"))
    Metal.MtlArray{T}(undef, dims)
end

# Wrapping host memory in an MTLBuffer needs a page aligned address and a whole
# number of pages, so the slab is padded to both. The HostSlab still reports the
# bytes that were asked for, since the plan's arithmetic counts those.
function alloc_host_slab(::MetalBackend, nbytes::Integer)
    page = Int(Sys.PAGESIZE)
    len = cld(max(Int(nbytes), 1), page) * page
    owner = Vector{UInt8}(undef, len + page)
    pad = Int(mod(-reinterpret(UInt, pointer(owner)), UInt(page)))
    base = pointer(owner) + pad
    aligned = Base.unsafe_wrap(Array, base, (len,))
    slab = MetalSlab(owner, Metal.unsafe_wrap(Metal.MtlArray{UInt8,1}, aligned, (len,)), reinterpret(UInt, base), len)
    register_slab!(slab)
    HostSlab(slab, base, Int(nbytes))
end

# Metal keeps its command queue in task local storage, so all the GPU work of
# one Funicular queue runs on one worker task and one command queue with it.
# Asking for a command queue per operation instead exhausts the driver a few
# thousand operations into a test run.
mutable struct MetalQueue
    inbox::Channel{Any}
    tail::Task
    failure::Any
end

const QUEUES = WeakRef[]
const QUEUES_LOCK = ReentrantLock()

function make_queue(::MetalBackend)
    inbox = Channel{Any}(Inf)
    queue = MetalQueue(inbox, Threads.@spawn(nothing), nothing)
    # invokelatest because the worker outlives the world it was spawned in, and
    # the closures a sweep hands it are usually younger than that.
    Threads.@spawn for item in inbox
        Base.invokelatest(item)
    end
    finalizer(_ -> Threads.@spawn(close(inbox)), queue)
    lock(QUEUES_LOCK) do
        filter!(ref -> ref.value !== nothing, QUEUES)
        push!(QUEUES, WeakRef(queue))
    end
    queue
end

# Every operation ends synchronized, so ordering between queues is the worker's
# and nothing on the GPU outlives the closure that issued it. A real backend
# orders its queues on the device instead, but this one is meant to check the
# logic rather than to achieve any overlap. A failure poisons the queue, the way
# a broken link poisons a chain of tasks on the reference backend.
function submit!(backend::MetalBackend, queue::MetalQueue, work)
    outcome = Ref{Any}(nothing)
    finished = Base.Event()
    jitter = backend.jitter
    put!(queue.inbox, function ()
        if queue.failure === nothing
            try
                jitter > 0 && sleep(jitter * rand())
                work()
                Metal.synchronize()
            catch err
                queue.failure = err
            end
        end
        outcome[] = queue.failure
        notify(finished)
    end)
    task = Threads.@spawn begin
        wait(finished)
        outcome[] === nothing || throw(outcome[])
    end
    queue.tail = task
    TaskEvent(task)
end

record_event(::MetalBackend, queue::MetalQueue) = TaskEvent(queue.tail)

function wait_event(backend::MetalBackend, queue::MetalQueue, event::TaskEvent)
    event.task === nothing && return nothing
    submit!(backend, queue, () -> wait(event.task))
    nothing
end

function sync_queue(::MetalBackend, queue::MetalQueue)
    wait(queue.tail)
    nothing
end

function sync_all(backend::MetalBackend)
    queues = lock(QUEUES_LOCK) do
        MetalQueue[ref.value for ref in QUEUES if ref.value isa MetalQueue]
    end
    for queue in queues
        sync_queue(backend, queue)
    end
    nothing
end

# Metal reaches CPU BLAS, not the GPU, for a complex GEMM whose destination is a
# view, and for a triangular solve of any shape: both then try to read private
# device memory from the host and fail. The portable kernels are used instead.
rdiv_upper!(backend::MetalBackend, Y::AbstractMatrix, R::AbstractMatrix) = rdiv_upper_ka!(backend, Y, R)
gram_accumulate!(backend::MetalBackend, C, A, B, α, β) = gram_accumulate_ka!(backend, C, A, B, α, β)

h2d!(dst, src, backend::MetalBackend; queue=nothing) = metal_copy!(dst, src, backend, queue)
d2h!(dst, src, backend::MetalBackend; queue=nothing) = metal_copy!(dst, src, backend, queue)

function metal_copy!(dst, src, backend::MetalBackend, queue)
    size(dst) == size(src) || throw(DimensionMismatch("copy between a $(size(dst)) destination and a $(size(src)) source"))
    queue === nothing || return submit!(backend, queue, () -> copy_now!(dst, src, backend))
    copy_now!(dst, src, backend)
    Metal.synchronize()
    TaskEvent(nothing)
end

function copy_now!(dst, src, backend::MetalBackend)
    if isdevice(dst) && isdevice(src)
        convert_copy!(backend, dst, src)
    elseif isdevice(dst)
        shared = shared_view(src)
        shared === nothing ? copy_unregistered!(dst, src) : convert_copy!(backend, dst, shared)
    elseif isdevice(src)
        shared = shared_view(dst)
        shared === nothing ? copy_unregistered!(dst, src) : convert_copy!(backend, shared, src)
    else
        copyto!(dst, src)
    end
    dst
end

isdevice(::Metal.MtlArray) = true
isdevice(A::SubArray) = isdevice(parent(A))
isdevice(::Any) = false

# The device's view of host memory that belongs to a plan's slab. Both
# processors address the same bytes, so the "transfer" is a kernel that widens
# or narrows on the way past.
function shared_view(A::Array{T}) where {T}
    slab = slab_containing(pointer(A))
    slab === nothing && return nothing
    offset = Int(reinterpret(UInt, pointer(A)) - slab.base)
    bytes = length(A) * sizeof(T)
    reshape(reinterpret(T, view(slab.gpu, (offset + 1):(offset + bytes))), size(A))
end

function shared_view(A::SubArray)
    whole = shared_view(parent(A))
    whole === nothing ? nothing : view(whole, parentindices(A)...)
end

shared_view(::Any) = nothing

# Host memory outside every slab: the k×k factors, the operator probes, the
# results that come back to a fresh Matrix. Metal copies those whole.
function copy_unregistered!(dst, src)
    (dst isa SubArray || src isa SubArray) && throw(ArgumentError("the Metal backend can copy a view of host memory only when that memory belongs to a ResidencyPlan's host slab, since a view of anything else has no device address. Copy the whole array instead"))
    copyto!(dst, src)
end

end # module FunicularMetalExt
