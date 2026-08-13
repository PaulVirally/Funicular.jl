module FunicularCUDAExt

# CUDA.jl exports a CUDABackend of its own, the KernelAbstractions one, so it is
# imported rather than brought into scope. The CUDABackend defined below is
# Funicular's own, and CUDA's is written out in full where it is needed.
import CUDA

import Funicular: alloc_device, alloc_host_slab, cuda_backend, d2h!, devicetype,
                  h2d!, ka_backend, make_queue, make_sweep_queues, record_event,
                  submit!, sync_all, sync_event, sync_queue, wait_event

using Funicular: DeviceBackend, SLAB_ALIGN, aligned_slab, convert_copy!

"""
    CUDABackend()

Device backend on an NVIDIA GPU, and the one Funicular is designed for. Panels
live in `CuArray`s, host slabs are page-locked so that copies out of them are
asynchronous, queues are explicit non-blocking streams, and ordering between
them is `CuEvent`s waited on by the device rather than by the host.

The backend holds no state: the staging buffers belong to the
[`ResidencyPlan`](@ref) and the streams to whichever sweep made them.
"""
struct CUDABackend <: DeviceBackend end

cuda_backend() = CUDABackend()

devicetype(::CUDABackend, ::Type{T}) where {T} = CUDA.CuMatrix{T}
ka_backend(::CUDABackend) = CUDA.CUDABackend()

function alloc_device(::CUDABackend, ::Type{T}, dims::Dims) where {T}
    buffer = CUDA.CuArray{T}(undef, dims)
    # Allocation is ordered on the stream of whatever task asked for the buffer,
    # and the sweeps then use it on streams of their own, so it has to have
    # happened before this returns. Buffers are checked out before a sweep
    # starts rather than inside one.
    CUDA.synchronize()
    buffer
end

# The whole slab is registered once and the panel views that alias it are
# page-locked with it. Without that, an asynchronous copy out of pageable memory
# silently becomes a synchronous one.
function alloc_host_slab(::CUDABackend, nbytes::Integer)
    owner = Vector{UInt8}(undef, Int(nbytes) + SLAB_ALIGN)
    CUDA.pin(owner)
    aligned_slab(owner, pointer(owner), nbytes)
end

# CUDA.jl remembers, for every array, the stream it was last used on, and makes
# the host wait for that stream before letting another one touch the array. A
# pipeline moves each staging buffer between three streams on purpose and orders
# them with events, so that safeguard fires on every panel: the writeback of
# panel j cannot be issued until the kernels of panel j have finished on the
# host's clock, and the upload of panel j+1 queues up behind it. We measured
# that, and it costs the entire overlap.
#
# As such, the three streams of one sweep share a stream for bookkeeping, the
# compute one, and every operation runs in that context while being issued on
# the stream it belongs to. CUDA.jl then sees one stream throughout and stays
# out of the way, and the ordering is left to the events.
struct SweepQueue
    stream::CUDA.CuStream
    owner::CUDA.CuStream
end

# A stream created with the default flags synchronizes against the legacy
# default stream, which would serialize the pipeline against anything else in
# the process.
new_stream() = CUDA.CuStream(flags=CUDA.STREAM_NON_BLOCKING)

function make_queue(::CUDABackend)
    stream = new_stream()
    SweepQueue(stream, stream)
end

function make_sweep_queues(::CUDABackend)
    up, compute, down = new_stream(), new_stream(), new_stream()
    (SweepQueue(up, compute), SweepQueue(compute, compute), SweepQueue(down, compute))
end

# The closure runs on the calling task inside the stream context, so the kernels
# it launches, the CUBLAS calls it makes and the copies the operator issues all
# land on this stream instead of on the task-local default one.
function submit!(backend::CUDABackend, queue::SweepQueue, work)
    CUDA.stream!(work, queue.stream)
    record_event(backend, queue)
end

function record_event(::CUDABackend, queue::SweepQueue)
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(event, queue.stream)
    event
end

# A device-side wait: the stream stops until the event is signalled while the
# calling task carries on issuing work.
function wait_event(::CUDABackend, queue::SweepQueue, event::CUDA.CuEvent)
    CUDA.wait(event, queue.stream)
    nothing
end

function sync_queue(::CUDABackend, queue::SweepQueue)
    CUDA.synchronize(queue.stream)
    nothing
end

function sync_event(::CUDABackend, event::CUDA.CuEvent)
    CUDA.synchronize(event)
    nothing
end

function sync_all(::CUDABackend)
    CUDA.device_synchronize()
    nothing
end

h2d!(dst, src, backend::CUDABackend; queue=nothing) = cuda_copy!(dst, src, backend, queue)
d2h!(dst, src, backend::CUDABackend; queue=nothing) = cuda_copy!(dst, src, backend, queue)

function cuda_copy!(dst, src, backend::CUDABackend, queue)
    size(dst) == size(src) || throw(DimensionMismatch("copy between a $(size(dst)) destination and a $(size(src)) source"))
    eltype(dst) === eltype(src) || return converting_copy!(dst, src, backend, queue)
    queue === nothing || return CUDA.stream!(queue.owner) do
        copy_region!(dst, src, queue.stream, true)
        record_event(backend, queue)
    end
    stream = CUDA.stream()
    copy_region!(dst, src, stream, false)
    record_event(backend, SweepQueue(stream, stream))
end

# Every copy is a 2D one. A whole panel and a column range of a staging buffer
# are contiguous, so a flat copy would do for them, but a row block is a slice
# of each panel and is not contiguous. One call that names both pitches covers
# all of them.
function copy_region!(dst, src, stream::CUDA.CuStream, async::Bool)
    T = eltype(dst)
    isempty(dst) && return dst
    width, height, dstpitch = copy_shape(dst)
    _, _, srcpitch = copy_shape(src)
    GC.@preserve dst src begin
        CUDA.unsafe_copy2d!(region_pointer(dst), memorytype(dst),
                            region_pointer(src), memorytype(src),
                            width, height;
                            dstPitch=dstpitch * sizeof(T),
                            srcPitch=srcpitch * sizeof(T),
                            async=async, stream=stream)
    end
    dst
end

# Elements down a column, columns, and the distance between two columns. Panel
# matrices are column major throughout, so the first dimension is always the
# contiguous one.
function copy_shape(A::AbstractArray)
    ndims(A) <= 2 || throw(ArgumentError("Funicular copies panels and blocks of them, which are at most two-dimensional, got a $(ndims(A))-dimensional array"))
    stride(A, 1) == 1 || throw(ArgumentError("a copy needs its columns contiguous but this array has a stride of $(stride(A, 1)) between neighbouring rows"))
    rows = size(A, 1)
    ndims(A) == 1 && return (rows, 1, rows)
    (rows, size(A, 2), stride(A, 2))
end

memorytype(::CUDA.CuArray) = CUDA.DeviceMemory
memorytype(::Array) = CUDA.HostMemory
memorytype(A::SubArray) = memorytype(parent(A))

region_pointer(A::CUDA.CuArray) = pointer(A)
region_pointer(A::Array) = pointer(A)
region_pointer(A::SubArray) = region_pointer(parent(A)) + view_offset(A) * sizeof(eltype(A))

# Where a view starts, in elements from the start of its parent. Computed here
# rather than taken from the pointer of the view, since the two array types
# reach their pointers by different routes.
function view_offset(A::SubArray)
    offset = 0
    for (d, index) in enumerate(A.indices)
        offset += (first(index) - 1) * stride(parent(A), d)
    end
    offset
end

isdevice(::CUDA.CuArray) = true
isdevice(A::SubArray) = isdevice(parent(A))
isdevice(::Any) = false

# This runs outside every sweep. A sweep hands materialize! and writeback! a
# staging buffer in the storage eltype. Only the narrow bytes cross the bus, and
# the conversion runs as a device kernel without allocating. A bare h2d! has no
# such buffer, so it allocates one and waits for the copy rather than letting
# the buffer outlive it. Being outside the pipeline, it runs in the context of
# its own stream rather than a sweep's, and it waits, so it does not have to
# work around the bookkeeping described above.
function converting_copy!(dst, src, backend::CUDABackend, queue)
    stream = queue === nothing ? CUDA.stream() : queue.stream
    CUDA.stream!(stream) do
        if isdevice(dst) && isdevice(src)
            convert_copy!(backend, dst, src)
        elseif isdevice(dst)
            staging = CUDA.CuArray{eltype(src)}(undef, size(src))
            copy_region!(staging, src, stream, true)
            convert_copy!(backend, dst, staging)
        else
            staging = CUDA.CuArray{eltype(dst)}(undef, size(dst))
            convert_copy!(backend, staging, src)
            copy_region!(dst, staging, stream, true)
        end
    end
    CUDA.synchronize(stream)
    record_event(backend, SweepQueue(stream, stream))
end

end # module FunicularCUDAExt
