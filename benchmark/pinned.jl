# Pinned against pageable host memory.
#
#     julia --project=test/cuda --startup-file=no benchmark/pinned.jl
#
# The host tier is page-locked because an asynchronous copy out of pageable
# memory is not asynchronous: the driver stages it through a bounce buffer of
# its own and the call blocks. This script measures the difference, and it is
# the first thing to look at when the overlap benchmark goes flat: if the two
# rows below come out at the same speed, the host tier is not pinned any more.
#
# Both sides go through Funicular's own h2d!, so the only difference between
# them is which memory the source lives in.

using CUDA
using Funicular

using Funicular: alloc_host_slab, alloc_device, h2d!, make_queue, slab_matrix, sync_queue

include("common.jl")

CUDA.functional() || error("no functional CUDA device")

const T = ComplexF64
const N = 1 << 22
const W = 8

backend = Funicular.cuda_backend()
dims = (N, W)
nbytes = prod(dims) * sizeof(T)

slab = alloc_host_slab(backend, nbytes)
locked = slab_matrix(slab, T, dims, 0)
pageable = Matrix{T}(undef, dims)
fill!(locked, one(T))
fill!(pageable, one(T))
device = alloc_device(backend, T, dims)
queue = make_queue(backend)

function upload(src)
    h2d!(device, src, backend; queue=queue)
    sync_queue(backend, queue)
end

banner("Pinned against pageable host memory",
       string(CUDA.name(CUDA.device()), ", ", Base.format_bytes(nbytes), " per copy"))

lockedtime = best(() -> upload(locked))
pageabletime = best(() -> upload(pageable))

@printf("%-10s %9s %9s\n", "source", "time", "bandwidth")
@printf("%-10s %8.2fms %7.1f GB/s\n", "pinned", 1e3 * lockedtime, gbps(nbytes, lockedtime))
@printf("%-10s %8.2fms %7.1f GB/s\n", "pageable", 1e3 * pageabletime, gbps(nbytes, pageabletime))
@printf("\npinned is %.2f× faster\n", pageabletime / lockedtime)

record("pinned", ("source", "bandwidth_gbps"),
       [("pinned", gbps(nbytes, lockedtime)), ("pageable", gbps(nbytes, pageabletime))])

if isdefined(CUDA, :is_pinned) && !CUDA.is_pinned(pointer(locked))
    println("\nWARNING: the host slab did not come back page-locked, so Funicular's")
    println("asynchronous copies are synchronous in practice and the pipeline cannot overlap.")
end

println("\nExpect pinned near the host link's line rate and pageable well under")
println("it, often by a factor of two or more. Two rows at the same speed mean")
println("CUDA.pin stopped taking, or panels stopped aliasing the slab it was")
println("called on.")
