# Sustained bandwidth of the disk tier. Unlike the rest of this directory it
# needs no GPU: it measures HDF5 moving whole panels between a filesystem and
# the host tier, and the device backend only decides where the sweep's staging
# buffers live.
#
#     julia --project=test/cuda --startup-file=no benchmark/disk.jl [scratch_dir]
#     julia --project=test --startup-file=no benchmark/disk.jl [scratch_dir]
#
# Point it at the filesystem the runs will actually use. A parallel filesystem
# is fast for whole-panel sequential reads and slow for anything smaller, and
# Funicular never asks it for anything smaller. There is no assertion here: what
# a disk does varies by more than any threshold worth writing down. The numbers
# are for the record and for noticing when they change.
#
# Panel widths are swept to show the granularity effect. A panel here is a few
# tens of megabytes rather than the gigabyte a real run uses, so read these as
# a shape rather than as the number a production run will see.

using Funicular
using HDF5
using Printf
using Random

include("common.jl")

const T = ComplexF64
const N = 1 << 17
const K = 64
const WIDTHS = (1, 2, 4, 8, 16)

backend = try
    @eval using CUDA
    CUDA.functional() ? Funicular.cuda_backend() : CPUBackend()
catch
    CPUBackend()
end

scratch = length(ARGS) >= 1 ? ARGS[1] : mktempdir()
mkpath(scratch)

banner("Disk tier bandwidth",
       @sprintf("%s, %d×%d %s (%s) in %s", nameof(typeof(backend)), N, K, T,
                Base.format_bytes(N * K * sizeof(T)), repr(scratch)))

A = randn(T, N, K)
bytes = sizeof(A)

header = ("panel_width", "write_gbps", "read_gbps")
rows = Tuple[]
@printf("%12s %8s %12s %12s\n", "panel width", "panels", "write", "read")
for w in WIDTHS
    # One panel of host budget, so every panel written goes to disk at once and
    # every panel read comes back from it.
    plan = ResidencyPlan(backend=backend, device_budget=2 * 2^30,
                         host_budget=4 * N * w * sizeof(T), scratch_dir=scratch,
                         panel_width=w)
    pm = PanelMatrix{T}(undef, N, K; plan=plan)
    written = @elapsed begin
        copyto!(pm, A)
        for panel in pm.panels
            Funicular.evict!(panel, plan)
        end
    end
    read = @elapsed begin
        for j in 1:npanels(pm)
            Funicular.prefetch!(pm, j)
            Funicular.evict!(pm.panels[j], plan)
        end
    end
    push!(rows, (w, gbps(bytes, written), gbps(bytes, read)))
    @printf("%12d %8d %8.2f GB/s %8.2f GB/s\n", w, npanels(pm), gbps(bytes, written), gbps(bytes, read))
    Funicular.free!(pm)
end

record("disk", header, rows)

println("\nA row that falls off at the narrow widths shows the per-call cost of HDF5")
println("and of the lock around it. That is why panels are sized in gigabytes in a")
println("real run. A flat row across the sweep means the filesystem, not the")
println("granularity, is the limit. A read column faster than the hardware")
println("can possibly be means the page cache still held what was just written, so")
println("point this at a scratch filesystem and a size the cache cannot hold.")
