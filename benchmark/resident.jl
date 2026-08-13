# Non-regression against a resident matrix. This is the case where the whole
# matrix fits on the GPU and the user could have written plain CUDA. Funicular
# must not make that case worse.
#
#     julia --project=test/cuda --startup-file=no benchmark/resident.jl
#
# It cannot make it free: a panel matrix keeps its panels in host memory, so a
# sweep pays one upload and one writeback per panel where a resident CuMatrix
# pays none. The pipeline is meant to hide those behind the compute, and what is
# left over when it does is the prologue and the epilogue, two panel transfers a
# two-buffer schedule cannot overlap with anything.
#
# So the table reports, per compute-to-copy ratio: the same device work done on
# a resident CuMatrix with no Funicular involved, the same work through a
# read-write sweep, the two end transfers the schedule cannot hide, and what is
# left once those are accounted for. That last column is Funicular's own
# overhead, and it is held to 3%.

using CUDA
using Funicular
using Printf
using Random

using Funicular: sweep!, updatepanels

include("common.jl")

CUDA.functional() || error("no functional CUDA device")

const T = ComplexF64
const N = 1 << 20
const K = 16
const W = 2
const RATIOS = (1.0, 2.0, 5.0, 10.0)
const SLACK = 0.03

backend = Funicular.cuda_backend()
plan = ResidencyPlan(backend=backend, device_budget=2 * 2^30, host_budget=2 * 2^30)
pm = PanelMatrix(randn(T, N, K); plan=plan, w=W)
panels = npanels(pm)

function spin!(panel, repeats)
    for _ in 1:repeats
        panel .= panel .* 1.0000001
    end
    panel
end

# The same work the sweep will do, on a buffer that is already where it needs to
# be. One call per panel, so the two sides issue the same number of kernels.
function resident!(buffers, repeats)
    for buffer in buffers
        spin!(buffer, repeats)
    end
    nothing
end

banner("Against a resident matrix",
       @sprintf("%s, %d×%d %s in %d panels of %d", CUDA.name(CUDA.device()), N, K, T, panels, W))

copytime = best(() -> sweep!((_, _) -> nothing, updatepanels(pm); nbuffers=1))
# A serialized pass moves each panel twice, up and back, so this is what one
# single-direction panel transfer costs.
transfer = copytime / (2 * panels)
resident = [CUDA.CuArray{T}(undef, N, W) for _ in 1:panels]
for buffer in resident
    fill!(buffer, one(T))
end
unit = best(() -> CUDA.@sync resident!(resident, 1))

@printf("one serialized pass of the copies: %7.2f ms, so %.2f ms per panel transfer\n",
        1e3 * copytime, 1e3 * transfer)
@printf("one elementwise pass over them:    %7.2f ms\n\n", 1e3 * unit)

header = ("ratio", "raw_ms", "sweep_ms", "unhidden_ms", "overhead")
rows = Tuple[]
@printf("%6s %10s %10s %12s %10s  %s\n",
        "ratio", "raw", "sweep", "unhidden", "overhead", "verdict")
for ratio in RATIOS
    repeats = max(1, round(Int, ratio * copytime / unit))
    raw = best(() -> CUDA.@sync resident!(resident, repeats))
    swept = best(() -> sweep!((_, panel) -> spin!(panel, repeats), updatepanels(pm); nbuffers=2))
    # The first upload and the last writeback have nothing to overlap against.
    unhidden = 2 * transfer
    overhead = (swept - raw - unhidden) / raw
    push!(rows, (ratio, 1e3 * raw, 1e3 * swept, 1e3 * unhidden, overhead))
    @printf("%6.1f %9.2fms %9.2fms %11.2fms %9.1f%%  %s\n",
            ratio, 1e3 * raw, 1e3 * swept, 1e3 * unhidden, 100 * overhead,
            verdict(overhead <= SLACK))
end

record("resident", header, rows)

println("\nThe overhead column is what a sweep costs beyond the device work and the")
println("two transfers no two-buffer schedule can hide. Over 3% there means the")
println("bookkeeping is showing: events made per panel, buffers not coming from the")
println("pool, or a host synchronization between panels.")
