# The main benchmark for this package. Run it on a CUDA machine:
#
#     julia --project=test/cuda --startup-file=no benchmark/overlap.jl
#
# The operators this package is for cost five to ten times as much to apply as
# their argument costs to move over the host link, so double buffering should
# hide the traffic entirely, and this script checks whether it does. For a
# synthetic panel function whose cost is tunable, it reports, per
# compute-to-copy ratio:
#
#   copy      one serialized pass of the copies alone, no compute
#   compute   the panel function alone, on a resident buffer, no copies
#   serial    the sweep at nbuffers = 1
#   pipeline  the sweep at nbuffers = 2
#
# and holds the pipeline to max(copy, compute) * 1.15. The compute is repeated
# elementwise passes rather than FFTs, since the question here is whether the
# copies disappear behind device work, and an FFT would only make the device
# work harder to attribute.

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
const RATIOS = (0.5, 1.0, 2.0, 5.0, 10.0)
const SLACK = 1.15

backend = Funicular.cuda_backend()
plan = ResidencyPlan(backend=backend, device_budget=2 * 2^30, host_budget=2 * 2^30)
pm = PanelMatrix(randn(T, N, K); plan=plan, w=W)
panels = npanels(pm)

@printf("%s, %d×%d %s in %d panels of %d, %s per panel\n\n",
        CUDA.name(CUDA.device()), N, K, T, panels, W,
        Base.format_bytes(N * W * sizeof(T)))

noop!(_, panel) = nothing

function spin!(panel, repeats)
    for _ in 1:repeats
        panel .= panel .* 1.0000001
    end
    panel
end

copytime = best(() -> sweep!(noop!, updatepanels(pm); nbuffers=1))
pipelinedcopy = best(() -> sweep!(noop!, updatepanels(pm); nbuffers=2))
resident = CUDA.CuArray{T}(undef, N, W)
unit = best(() -> CUDA.@sync spin!(resident, 1)) * panels

@printf("one serialized pass of the copies: %7.2f ms (%.1f GB/s)\n",
        1e3 * copytime, 2 * sizeof(T) * N * K / copytime / 2^30)
# With no compute to hide behind, two buffers can still overlap an upload with
# the previous writeback, since the two use opposite directions of the bus. If
# this is not clearly under the line above, the copies are not overlapping each
# other either and the trouble is in the schedule rather than in the compute.
@printf("the same copies at two buffers:    %7.2f ms\n", 1e3 * pipelinedcopy)
@printf("one elementwise pass over them:    %7.2f ms\n\n", 1e3 * unit)

rows = Tuple[]
@printf("%6s %8s %9s %9s %9s %9s %9s  %s\n",
        "ratio", "repeats", "copy", "compute", "serial", "pipeline", "budget", "verdict")
for ratio in RATIOS
    repeats = max(1, round(Int, ratio * copytime / unit))
    compute = best(() -> CUDA.@sync spin!(resident, repeats)) * panels
    serial = best(() -> sweep!((_, panel) -> spin!(panel, repeats), updatepanels(pm); nbuffers=1))
    pipeline = best(() -> sweep!((_, panel) -> spin!(panel, repeats), updatepanels(pm); nbuffers=2))
    budget = SLACK * max(copytime, compute)
    push!(rows, (ratio, 1e3 * copytime, 1e3 * compute, 1e3 * serial, 1e3 * pipeline, 1e3 * budget))
    @printf("%6.1f %8d %8.2fms %8.2fms %8.2fms %8.2fms %8.2fms  %s\n",
            ratio, repeats, 1e3 * copytime, 1e3 * compute, 1e3 * serial,
            1e3 * pipeline, 1e3 * budget, verdict(pipeline <= budget))
end

record("overlap", ("ratio", "copy_ms", "compute_ms", "serial_ms", "pipeline_ms", "budget_ms"), rows)

println("\nA pipeline column at or under budget means the overlap claim holds.")
println("Over budget at a ratio of 5 or 10 means something serializes: pageable")
println("host memory, a synchronize inside the loop, or CUBLAS on another stream.")
