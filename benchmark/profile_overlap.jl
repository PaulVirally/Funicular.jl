# A trace for diagnosing a pipeline that is not overlapping. Run it on a CUDA
# machine:
#
#     julia --project=test/cuda --startup-file=no benchmark/profile_overlap.jl
#
# When benchmark/overlap.jl reports that the pipeline hides only a fraction of
# the panel transfers, and by the same constant however heavy the compute is, the
# three-queue schedule is not doing what it should: once compute outweighs the
# copies it should hide all of them. This script prints a trace showing where the
# time actually goes.
#
# It is deliberately small, four panels and eight kernels each, so the tables fit
# on a screen. Three things to read out of them:
#
#   1. The host-side table: is the copy issued as cuMemcpy2DAsync, or as the
#      synchronous cuMemcpy2D? Is there a cuStreamSynchronize or a
#      cuEventSynchronize between the panels, rather than only at the end?
#   2. The device-side table: do the memcpy rows and the kernel rows overlap in
#      time, or does each memcpy sit in a gap between kernels?
#   3. The copy-only trace at the bottom: with no compute at all, does the
#      upload of one panel overlap the writeback of the previous one? That
#      separates "copies do not overlap compute" from "copies do not overlap
#      anything".
#
# NVML is not involved in any of this, so the NVML_ERROR_LIB_RM_VERSION_MISMATCH
# warning does not affect the trace.

using CUDA
using Funicular
using Printf
using Random

using Funicular: sweep!, updatepanels

CUDA.functional() || error("no functional CUDA device")

const T = ComplexF64
const N = 1 << 20
const K = 8
const W = 2
const REPEATS = 8

backend = Funicular.cuda_backend()
plan = ResidencyPlan(backend=backend, device_budget=2 * 2^30, host_budget=2 * 2^30)
pm = PanelMatrix(randn(T, N, K); plan=plan, w=W)

@printf("%s, %d×%d %s in %d panels of %d, %s per panel\n",
        CUDA.name(CUDA.device()), N, K, T, npanels(pm), W,
        Base.format_bytes(N * W * sizeof(T)))
@printf("one panel is %s each way, %d kernels of compute\n\n",
        Base.format_bytes(N * W * sizeof(T)), REPEATS)

function spin!(_, panel)
    for _ in 1:REPEATS
        panel .= panel .* 1.0000001
    end
end

quiet!(_, panel) = nothing

# Warm up the pool, the pinning, the kernels and the streams.
for _ in 1:3
    sweep!(spin!, updatepanels(pm); nbuffers=2)
    sweep!(quiet!, updatepanels(pm); nbuffers=2)
end

elapsed = @elapsed sweep!(spin!, updatepanels(pm); nbuffers=2)
@printf("the sweep about to be traced takes %.2f ms\n", 1e3 * elapsed)

println("\n=== compute and copies, nbuffers = 2 ===")
CUDA.@profile trace=true sweep!(spin!, updatepanels(pm); nbuffers=2)

println("\n=== the same sweep serialized, nbuffers = 1 ===")
CUDA.@profile trace=true sweep!(spin!, updatepanels(pm); nbuffers=1)

println("\n=== copies only, no compute, nbuffers = 2 ===")
CUDA.@profile trace=true sweep!(quiet!, updatepanels(pm); nbuffers=2)

println("\ndone")
