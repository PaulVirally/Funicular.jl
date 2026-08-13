# The unified-memory baseline. It is meant to be kept forever, because CUDA can
# already oversubscribe device memory by putting the matrix in unified memory and
# letting the driver page it in on demand. If Funicular does not clearly beat
# that, the pipeline has a bug.
#
#     julia --project=test/cuda --startup-file=no benchmark/unified.jl
#
# Both sides do the same work: apply a synthetic operator to every column of a
# tall matrix, a panel at a time, writing the result into a second tall matrix.
# The unified side allocates both in unified memory and calls the operator on
# views of them, so the driver moves pages when it finds them missing. The
# Funicular side keeps both in the host tier and streams panels through a device
# budget an eighth of the size.
#
# SCALE sets how far past device memory the pair of matrices goes. At the
# default of 1.5 neither member of the pair fits alongside the other. That is
# the regime the comparison is about. The matrices are host-backed either way,
# so the machine needs SCALE times the device's memory free in RAM. The script
# prints what it is about to ask for and stops if too little host memory is free.

using CUDA
using Funicular
using LinearAlgebra
using Printf
using Random

using Funicular: alloc_host_slab, h2d!, make_queue, slab_matrix, sync_queue

include("common.jl")

CUDA.functional() || error("no functional CUDA device")

const T = ComplexF64
const SCALE = 1.5
const RATIO = 5.0
const N = 1 << 20
const W = 8

backend = Funicular.cuda_backend()
total = CUDA.total_memory()
k = max(4 * W, (floor(Int, SCALE * total / 2) ÷ (N * sizeof(T))) ÷ W * W)
bytes = N * k * sizeof(T)
panelbytes = N * W * sizeof(T)

banner("Funicular against unified memory",
       @sprintf("%s, %s device memory, two %d×%d %s matrices of %s each",
                CUDA.name(CUDA.device()), Base.format_bytes(total), N, k, T,
                Base.format_bytes(bytes)))

if 2 * bytes > 0.8 * Sys.free_memory()
    error("this needs $(Base.format_bytes(2 * bytes)) of host memory and only $(Base.format_bytes(Sys.free_memory())) is free. Lower SCALE at the top of this file")
end

function spin!(dst, src, repeats)
    dst .= src
    for _ in 1:repeats
        dst .= dst .* 1.0000001
    end
    dst
end

# A panel-capable operator, so both sides apply it the same way and the only
# difference between them is where the panel came from.
struct Spin
    n::Int
    repeats::Int
end

Base.size(G::Spin) = (G.n, G.n)
Base.size(G::Spin, d::Integer) = d <= 2 ? G.n : 1
Base.eltype(::Spin) = T
Base.adjoint(G::Spin) = G
Funicular.panel_capable(::Spin) = true
LinearAlgebra.mul!(dst::AbstractVecOrMat, G::Spin, src::AbstractVecOrMat) = spin!(dst, src, G.repeats)

# Calibrate the operator so one panel of compute costs RATIO panels of transfer,
# since that is the regime the operators this is for live in. The transfer is
# timed out of page-locked memory, because that is what the Funicular side copies.
probe = CUDA.CuArray{T}(undef, N, W)
fill!(probe, one(T))
slab = alloc_host_slab(backend, panelbytes)
staging = slab_matrix(slab, T, (N, W), 0)
fill!(staging, one(T))
queue = make_queue(backend)
unit = best(() -> CUDA.@sync spin!(probe, probe, 1))
copyunit = best(() -> (h2d!(probe, staging, backend; queue=queue); sync_queue(backend, queue)))
repeats = max(1, round(Int, RATIO * copyunit / unit))
G = Spin(N, repeats)
@printf("one panel is %s and crosses in %.2f ms; %d elementwise passes make its compute %.1f× that\n\n",
        Base.format_bytes(panelbytes), 1e3 * copyunit, repeats, RATIO)

println("allocating the Funicular pair ...")
plan = ResidencyPlan(backend=backend, device_budget=total ÷ 8,
                     host_budget=2 * bytes + 2^30, panel_width=W)
Xf = PanelMatrix{T}(undef, N, k; plan=plan)
Yf = PanelMatrix{T}(undef, N, k; plan=plan)
funicular = best(() -> panelmul!(Yf, G, Xf), 2)
Funicular.free!(Xf)
Funicular.free!(Yf)
Xf = Yf = plan = nothing
GC.gc()

println("allocating the unified pair ...")
Xu = CuArray{T,2,CUDA.UnifiedMemory}(undef, N, k)
Yu = CuArray{T,2,CUDA.UnifiedMemory}(undef, N, k)
fill!(Xu, one(T))

function unifiedapply!(Y, G, X, w)
    for first in 1:w:size(X, 2)
        cols = first:min(first + w - 1, size(X, 2))
        mul!(view(Y, :, cols), G, view(X, :, cols))
    end
    CUDA.synchronize()
    Y
end

unified = best(() -> unifiedapply!(Yu, G, Xu, W), 2)

@printf("\n%-12s %10s %14s\n", "run", "time", "throughput")
@printf("%-12s %9.2fs %10.2f GB/s\n", "unified", unified, gbps(2 * bytes, unified))
@printf("%-12s %9.2fs %10.2f GB/s\n", "funicular", funicular, gbps(2 * bytes, funicular))
@printf("\nFunicular is %.2f× the unified-memory run  %s\n",
        unified / funicular, verdict(unified / funicular >= 2))

record("unified", ("run", "seconds"), [("unified", unified), ("funicular", funicular)])

println("\nUnder 2× means the pipeline is not doing its job: the driver's demand")
println("paging is as good as an explicit schedule, and it should not be when the")
println("access pattern is known a panel in advance. Read benchmark/overlap.jl next.")
