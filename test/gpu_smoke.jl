# Run this on a CUDA machine before the test suite:
#
#     julia --project=test/cuda --startup-file=no test/gpu_smoke.jl
#
# It walks the CUDA backend from the outside in, one printed step at a time, so
# that the first thing to go wrong says which call it was. Every step is small
# and takes seconds, and if a step throws, then everything before it worked.
#
# The environment, once:
#
#     julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

using CUDA
using Funicular
using HDF5
using LinearAlgebra
using Printf
using Random

using Funicular: alloc_device, alloc_host_slab, device_bytes_allocated, d2h!,
                 devicetype, h2d!, ka_backend, make_queue, panelstorage,
                 readpanels, record_event, sweep!, sync_queue, transfers_are_real,
                 updatepanels, wait_event, writepanels

CUDA.functional() || error("no functional CUDA device")

step(name) = @printf("\n--- %s ---\n", name)
report(name, ok) = @printf("%-46s %s\n", name, ok ? "ok" : "FAILED")

step("environment")
CUDA.versioninfo()
@printf("device %s, %s total\n", CUDA.name(CUDA.device()),
        Base.format_bytes(CUDA.total_memory()))

step("1. the backend and its types")
backend = Funicular.cuda_backend()
@printf("backend %s\n", backend)
report("devicetype is a CuMatrix", devicetype(backend, ComplexF64) === CuMatrix{ComplexF64})
report("kernels launch on the CUDA backend", ka_backend(backend) isa CUDA.CUDABackend)
report("transfers are real", transfers_are_real(backend))
buffer = alloc_device(backend, ComplexF64, (16, 4))
report("alloc_device", buffer isa CuMatrix{ComplexF64} && size(buffer) == (16, 4))

step("2. the host slab is page-locked")
slab = alloc_host_slab(backend, 1 << 20)
address = convert(Ptr{Nothing}, slab.ptr)
if isdefined(CUDA, :is_pinned)
    report("CUDA.is_pinned on the slab", CUDA.is_pinned(address))
else
    println("CUDA.is_pinned is gone; please paste `names(CUDA, all = true)` matching /pin/")
end

step("3. a panel matrix round trip")
const N = 1 << 10
const K = 23
const W = 7
plan = ResidencyPlan(backend=backend, device_budget=512 * 2^20, host_budget=512 * 2^20)
A = randn(ComplexF64, N, K)
pm = PanelMatrix(A; plan=plan, w=W)
println(pm)
report("panels partition k", sum(panelwidth(pm, j) for j in 1:npanels(pm)) == K)
report("dense round trip", Matrix(pm) == A)

step("4. queues, events, and an asynchronous copy")
up, down = make_queue(backend), make_queue(backend)
device = alloc_device(backend, ComplexF64, (N, W))
back = zeros(ComplexF64, N, W)
issue = @elapsed begin
    h2d!(device, panelstorage(pm.panels[1]), backend; queue=up)
    wait_event(backend, down, record_event(backend, up))
    d2h!(back, device, backend; queue=down)
end
drain = @elapsed sync_queue(backend, down)
@printf("issue %.3f ms, then drain %.3f ms\n", 1e3 * issue, 1e3 * drain)
println("(an issue much cheaper than the drain means the copies really were queued)")
report("panel 1 came back", back == A[:, 1:W])

step("5. one sweep over the panels")
foreachpanel(pm) do j, panel
    panel .*= 2
end
report("read-write sweep, ragged last panel", Matrix(pm) == 2 .* A)
foreachpanel(pm; write=false, nbuffers=1) do j, panel end
report("nbuffers = 1 sweep", Matrix(pm) == 2 .* A)

step("6. the row traversal, where the strided copies are")
copyto!(pm, A)
G = gram(pm)
@printf("‖gram(X) - X'X‖ / ‖X'X‖ = %.3e\n", norm(G - A'A) / norm(A'A))
report("gram against dense", isapprox(G, A'A; rtol=1e-10))

R = cholqr2!(pm)
Q = Matrix(pm)
@printf("‖Q'Q - I‖ = %.3e, ‖QR - A‖ / ‖A‖ = %.3e\n",
        norm(Q'Q - I), norm(Q * R - A) / norm(A))
report("cholqr2! orthonormalizes", norm(Q'Q - I) < 100 * N * K * eps(Float64))
report("cholqr2! factors", isapprox(Q * R, A; rtol=1e-10))

step("7. the operator")
M = randn(ComplexF64, N, N)
M .= (M .+ M') ./ 2
dense = CuArray(M)
report("check_operator", check_operator(dense; backend=backend))
X = PanelMatrix(A; plan=plan, w=W)
Y = PanelMatrix{ComplexF64}(undef, N, K; plan=plan, w=W)
panelmul!(Y, dense, X)
report("panelmul! against dense", isapprox(Matrix(Y), M * A; rtol=1e-10))

cholqr2!(X)
S = project(X, dense)
Qx = Matrix(X)
report("project against dense", isapprox(S, Qx' * M * Qx; rtol=1e-8))

step("8. narrowed host storage")
narrow = ResidencyPlan(backend=backend, device_budget=512 * 2^20,
                       host_budget=512 * 2^20, host_eltype=ComplexF32)
Z = PanelMatrix(A; plan=narrow, w=W)
report("stored narrow, read back wide", Matrix(Z) == ComplexF64.(ComplexF32.(A)))
foreachpanel(Z) do j, panel
    panel .*= 2
end
report("narrowed sweep", Matrix(Z) == 2 .* ComplexF64.(ComplexF32.(A)))

step("9. allocation inside a sweep, after warmup")
double!(_, panel) = (panel .*= 2)
for warmup in 1:2
    sweep!(double!, updatepanels(X))
    gram(X)
end
@printf("device bytes the plan holds: %s\n", Base.format_bytes(device_bytes_allocated(plan)))
@printf("allocated inside a data sweep: %d B\n", CUDA.@allocated(sweep!(double!, updatepanels(X))))
@printf("allocated inside a gram:       %d B\n", CUDA.@allocated(gram(X)))
println("(the data sweep must be 0, or a sweep is allocating. The gram")
println("allocates 32 B per row block: CUDA.jl stages a gemm's two scalars in")
println("device memory. test/test_cuda.jl holds it to exactly that count)")

step("10. the disk tier under a host budget smaller than the matrix")
C = randn(ComplexF64, N, 8 * W)
mktempdir() do dir
    block = N * W * sizeof(ComplexF64)
    cold = ResidencyPlan(backend=backend, device_budget=512 * 2^20,
                         host_budget=5 * block, scratch_dir=dir)
    spilled = PanelMatrix(C; plan=cold, w=W)
    println(spilled)
    report("collect after spilling", Matrix(spilled) == C)
    foreachpanel(spilled) do j, panel
        panel .*= 2
    end
    report("a sweep fed from disk", Matrix(spilled) == 2 .* C)
    @printf("panels read from disk: %d, written: %d\n",
            Funicular.disk_reads(spilled), Funicular.disk_writes(spilled))
    println("(a panel is read straight into page-locked memory, so the copy out of")
    println("it stays asynchronous; nonzero counts mean panels really did cycle)")

    path = joinpath(dir, "smoke.h5")
    save(spilled, path)
    reloaded = load(PanelMatrix, path; plan=ResidencyPlan(backend=backend, device_budget=512 * 2^20, host_budget=512 * 2^20))
    report("save and load", Matrix(reloaded) == 2 .* C)
    Funicular.free!(reloaded)
    Funicular.free!(spilled)
end

step("11. does the pipeline overlap")
wide = ResidencyPlan(backend=backend, device_budget=2 * 2^30, host_budget=2 * 2^30)
big = PanelMatrix(randn(ComplexF64, 1 << 19, 32); plan=wide, w=4)
function spin!(_, panel)
    for _ in 1:100
        panel .= panel .* 1.0001f0
    end
end
for nbuffers in (1, 2)
    sweep!(spin!, updatepanels(big); nbuffers=nbuffers)
    elapsed = @elapsed sweep!(spin!, updatepanels(big); nbuffers=nbuffers)
    @printf("nbuffers = %d: %.1f ms\n", nbuffers, 1e3 * elapsed)
end
println("(two buffers should save about one panel copy per panel; benchmark/overlap.jl is the real measurement)")

step("12. ghost panels, generated rather than stored")
Ω = GhostPanels(ComplexF64, N, K; plan=plan, seed=0xdeadbeef, w=W)
first_pass = Matrix(Ω)
for panel in Ω.panels
    Funicular.evict!(panel, plan)
end
report("regenerates the same entries", Matrix(Ω) == first_pass)
GΩ = PanelMatrix{ComplexF64}(undef, N, K; plan=plan, w=W)
panelmul!(GΩ, dense, Ω)
report("streams into an operator sweep", isapprox(Matrix(GΩ), M * first_pass; rtol=1e-10))
println("(no host block outlives the sweep that asked for it, and nothing is written)")

println("\ndone")
