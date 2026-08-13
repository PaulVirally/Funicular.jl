# cholqr2! against a dense QR on the device, for sizes where the dense one fits.
#
#     julia --project=test/cuda --startup-file=no benchmark/cholqr.jl
#
# It measures two things. The first is the cost model: CholeskyQR2 is four
# passes over the matrix and two k×k Cholesky factorizations on the host,
# against a Householder QR that reads and writes the matrix many times but never
# leaves the device. The second is a cross-check: the Gram matrix of the
# orthonormalized Q against the identity, next to what LAPACK's QR gets on the
# same matrix.
#
# The comparison flatters the dense side in two ways. qr! stops at the
# factorization and never forms Q, which cholqr2! has already done by the time
# it returns, and qr! runs on a matrix that is already resident while cholqr2!
# streams the panels up and back. Read the table as the price of streaming
# rather than as a claim that one algorithm beats the other.

using CUDA
using Funicular
using LinearAlgebra
using Printf
using Random

include("common.jl")

CUDA.functional() || error("no functional CUDA device")

const T = ComplexF64
const N = 1 << 18
const KS = (16, 32, 64, 128)

backend = Funicular.cuda_backend()

banner("CholeskyQR2 against a device QR",
       string(CUDA.name(CUDA.device()), ", N = ", N, ", ", T))

# Moderately conditioned, since that is the regime a randomized range finder
# hands over and the regime CholeskyQR2 is for.
function testmatrix(k)
    A = randn(T, N, k)
    A .*= reshape(T.(exp10.(range(0, -3, length=k))), 1, k)
    A
end

header = ("k", "cholqr2_ms", "qr_ms", "cholqr2_orth", "lapack_orth")
rows = Tuple[]
@printf("%5s %12s %10s %14s %14s\n", "k", "cholqr2!", "qr!", "‖Q'Q-I‖", "LAPACK same")
for k in KS
    A = testmatrix(k)
    plan = ResidencyPlan(backend=backend, device_budget=4 * 2^30, host_budget=4 * 2^30)
    # Four panels rather than the one the budget would allow, so that what is
    # being timed is the streaming schedule and not a single resident block.
    pm = PanelMatrix(A; plan=plan, w=max(1, k ÷ 4))

    device = CuArray(A)
    scratch = similar(device)
    # Every pass of cholqr2! does the same work whatever the matrix holds, so
    # repeating it on the Q left by the previous sample measures the same thing.
    dense = best(() -> CUDA.@sync qr!(copyto!(scratch, device)), 3)
    streamed = best(() -> cholqr2!(pm), 3)

    orth = opnorm(gram(pm) - I)
    # The Householder reference is taken on the host, where forming the thin Q
    # is one documented call. It is the same algorithm CUSOLVER runs above, and
    # this column is about accuracy rather than speed.
    Qref = qr(A).Q * Matrix{T}(I, N, k)
    denseorth = opnorm(Qref' * Qref - I)

    push!(rows, (k, 1e3 * streamed, 1e3 * dense, orth, denseorth))
    @printf("%5d %11.2fms %9.2fms %14.2e %14.2e\n", k, 1e3 * streamed, 1e3 * dense, orth, denseorth)
    Funicular.free!(pm)
    # Each pass through this loop holds a plan's page-locked slabs and two
    # device matrices. Nothing here reuses them, so let them go before the next
    # k asks for larger ones.
    A = device = scratch = Qref = pm = plan = nothing
    GC.gc()
end

record("cholqr", header, rows)

println("\n‖Q'Q-I‖ should sit near 100·N·k·eps for both. A cholqr2! column that")
println("climbs with k faster than that means the test matrix has outrun the")
println("κ ≲ 1/sqrt(eps) that CholeskyQR2 holds, and the shifted third pass is")
println("carrying it.")
