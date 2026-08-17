# # Mini RSVD on a GPU with a disk tier
#
# This is a complete, runnable program. It computes the top singular values of a
# matrix-free operator whose tall-skinny basis does not fit in the host memory
# it is given, so panels cycle through the disk tier while the GPU works, and it
# checks the answer against a spectrum it knows in advance.
#
# Run it with a CUDA environment on the path:
#
# ```
# julia --project=test/cuda --startup-file=no docs/demo/minirsvd.jl
# ```
#
# Almost all of it is the public API. The only things reached through
# `Funicular.` are the traits, the counters at the end that show the disk tier
# was used, and `free!`.
#
# It needs HDF5 loaded for the disk tier, and it writes about three gigabytes
# under `mktempdir()`. Point `scratch_dir` at the fast scratch filesystem a real
# run would use rather than at a small `/tmp`.

using CUDA
using Funicular
using HDF5
using LinearAlgebra
using Random

# ## Sizes
#
# `N` is the length of one vector the operator acts on. A real problem puts it in
# the tens of millions rather than at the million used here. Nothing about the
# program changes with it, only how long it takes and how much has to spill. `K`
# is how many columns the randomized method keeps.

const T = ComplexF64
const N = 1 << 20
const K = 64
const RANK = 8

# ## The operator
#
# A stand-in for an expensive matrix-free operator: Hermitian, applied without
# ever forming a matrix, with a spectrum chosen so the answer is known. `U` is
# orthonormal and `N × RANK`, so `G = U diag(σ) U' + ε I` has singular values
# `σ .+ ε` at the top and `ε` everywhere below.
#
# Funicular needs four methods from an operator, plus three traits that are
# optional. `panel_capable` says that `mul!` takes a whole panel rather than one
# column, and `workspace_bytes` is the device memory the operator holds for
# itself, which the plan then holds back from the buffer pool. Note that an
# operator that understates `workspace_bytes` will cause a run to die of an
# out-of-memory error halfway through.

struct LowRankPlusNoise{M,V,C}
    U::M
    σ::V
    ε::Float64
    work::C
end

Base.size(G::LowRankPlusNoise) = (size(G.U, 1), size(G.U, 1))
Base.size(G::LowRankPlusNoise, d::Integer) = d <= 2 ? size(G.U, 1) : 1
Base.eltype(G::LowRankPlusNoise) = eltype(G.U)
Base.adjoint(G::LowRankPlusNoise) = G

Funicular.panel_capable(::LowRankPlusNoise) = true
Funicular.ishermitian_op(::LowRankPlusNoise) = true
Funicular.workspace_bytes(G::LowRankPlusNoise) = sizeof(G.U) + sizeof(G.work) + sizeof(G.σ)

function LinearAlgebra.mul!(Y::AbstractMatrix, G::LowRankPlusNoise, X::AbstractMatrix)
    C = view(G.work, :, axes(X, 2))
    mul!(C, G.U', X)
    C .*= G.σ
    mul!(Y, G.U, C)
    Y .+= G.ε .* X
    Y
end

function LinearAlgebra.mul!(y::AbstractVector, G::LowRankPlusNoise, x::AbstractVector)
    c = view(G.work, :, 1)
    mul!(c, G.U', x)
    c .*= G.σ
    mul!(y, G.U, c)
    y .+= G.ε .* x
    y
end

# ## Building it
#
# The panel width is fixed here so the numbers below are easy to follow. Left to
# itself the plan picks the widest panel its device budget allows, aiming at one
# to four gigabytes per panel.

const W = 8

Random.seed!(0xdeadbeef)
basis = CuArray(qr(randn(T, N, RANK)).Q * Matrix{T}(I, N, RANK))
spectrum = 2.0 .^ -(0:(RANK - 1))
noise = 1e-6
G = LowRankPlusNoise(basis, CuArray(T.(spectrum)), noise, similar(basis, RANK, W))

# `check_operator` applies `G` and its adjoint to a few random probes and
# complains if the two disagree, if the shapes are wrong, or if either trait is
# not true. Run it once when you write an operator, not in the inner loop.

check_operator(G; backend=Funicular.cuda_backend())

# ## The plan
#
# `device_budget` is what Funicular may allocate on the GPU: two staging buffers
# per matrix in a sweep plus the operator's workspace. `host_budget` caps the
# page-locked host tier. It is deliberately smaller here than the three `N × K`
# matrices this program holds, so it forces panels out to the disk tier rather
# than letting the whole run sit in memory.

backend = Funicular.cuda_backend()
plan = ResidencyPlan(backend=backend,
                     device_budget=2 * 2^30,
                     host_budget=2 * 2^30,
                     workspace_bytes=Funicular.workspace_bytes(G),
                     scratch_dir=mktempdir(),
                     panel_width=W)

println("one panel is ", Base.format_bytes(N * W * sizeof(T)),
        "; one matrix is ", Base.format_bytes(N * K * sizeof(T)))

# ## The random test matrix
#
# `Ω` is the Gaussian a randomized method starts from. It is never stored: each
# column is regenerated from the seed and the column index when a sweep asks for
# the panel holding it, and dropped when the host tier needs the room. At the
# sizes this package is for, that saves hundreds of gigabytes of scratch space.

Ω = GhostPanels(T, N, K; plan=plan, seed=0xdeadbeef)

# ## Subspace iteration
#
# The randomized range finder below is three operator applications with an
# orthonormalization after each. Orthonormalizing between every application
# rather than after several of them keeps the conditioning of `Y` inside what
# CholeskyQR2 can hold.

function rangefinder(G, Ω, plan; iterations=2)
    Y = PanelMatrix{eltype(Ω)}(undef, size(Ω)...; plan=plan)
    Z = PanelMatrix{eltype(Ω)}(undef, size(Ω)...; plan=plan)
    panelmul!(Y, G, Ω)
    cholqr2!(Y)
    for _ in 1:iterations
        panelmul!(Z, G, Y)
        cholqr2!(Z)
        Y, Z = Z, Y
    end
    Y, Z
end

Y, Z = rangefinder(G, Ω, plan)

# `project` forms the `K × K` operator `Y' G Y` without ever storing `G Y`. It
# is the expensive one: `p (p + 1)` panel uploads for `p` panels, against two
# sweeps for `panelmul!` followed by `gram`. Use it when you cannot afford the
# second `N × K` matrix, and `panelmul!` plus `gram` when you can.

S = project(Y, G)

# ## The answer
#
# `S` is `K × K` and dense, so the small eigenproblem at the end of every
# randomized method is a plain LAPACK call on the host.

computed = svdvals(S)[1:RANK]
expected = spectrum .+ noise

println("\n  computed        expected      relative error")
for (c, e) in zip(computed, expected)
    println(rpad(round(c, sigdigits=8), 16), rpad(round(e, sigdigits=8), 16),
            round(abs(c - e) / e, sigdigits=3))
end

@assert isapprox(computed, expected; rtol=1e-6)
@assert opnorm(gram(Y) - I) < 100 * N * K * eps(real(T))

# ## What the tiers did
#
# Panels moved between host and disk because `host_budget` could not hold three
# matrices at once. If both counters are zero, the run stayed in host memory and
# the disk tier was never exercised. Raise `K` or lower `host_budget` if you
# want to see it work.

println("\npanel reads from disk:  ", Funicular.disk_reads(Y) + Funicular.disk_reads(Z))
println("panel writes to disk:   ", Funicular.disk_writes(Y) + Funicular.disk_writes(Z))

# ## Keeping the basis
#
# `save` writes a `PanelMatrix` as one HDF5 file with a chunk per panel, and
# `load` opens it with the panels still on disk, streaming up as they are asked
# for. That file is the matrix's cold tier rather than a copy of it, so an
# operation that modifies a panel writes it back.

path = joinpath(mktempdir(), "basis.h5")
save(Y, path)
println("\nsaved ", Base.format_bytes(filesize(path)), " to ", path)

reloaded = load(PanelMatrix, path; plan=plan, readonly=true)
@assert opnorm(gram(reloaded) - I) < 100 * N * K * eps(real(T))
println("reloaded and still orthonormal")

Funicular.free!(reloaded)
Funicular.free!(Y)
Funicular.free!(Z)

# ## Without a GPU
#
# The same program runs on the reference backend: drop `using CUDA`, replace the
# `backend` line with `backend = CPUBackend()`, and build `basis` and the
# operator's spectrum and workspace as ordinary `Array`s. The vendor API is
# there so that nothing else has to change. It will not be fast, since the
# reference backend's device is host memory and its queues are tasks, but the
# logic it runs is the same as on the GPU.
