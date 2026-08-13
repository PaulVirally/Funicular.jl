# The tests only the CUDA backend can be held to: allocation and synchronization
# discipline, the page-locking of the host tier, and the stream context a panel
# function runs in. The correctness tests every backend shares are in the other
# files of this suite.

@testset "CUDA backend traits" begin
    b = current_backend()
    @test b === Funicular.cuda_backend()
    @test devicetype(b, ComplexF64) === CUDA.CuMatrix{ComplexF64}
    @test alloc_device(b, ComplexF64, (7, 3)) isa CUDA.CuMatrix{ComplexF64}
    @test supports_eltype(b, ComplexF64)
    @test transfers_are_real(b)
    @test record_event(b, make_queue(b)) isa CUDA.CuEvent

    # A sweep's three queues share the compute stream for CUDA.jl's per-array
    # bookkeeping, so that moving a staging buffer between them does not make
    # the host wait. A queue made on its own is its own owner.
    up, compute, down = Funicular.make_sweep_queues(b)
    @test up.owner === compute.stream === compute.owner === down.owner
    @test up.stream !== compute.stream !== down.stream
    solo = make_queue(b)
    @test solo.owner === solo.stream
end

@testset "host slabs are page-locked" begin
    plan = testplan()
    pm = PanelMatrix{ComplexF64}(undef, 64, 8; plan=plan, w=4)
    address = convert(Ptr{Nothing}, pointer(panelstorage(pm.panels[1])))
    # is_pinned is internal to CUDA.jl. It checks a property of the slab rather
    # than anything Funicular does, so a rename reports here instead of failing.
    if isdefined(CUDA, :is_pinned)
        @test CUDA.is_pinned(address)
    else
        @info "CUDA.is_pinned is gone, so the page-locking of the host slab goes unchecked here"
    end
end

@testset "panel functions run on the sweep's stream" begin
    T = ComplexF64
    plan = testplan()
    pm = PanelMatrix(randmatrix(T, 32, 12); plan=plan, w=5)
    outer = CUDA.stream()
    seen = CUDA.CuStream[]

    foreachpanel(pm; write=false) do j, panel
        push!(seen, CUDA.stream())
    end

    @test length(seen) == npanels(pm)
    # One stream for the whole sweep, and not the one the caller was on: a
    # CUBLAS call or a kernel launched from a panel function has to land on the
    # compute queue rather than on the task-local default stream.
    @test all(stream -> stream === first(seen), seen)
    @test first(seen) !== outer
end

# CUDA.jl passes the scalar arguments of a CUBLAS call by device pointer, so it
# stages each of them in device memory: 32 bytes for a gemm's α and β in
# ComplexF64, 16 for a trsm's α. That is the vendor's allocation and no sweep can
# avoid it while it uses the vendor's BLAS. Measuring it here rather than
# hard-coding it holds an operation to exactly what its CUBLAS calls cost, so
# anything Funicular allocated inside its own loop still shows up.
function blas_staging(T, k)
    b = current_backend()
    block = alloc_device(b, T, (32, k))
    small = alloc_device(b, T, (k, k))
    factor = todevice(b, Matrix(UpperTriangular(randmatrix(T, k, k) + 4I)))
    fill!(block, one(T))
    for warmup in 1:2
        gram_accumulate!(b, small, block, block, one(T), zero(T))
        rdiv_upper!(b, block, factor)
    end
    gemm = CUDA.@allocated gram_accumulate!(b, small, block, block, one(T), zero(T))
    trsm = CUDA.@allocated rdiv_upper!(b, block, factor)
    (gemm, trsm)
end

@testset "sweeps allocate no device memory after warmup" begin
    T = ComplexF64
    N, k, w = 256, 24, 7
    plan = testplan(device_budget=64 * 2^20, host_budget=64 * 2^20)
    X = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    # cholqr2! is counted below at two passes, so it gets a matrix whose
    # conditioning cannot send it down the shifted third pass.
    Z = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    G = deviceoperator(randmatrix(T, N, N))
    panels = npanels(X)
    blocks = nsteps(rowtraversal(X))
    double!(_, panel) = (panel .*= 2)
    copying!(_, source, target) = (target .= source)

    for warmup in 1:2
        sweep!(double!, updatepanels(X))
        sweep!(copying!, readpanels(X), writepanels(Y))
        gram(X)
        panelmul!(Y, G, X)
        cholqr2!(Z)
    end
    gemm, trsm = blas_staging(T, k)
    @info "CUBLAS stages its scalars in device memory" gemm trsm panels blocks

    # Data movement reaches no BLAS at all, so its allocation has to be zero.
    @test CUDA.@allocated(sweep!(double!, updatepanels(X))) == 0
    @test CUDA.@allocated(sweep!(copying!, readpanels(X), writepanels(Y))) == 0

    # One gemm per panel, and one per row block for gram. cholqr2! is two passes
    # of a gram and a triangular solve over the same row blocks.
    @test CUDA.@allocated(panelmul!(Y, G, X)) == panels * gemm
    @test CUDA.@allocated(gram(X)) == blocks * gemm
    @test CUDA.@allocated(cholqr2!(Z)) == 2 * blocks * (gemm + trsm)
end

@testset "a narrowed sweep allocates nothing either" begin
    T, S = ComplexF64, ComplexF32
    N, k, w = 256, 24, 7
    plan = testplan(device_budget=64 * 2^20, host_budget=64 * 2^20, host_eltype=S)
    X = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    blocks = nsteps(rowtraversal(X))
    double!(_, panel) = (panel .*= 2)

    for warmup in 1:2
        sweep!(double!, updatepanels(X))
        gram(X)
    end
    gemm, _ = blas_staging(T, k)

    # The precision conversion works out of pool-owned staging buffers, so the
    # narrow copy has somewhere to land without allocating mid-sweep. The widen
    # is a kernel and allocates nothing either.
    @test CUDA.@allocated(sweep!(double!, updatepanels(X))) == 0
    @test CUDA.@allocated(gram(X)) == blocks * gemm
    @test device_bytes_allocated(plan) >= 2 * (panel_bytes(N, w, T) + panel_bytes(N, w, S))
end
