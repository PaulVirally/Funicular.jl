# The internal vendor API, run against every backend the machine offers.
# The CPU reference backend's own quirks are in test_device_api.jl.

@testset "backend traits" begin
    b = current_backend()
    T = defaulteltype()
    buffer = alloc_device(b, T, (7, 3))
    @test buffer isa devicetype(b, T)
    @test size(buffer) == (7, 3)
    @test eltype(buffer) === T
    @test all(S -> supports_eltype(b, S), testeltypes())
    @test supports_eltype(b, ComplexF64) == supports_eltype(b, Float64)
    @test ka_backend(b) isa KernelAbstractions.Backend
    @test transfers_are_real(b) isa Bool
end

@testset "host slab" begin
    b = current_backend()
    T = defaulteltype()
    slab = alloc_host_slab(b, 4096)
    @test sizeof(slab) == 4096
    @test reinterpret(UInt, slab.ptr) % Funicular.SLAB_ALIGN == 0

    A = slab_matrix(slab, T, (4, 8), 0)
    B = slab_matrix(slab, T, (4, 8), 0)
    A[3, 5] = one(T)
    @test B[3, 5] == one(T)

    C = slab_matrix(slab, T, (4, 8), 4 * 8 * sizeof(T))
    fill!(C, zero(T))
    @test A[3, 5] == one(T)
    @test_throws ArgumentError slab_matrix(slab, T, (4, 8), 4000)
    @test_throws ArgumentError slab_matrix(slab, T, (4, 8), -1)
end

@testset "synchronous copies $T" for T in testeltypes()
    b = current_backend()
    src = randmatrix(T, 6, 4)
    dev = alloc_device(b, T, (6, 4))
    out = zeros(T, 6, 4)
    @test sync_event(b, h2d!(dev, src, b)) === nothing
    @test host(dev) == src
    d2h!(out, dev, b)
    @test out == src

    S = T <: Complex ? Complex{Float16} : Float16
    narrow = alloc_device(b, S, (6, 4))
    h2d!(narrow, src, b)
    d2h!(out, narrow, b)
    @test out == T.(S.(src))

    @test_throws DimensionMismatch h2d!(alloc_device(b, T, (6, 5)), src, b)
end

@testset "queue ordering" begin
    b = jittery_backend()
    queue = make_queue(b)
    order = Int[]
    for i in 1:64
        submit!(b, queue, () -> push!(order, i))
    end
    sync_queue(b, queue)
    @test order == collect(1:64)
end

@testset "cross-queue ordering through events" begin
    b = jittery_backend()
    T = defaulteltype()
    for rep in 1:repetitions()
        up, down = make_queue(b), make_queue(b)
        src = fill(T(rep), 5, 3)
        dev = alloc_device(b, T, (5, 3))
        out = zeros(T, 5, 3)
        h2d!(dev, src, b; queue=up)
        wait_event(b, down, record_event(b, up))
        d2h!(out, dev, b; queue=down)
        sync_queue(b, down)
        @test out == src
    end
end

@testset "event and queue synchronization" begin
    b = jittery_backend()
    queue = make_queue(b)
    counter = Ref(0)
    for _ in 1:8
        submit!(b, queue, () -> counter[] += 1)
    end
    event = record_event(b, queue)
    sync_event(b, event)
    @test counter[] == 8

    # sync_all drains the queues that are still reachable, so this one has to be
    # held across the call for the drain to include it.
    submit!(b, queue, () -> counter[] += 1)
    GC.@preserve queue sync_all(b)
    @test counter[] == 9
end

@testset "failed queue work surfaces" begin
    b = current_backend()
    queue = make_queue(b)
    @test_throws QUEUE_FAILURE begin
        submit!(b, queue, () -> error("boom"))
        sync_queue(b, queue)
    end
end
