const SWEEP_WIDTHS = (23, 8, 7, 1)

@testset "read-only sweep sees every panel w=$w nbuffers=$nb" for w in SWEEP_WIDTHS, nb in (1, 2)
    T = defaulteltype()
    N, k = 9, 23
    A = randmatrix(T, N, k)
    plan = testplan(backend=jittery_backend())
    pm = PanelMatrix(A; plan=plan, w=w)
    order = Int[]
    seen = Vector{Matrix{T}}(undef, npanels(pm))

    foreachpanel(pm; write=false, nbuffers=nb) do j, panel
        push!(order, j)
        seen[j] = host(panel)
        @test size(panel) == (N, panelwidth(pm, j))
    end

    @test order == 1:npanels(pm)
    @test all(j -> seen[j] == A[:, panelrange(pm, j)], eachindex(seen))
    @test Matrix(pm) == A
end

@testset "read-write sweep w=$w nbuffers=$nb" for w in SWEEP_WIDTHS, nb in (1, 2)
    T = defaulteltype()
    N, k = 9, 23
    A = randmatrix(T, N, k)
    plan = testplan(backend=jittery_backend())
    pm = PanelMatrix(A; plan=plan, w=w)

    foreachpanel(pm; nbuffers=nb) do j, panel
        panel .= 2 .* panel .+ T(j)
    end

    expected = 2 .* A .+ T.([j for _ in 1:N, j in map(c -> cld(c, w), 1:k)])
    @test Matrix(pm) == expected
end

@testset "produce sweep w=$w nbuffers=$nb" for w in SWEEP_WIDTHS, nb in (1, 2)
    T = defaulteltype()
    N, k = 9, 23
    A = randmatrix(T, N, k)
    plan = testplan(backend=jittery_backend())
    X = PanelMatrix(A; plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)

    sweep!(readpanels(X), writepanels(Y); nbuffers=nb) do j, source, target
        target .= 3 .* source
    end

    @test Matrix(Y) == 3 .* A
    @test Matrix(X) == A
end

@testset "narrowed host storage streams through the sweep" begin
    T = defaulteltype()
    S = narrowed(T)
    N, k, w = 9, 20, 6
    A = randmatrix(T, N, k)
    plan = testplan(backend=jittery_backend(), host_eltype=S)
    pm = PanelMatrix(A; plan=plan, w=w)
    narrowed_A = T.(S.(A))

    foreachpanel(pm) do j, panel
        @test eltype(panel) === T
        panel .*= 2
    end

    @test Matrix(pm) == T.(S.(2 .* narrowed_A))
end

@testset "narrowing stages through buffers out of the pool" begin
    T = defaulteltype()
    S = narrowed(T)
    N, k, w = 9, 20, 6
    plan = testplan(host_eltype=S)
    pm = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)

    # Only the narrow bytes cross the bus, so a sweep over a narrowed matrix
    # holds two buffers of each eltype and allocates neither one in its loop.
    foreachpanel(pm) do j, panel end
    @test device_bytes_allocated(plan) == 2 * (panel_bytes(N, w, T) + panel_bytes(N, w, S))

    foreachpanel(pm) do j, panel end
    @test device_bytes_allocated(plan) == 2 * (panel_bytes(N, w, T) + panel_bytes(N, w, S))

    # The row traversal converts a whole block at a time rather than a panel at
    # a time, so it gets its own pass.
    before = Matrix(pm)
    sweep!(updatepanels(pm); traversal=rowtraversal(pm)) do _, block
        block .*= 2
    end
    @test Matrix(pm) == T.(S.(2 .* before))
end

@testset "one panel's buffer never leaks into another nbuffers=$nb" for nb in (1, 2)
    T = defaulteltype()
    N, k, w = 6, 23, 5
    A = randmatrix(T, N, k)
    stamped = reduce(hcat, [fill(T(j), N, min(w, k - (j - 1) * w)) for j in 1:cld(k, w)])
    for _ in 1:repetitions()
        plan = testplan(backend=jittery_backend())
        pm = PanelMatrix(A; plan=plan, w=w)
        seen = Vector{Matrix{T}}(undef, npanels(pm))

        foreachpanel(pm; nbuffers=nb) do j, panel
            seen[j] = host(panel)
            panel .= T(j)
        end

        @test all(j -> seen[j] == A[:, panelrange(pm, j)], eachindex(seen))
        @test Matrix(pm) == stamped
    end
end

@testset "produce sweep keeps its input and output buffers apart" begin
    T = defaulteltype()
    N, k, w = 6, 23, 5
    A = randmatrix(T, N, k)
    for _ in 1:repetitions()
        plan = testplan(backend=jittery_backend())
        X = PanelMatrix(A; plan=plan, w=w)
        Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)

        sweep!(readpanels(X), writepanels(Y)) do j, source, target
            target .= T(j)
            @test host(source) == A[:, panelrange(X, j)]
        end

        @test Matrix(X) == A
    end
end

@testset "nbuffers 1 and 2 move data bitwise alike w=$w" for w in SWEEP_WIDTHS
    T = defaulteltype()
    N, k = 12, 23
    A = randmatrix(T, N, k)
    results = map((1, 2)) do nb
        plan = testplan(backend=jittery_backend())
        pm = PanelMatrix(A; plan=plan, w=w)
        foreachpanel(pm; nbuffers=nb) do j, panel
            panel .= sqrt.(abs.(panel)) .* panel
        end
        Matrix(pm)
    end
    @test results[1] == results[2]
end

@testset "nbuffers 1 and 2 reduce bitwise alike w=$w" for w in SWEEP_WIDTHS
    T = defaulteltype()
    N, k = 12, 23
    A = randmatrix(T, N, k)
    sums = map((1, 2)) do nb
        plan = testplan(backend=jittery_backend())
        pm = PanelMatrix(A; plan=plan, w=w)
        total = Ref(zero(T))
        foreachpanel(pm; write=false, nbuffers=nb) do j, panel
            total[] += sum(panel)
        end
        total[]
    end
    @test sums[1] === sums[2]
end

@testset "dirty and epoch bookkeeping" begin
    T = defaulteltype()
    N, k, w = 8, 20, 6
    plan = testplan()
    pm = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    @test all(p -> p.epoch == 0 && !p.dirty, pm.panels)

    foreachpanel(pm; write=false) do j, panel end
    @test all(p -> p.epoch == 0 && !p.dirty, pm.panels)

    foreachpanel(pm) do j, panel
        fill!(panel, zero(T))
    end
    @test all(p -> p.epoch == 1 && p.dirty, pm.panels)

    foreachpanel(pm) do j, panel end
    @test all(p -> p.epoch == 2, pm.panels)

    X = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    sweep!(readpanels(X), writepanels(pm)) do j, source, target
        target .= source
    end
    @test all(p -> p.epoch == 0 && !p.dirty, X.panels)
    @test all(p -> p.epoch == 3, pm.panels)
end

@testset "sweeps give their buffers back to the pool" begin
    T = defaulteltype()
    N, k, w = 8, 20, 5
    plan = testplan()
    X = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    @test device_bytes_allocated(plan) == 0

    foreachpanel(X; write=false) do j, panel end
    pair = 2 * panel_bytes(N, w, T)
    @test device_bytes_allocated(plan) == pair

    foreachpanel(X) do j, panel end
    @test device_bytes_allocated(plan) == pair

    sweep!(readpanels(X), writepanels(Y)) do j, source, target
        target .= source
    end
    @test device_bytes_allocated(plan) == 2 * pair

    sweep!(readpanels(X), writepanels(Y)) do j, source, target
        target .= source
    end
    @test device_bytes_allocated(plan) == 2 * pair
end

@testset "nbuffers is capped at the panel count" begin
    T = defaulteltype()
    plan = testplan()
    pm = PanelMatrix(randmatrix(T, 8, 20); plan=plan, w=20)
    @test npanels(pm) == 1
    foreachpanel(pm; nbuffers=2) do j, panel end
    @test device_bytes_allocated(plan) == panel_bytes(8, 20, T)
end

@testset "a failing panel function surfaces and frees its buffers" begin
    T = defaulteltype()
    N, k, w = 8, 20, 5
    plan = testplan()
    pm = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)

    @test_throws QUEUE_FAILURE foreachpanel(pm) do j, panel
        j == 2 && error("boom")
    end
    @test device_bytes_allocated(plan) == 2 * panel_bytes(N, w, T)

    foreachpanel(pm; write=false) do j, panel end
    @test device_bytes_allocated(plan) == 2 * panel_bytes(N, w, T)
end

@testset "sweep validation" begin
    T = defaulteltype()
    plan = testplan()
    X = PanelMatrix{T}(undef, 10, 20; plan=plan, w=7)
    Y = PanelMatrix{T}(undef, 10, 20; plan=plan, w=7)
    Z = PanelMatrix{T}(undef, 10, 20; plan=plan, w=5)
    noop = (j, panels...) -> nothing

    @test_throws ArgumentError sweep!(noop)
    @test_throws ArgumentError sweep!(noop, readpanels(X), writepanels(Z))
    @test_throws ArgumentError sweep!(noop, readpanels(X), writepanels(X))
    @test_throws ArgumentError sweep!(noop, readpanels(X); nbuffers=0)

    other = PanelMatrix{T}(undef, 10, 20; plan=testplan(), w=7)
    @test_throws ArgumentError sweep!(noop, readpanels(X), writepanels(other))
    @test sweep!(noop, readpanels(X), writepanels(Y)) === nothing
end

@testset "row blocks tile the rows exactly N=$N height=$h" for N in (12, 13, 1), h in (1, 3, 5, 20)
    traversal = RowSteps(N, h)
    @test reduce(vcat, traversal.blocks) == 1:N
    @test all(block -> length(block) <= min(h, N), traversal.blocks)
    @test length(first(traversal.blocks)) == min(h, N)
    @test nsteps(traversal) == cld(N, min(h, N))
end

@testset "row block height matches the panel buffer" begin
    @test row_block_height(120, 24, 24) == 120
    @test row_block_height(120, 24, 6) == 30
    @test row_block_height(120, 24, 1) == 5
    @test row_block_height(9, 23, 1) == 1
    @test row_block_height(120, 24, 100) == 120
end

@testset "read-only row sweep sees the whole matrix w=$w nbuffers=$nb" for w in SWEEP_WIDTHS, nb in (1, 2)
    T = defaulteltype()
    N, k = 13, 23
    A = randmatrix(T, N, k)
    plan = testplan(backend=jittery_backend())
    pm = PanelMatrix(A; plan=plan, w=w)
    traversal = rowtraversal(pm)
    seen = zeros(T, N, k)

    sweep!(readpanels(pm); traversal=traversal, nbuffers=nb) do b, block
        @test size(block) == (length(traversal.blocks[b]), k)
        seen[traversal.blocks[b], :] = host(block)
    end

    @test seen == A
    @test Matrix(pm) == A
end

@testset "read-write row sweep w=$w nbuffers=$nb" for w in SWEEP_WIDTHS, nb in (1, 2)
    T = defaulteltype()
    N, k = 13, 23
    A = randmatrix(T, N, k)
    R = Matrix(UpperTriangular(randmatrix(T, k, k) + 4I))
    backend = jittery_backend()
    plan = testplan(backend=backend)
    pm = PanelMatrix(A; plan=plan, w=w)
    factor = todevice(backend, R)

    sweep!(updatepanels(pm); traversal=rowtraversal(pm), nbuffers=nb) do _, block
        rdiv_upper!(backend, block, factor)
    end

    @test Matrix(pm) ≈ A / UpperTriangular(R) rtol=blastol(T, N, k)
end

@testset "row sweeps move data the same way at nbuffers 1 and 2 w=$w" for w in SWEEP_WIDTHS
    T = defaulteltype()
    N, k = 13, 23
    A = randmatrix(T, N, k)
    results = map((1, 2)) do nb
        plan = testplan(backend=jittery_backend())
        pm = PanelMatrix(A; plan=plan, w=w)
        sweep!(updatepanels(pm); traversal=rowtraversal(pm), nbuffers=nb) do _, block
            block .= 2 .* block
        end
        Matrix(pm)
    end
    @test results[1] == results[2]
end

@testset "a repeated panel is refused when the sweep writes" begin
    T = defaulteltype()
    plan = testplan()
    pm = PanelMatrix(randmatrix(T, 8, 20); plan=plan, w=5)
    repeated = PanelSteps([1, 2, 1, 2], [1, 2, 1, 2])
    noop = (label, panels...) -> nothing

    @test_throws ArgumentError sweep!(noop, updatepanels(pm); traversal=repeated)
    @test sweep!(noop, readpanels(pm); traversal=repeated) === nothing
end

@testset "a custom traversal labels its steps" begin
    T = defaulteltype()
    N, k, w = 8, 20, 5
    A = randmatrix(T, N, k)
    plan = testplan(backend=jittery_backend())
    pm = PanelMatrix(A; plan=plan, w=w)
    traversal = PanelSteps([2, 1, 2], [:second, :first, :again])
    labels = Symbol[]

    sweep!(readpanels(pm); traversal=traversal) do label, panel
        push!(labels, label)
        @test host(panel) == A[:, panelrange(pm, label === :first ? 1 : 2)]
    end
    @test labels == [:second, :first, :again]
end

@testset "the panel feed is bounded and in order" begin
    T = defaulteltype()
    N, k, w = 6, 20, 5
    plan = testplan()
    pm = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    feed = stage_channel(PanelSteps(npanels(pm)), (readpanels(pm),); depth=2)
    @test [take!(feed) for _ in 1:npanels(pm)] == 1:npanels(pm)
    @test stage_host!(pm.panels[1], plan) === pm.panels[1]

    blocked = stage_channel(PanelSteps(npanels(pm)), (readpanels(pm),); depth=1)
    yield()
    @test Base.n_avail(blocked) <= 2
    close(blocked)
end
