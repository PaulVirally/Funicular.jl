const PANEL_WIDTHS = (41, 21, 17, 1)

@testset "dense round trip T=$T w=$w" for T in testeltypes(), w in PANEL_WIDTHS
    N, k = 13, 41
    A = randmatrix(T, N, k)
    plan = testplan()
    pm = PanelMatrix(A; plan=plan, w=w)

    @test size(pm) == (N, k)
    @test size(pm, 1) == N
    @test size(pm, 2) == k
    @test size(pm, 3) == 1
    @test_throws ArgumentError size(pm, 0)
    @test eltype(pm) === T
    @test eltype(typeof(pm)) === T
    @test panelwidth(pm) == w
    @test npanels(pm) == cld(k, w)
    @test sum(j -> panelwidth(pm, j), 1:npanels(pm)) == k
    @test reduce(vcat, [collect(panelrange(pm, j)) for j in 1:npanels(pm)]) == collect(1:k)
    @test panelwidth(pm, npanels(pm)) == (k % w == 0 ? w : k % w)
    @test Matrix(pm) == A
    free!(pm)
end

@testset "ragged last panel is narrower" begin
    T = defaulteltype()
    plan = testplan()
    pm = PanelMatrix{T}(undef, 8, 41; plan=plan, w=17)
    @test npanels(pm) == 3
    @test [panelwidth(pm, j) for j in 1:3] == [17, 17, 7]
    @test size(panelstorage(pm.panels[3])) == (8, 7)
    @test panelrange(pm, 3) == 35:41
    @test_throws BoundsError panelrange(pm, 4)
    @test_throws BoundsError panelwidth(pm, 0)
end

@testset "narrowed host storage" begin
    T = defaulteltype()
    S = narrowed(T)
    N, k = 9, 20
    A = randmatrix(T, N, k)
    plan = testplan(host_eltype=S)
    pm = PanelMatrix(A; plan=plan, w=7)
    @test storage_eltype(pm.panels[1]) === S
    @test eltype(pm) === T
    @test Matrix(pm) == T.(S.(A))
    @test Matrix(pm) ≈ A rtol=10 * eps(real(S))

    if T <: Complex
        @test_throws ArgumentError PanelMatrix(A; plan=testplan(host_eltype=real(S)))
    end
    real_plan = testplan(host_eltype=real(S))
    @test storage_eltype(PanelMatrix(randmatrix(real(T), N, k); plan=real_plan).panels[1]) === real(S)
end

@testset "no scalar indexing" begin
    T = defaulteltype()
    plan = testplan()
    pm = PanelMatrix{T}(undef, 4, 4; plan=plan)
    @test !(pm isa AbstractMatrix)
    @test !(PanelMatrix <: AbstractArray)
    @test_throws MethodError pm[1, 1]
    @test_throws MethodError pm[1, 1]=0
    @test_throws MethodError pm[3]
end

@testset "constructor validation" begin
    T = defaulteltype()
    plan = testplan()
    @test_throws ArgumentError PanelMatrix{T}(undef, 0, 4; plan=plan)
    @test_throws ArgumentError PanelMatrix{T}(undef, 4, 0; plan=plan)
    @test_throws ArgumentError PanelMatrix{Complex}(undef, 4, 4; plan=plan)
    @test_throws DimensionMismatch copyto!(PanelMatrix{T}(undef, 4, 4; plan=plan), zeros(T, 4, 5))
end

@testset "conformality" begin
    T = defaulteltype()
    plan = testplan()
    X = PanelMatrix{T}(undef, 10, 20; plan=plan, w=7)
    Y = PanelMatrix{T}(undef, 10, 20; plan=plan, w=7)
    Z = PanelMatrix{T}(undef, 10, 20; plan=plan, w=5)
    W = PanelMatrix{T}(undef, 11, 20; plan=plan, w=7)
    @test assert_conformal(X, Y) === nothing
    @test_throws ArgumentError assert_conformal(X, Z)
    @test_throws ArgumentError assert_conformal(X, W)
end

@testset "collect guard" begin
    T = defaulteltype()
    plan = testplan()
    pm = PanelMatrix{T}(undef, 16, 16; plan=plan)
    @test_throws ArgumentError Matrix(pm; max_bytes=16 * 16 * sizeof(T) - 1)
    @test size(Matrix(pm; max_bytes=16 * 16 * sizeof(T))) == (16, 16)
end

@testset "show" begin
    T = defaulteltype()
    S = narrowed(T)
    plan = testplan()
    pm = PanelMatrix{T}(undef, 100, 41; plan=plan, w=17)
    text = sprint(show, pm)
    @test occursin("100×41", text)
    @test occursin("w = 17", text)
    @test occursin("3 panels", text)
    @test !occursin("host storage", text)

    narrow = PanelMatrix{T}(undef, 100, 41; plan=testplan(host_eltype=S), w=17)
    @test occursin("host storage $S", sprint(show, narrow))
    free!(narrow)
    @test occursin("released", sprint(show, narrow))
end

@testset "release and eviction" begin
    T = defaulteltype()
    plan = testplan(host_budget=2^20)
    pm = PanelMatrix{T}(undef, 1024, 8; plan=plan, w=4)
    used = host_bytes_in_use(plan.hostpool)
    @test used == 2 * 1024 * 4 * sizeof(T)

    @test_throws ArgumentError evict!(pm.panels[1], plan)
    pm.panels[1].dirty = true
    @test_throws ArgumentError evict!(pm.panels[1], plan)

    free!(pm)
    @test host_bytes_in_use(plan.hostpool) == 0
    @test !isresident(pm.panels[1])
    @test_throws ArgumentError materialize!(zerobuffer(T, 1024, 4), pm.panels[1], plan)
    @test_throws ArgumentError writeback!(pm.panels[1], zerobuffer(T, 1024, 4), plan)

    again = PanelMatrix{T}(undef, 1024, 8; plan=plan, w=4)
    @test host_bytes_in_use(plan.hostpool) == used
    @test host_bytes_reserved(plan.hostpool) <= plan.hostpool.budget
end

@testset "failed construction releases its blocks" begin
    T = defaulteltype()
    plan = testplan(host_budget=3 * 1024 * 4 * sizeof(T))
    pool = plan.hostpool
    @test_throws ArgumentError PanelMatrix{T}(undef, 1024, 16; plan=plan, w=4)
    @test host_bytes_in_use(pool) == 0
    @test host_bytes_free(pool) == sum(pool.cursors)
end

@testset "epoch and dirty bookkeeping" begin
    T = defaulteltype()
    plan = testplan()
    pm = PanelMatrix{T}(undef, 6, 6; plan=plan, w=3)
    @test all(p -> p.epoch == 0 && !p.dirty, pm.panels)

    copyto!(pm, randmatrix(T, 6, 6))
    @test all(p -> p.epoch == 1 && p.dirty, pm.panels)

    writeback!(pm.panels[1], zerobuffer(T, 6, 3), plan)
    @test pm.panels[1].epoch == 2
    @test pm.panels[2].epoch == 1

    materialize!(zerobuffer(T, 6, 3), pm.panels[2], plan)
    @test pm.panels[2].epoch == 1
end
