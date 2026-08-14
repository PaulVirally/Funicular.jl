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

@testset "copycols! from a host matrix T=$T w=$w" for T in testeltypes(), w in PANEL_WIDTHS
    N, k = 13, 41
    A = randmatrix(T, N, k)
    plan = testplan()
    for dcols in (1:9, 12:28, 33:41, 1:41, 20:20)
        dest = PanelMatrix(A; plan=plan, w=w)
        B = randmatrix(T, N, length(dcols))
        reference = copy(A)
        reference[:, dcols] = B

        @test copycols!(dest, dcols, B) === dest
        @test Matrix(dest) == reference
        @test all(panel -> panel.pins == 0, dest.panels)
        free!(dest)
    end
end

@testset "copycols! between panel matrices T=$T widths=$widths" for T in testeltypes(), widths in ((7, 7), (7, 5), (23, 4), (1, 6))
    sw, dw = widths
    N, sk, dk = 12, 23, 30
    A = randmatrix(T, N, sk)
    B = randmatrix(T, N, dk)
    plan = testplan()
    for (dcols, scols) in ((1:sk, 1:sk), (8:20, 3:15), (25:30, 1:6), (14:14, 23:23))
        src = PanelMatrix(A; plan=plan, w=sw)
        dest = PanelMatrix(B; plan=plan, w=dw)
        reference = copy(B)
        reference[:, dcols] = A[:, scols]

        @test copycols!(dest, dcols, src, scols) === dest
        @test Matrix(dest) == reference
        @test Matrix(src) == A
        @test all(panel -> panel.pins == 0, src.panels)
        @test all(panel -> panel.pins == 0, dest.panels)
        free!(src)
        free!(dest)
    end
end

@testset "copycols! grows a sketch" begin
    T = defaulteltype()
    N, m = 20, 9
    plan = testplan()
    A = randmatrix(T, N, m)
    narrow = PanelMatrix(A; plan=plan, w=4)
    wide = PanelMatrix{T}(undef, N, 2m; plan=plan, w=5)

    copycols!(wide, 1:m, narrow, 1:m)
    copycols!(wide, (m + 1):2m, randmatrix(T, N, m))
    @test Matrix(wide)[:, 1:m] == A
end

@testset "a ghost matrix is a copycols! source" begin
    T = defaulteltype()
    N, k, w = 20, 8, 3
    plan = testplan()
    Ω = GhostPanels(T, N, k; plan=plan, seed=0xC0FFEE, w=w)
    A = Matrix(Ω)
    dest = PanelMatrix{T}(undef, N, 2k; plan=plan, w=5)

    copycols!(dest, 1:k, Ω, 1:k)
    copycols!(dest, (k + 1):2k, Ω, 1:k)
    collected = Matrix(dest)
    @test collected[:, 1:k] == A
    @test collected[:, (k + 1):2k] == A
    # Reading a ghost leaves it a ghost, with nothing written and nothing dirty.
    @test occursin("ghost", sprint(show, Ω))
    @test !any(panel -> panel.dirty, Ω.panels)
    @test Matrix(Ω) == A
end

@testset "copyto! repartitions a matrix" begin
    T = defaulteltype()
    N, k = 9, 20
    A = randmatrix(T, N, k)
    plan = testplan()
    src = PanelMatrix(A; plan=plan, w=7)
    dest = PanelMatrix{T}(undef, N, k; plan=plan, w=4)

    @test copyto!(dest, src) === dest
    @test panelwidth(dest) == 4
    @test Matrix(dest) == A
    @test_throws DimensionMismatch copyto!(PanelMatrix{T}(undef, N, k + 1; plan=plan, w=4), src)
    @test_throws DimensionMismatch copyto!(PanelMatrix{T}(undef, N + 1, k; plan=plan, w=4), src)
end

@testset "copycols! through narrowed host storage" begin
    T = defaulteltype()
    S = narrowed(T)
    N, k = 9, 12
    A = randmatrix(T, N, k)
    src = PanelMatrix(A; plan=testplan(host_eltype=S), w=5)
    dest = PanelMatrix{T}(undef, N, k; plan=testplan(), w=4)

    copycols!(dest, 1:k, src, 1:k)
    @test Matrix(dest) == T.(S.(A))

    stored = PanelMatrix{T}(undef, N, k; plan=testplan(host_eltype=S), w=3)
    copycols!(stored, 1:k, A)
    @test Matrix(stored) == T.(S.(A))
end

@testset "copycols! validation" begin
    T = defaulteltype()
    N, k = 8, 12
    plan = testplan()
    dest = PanelMatrix{T}(undef, N, k; plan=plan, w=5)
    src = PanelMatrix(randmatrix(T, N, k); plan=plan, w=4)
    A = randmatrix(T, N, 4)
    Ω = GhostPanels(T, N, k; plan=plan, w=4)

    @test_throws ArgumentError copycols!(dest, 10:14, A)
    @test_throws ArgumentError copycols!(dest, 0:3, A)
    @test_throws DimensionMismatch copycols!(dest, 1:4, randmatrix(T, N, 5))
    @test_throws DimensionMismatch copycols!(dest, 1:4, randmatrix(T, N + 1, 4))
    @test_throws ArgumentError copycols!(dest, 1:4, src, 10:13)
    @test_throws DimensionMismatch copycols!(dest, 1:4, src, 1:5)
    @test_throws DimensionMismatch copycols!(dest, 1:4, PanelMatrix{T}(undef, N + 1, k; plan=plan, w=4), 1:4)
    @test_throws ArgumentError copycols!(dest, 1:4, dest, 5:8)
    @test_throws ArgumentError copycols!(Ω, 1:4, A)
    @test_throws ArgumentError copycols!(Ω, 1:4, src, 1:4)
    # Every one of those is refused before anything is staged.
    @test all(panel -> panel.pins == 0, dest.panels)
    @test all(panel -> panel.pins == 0, src.panels)
end

@testset "similar takes the plan and the width along" begin
    T = defaulteltype()
    S = last(testeltypes())
    N, k = 40, 23
    plan = testplan()
    pm = PanelMatrix{T}(undef, N, k; plan=plan, w=7)

    like = similar(pm)
    @test size(like) == (N, k)
    @test eltype(like) === T
    @test panelwidth(like) == 7
    @test Funicular.plan(like) === plan
    @test Funicular.plan(like) === Funicular.plan(pm)

    @test eltype(similar(pm, S)) === S
    @test size(similar(pm, S)) == (N, k)
    @test panelwidth(similar(pm, S)) == 7

    tall = similar(pm, T, (N + 17, k))
    @test size(tall) == (N + 17, k)
    @test panelwidth(tall) == 7
    # A width wider than the matrix would leave the first panel ragged.
    @test panelwidth(similar(pm, T, (N, 3))) == 3
end

@testset "a similar companion is panelmul! conformal" begin
    T = defaulteltype()
    N, m, k = 40, 17, 23
    plan = testplan()
    A = randmatrix(T, N, k)
    G = randmatrix(T, m, N)
    X = PanelMatrix(A; plan=plan, w=7)
    Y = similar(X, T, (m, k))

    panelmul!(Y, deviceoperator(G), X)
    @test Matrix(Y) ≈ G * A rtol=blastol(T, N, k)
end

if HAS_HDF5
    @testset "copycols! stages both sides through the disk tier" begin
        T = defaulteltype()
        N, k, w = 10, 24, 4
        A = randmatrix(T, N, k)
        B = randmatrix(T, N, k)
        mktempdir() do dir
            # Room for the two panels one run holds and no more, so every other
            # panel of both matrices is out on disk while the copy runs.
            plan = testplan(scratch_dir=dir, panel_width=w,
                            host_budget=panelbudget(2, N, w, T))
            src = PanelMatrix(A; plan=plan, w=w)
            dest = PanelMatrix(B; plan=plan, w=w)
            reference = copy(B)
            reference[:, 5:20] = A[:, 1:16]

            copycols!(dest, 5:20, src, 1:16)
            @test Matrix(dest) == reference
            @test Matrix(src) == A
            @test disk_reads(src) > 0
            @test disk_reads(dest) > 0
            @test disk_writes(dest) > 0
        end
    end

    @testset "similar opens a store of its own" begin
        T = defaulteltype()
        mktempdir() do dir
            plan = testplan(scratch_dir=dir, panel_width=4)
            pm = PanelMatrix{T}(undef, 8, 12; plan=plan, w=4)
            @test length(readdir(dir)) == 1

            like = similar(pm)
            @test length(readdir(dir)) == 2
            @test like.store !== pm.store
            free!(like)
            @test length(readdir(dir)) == 1
            free!(pm)
        end
    end
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
