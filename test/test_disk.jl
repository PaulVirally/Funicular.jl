# A sweep whose producer task cannot get host memory fails on that task, so the
# caller sees the channel's wrapper around the error rather than the error
# itself.
const STAGING_FAILURE = Union{TaskFailedException,ArgumentError}

@testset "three tier round trip T=$T w=$w" for T in testeltypes(), w in (8, 7, 1)
    N, k = 12, 32
    A = randmatrix(T, N, k)
    mktempdir() do dir
        plan = testplan(scratch_dir=dir, panel_width=w,
                        host_budget=panelbudget(2, N, w, T))
        pm = PanelMatrix(A; plan=plan, w=w)

        @test npanels(pm) > 2
        @test count(ondisk, pm.panels) == npanels(pm) - 2
        @test Matrix(pm) == A
        @test disk_reads(pm) > 0
        @test disk_writes(pm) > 0
        @test length(readdir(dir)) == 1

        free!(pm)
        @test isempty(readdir(dir))
    end
end

if storable(narrowed(defaulteltype()))
    @testset "narrowed storage through the disk tier" begin
        T = defaulteltype()
        S = narrowed(T)
        N, k, w = 9, 20, 4
        A = randmatrix(T, N, k)
        mktempdir() do dir
            plan = testplan(scratch_dir=dir, panel_width=w, host_eltype=S,
                            host_budget=panelbudget(2, N, w, S))
            pm = PanelMatrix(A; plan=plan, w=w)
            @test pm.store.stored === S
            @test pm.store.computed === T
            @test Matrix(pm) == T.(S.(A))
            @test disk_reads(pm) > 0
        end
    end
end

@testset "sweeps stream panels through the disk tier" begin
    T = defaulteltype()
    N, k, w = 10, 24, 4
    A = randmatrix(T, N, k)
    mktempdir() do dir
        plan = testplan(scratch_dir=dir, panel_width=w,
                        host_budget=panelbudget(5, N, w, T))
        pm = PanelMatrix(A; plan=plan, w=w)
        before = disk_reads(pm)
        foreachpanel(pm) do _, block
            block .*= 2
        end
        @test disk_reads(pm) > before
        @test Matrix(pm) == 2 .* A
    end
end

@testset "panelmul! streams a disk backed pair" begin
    T = defaulteltype()
    N, k, w = 10, 24, 4
    A = randmatrix(T, N, k)
    G = randmatrix(T, N, N)
    mktempdir() do dir
        # Eight panels of room for the twelve the two matrices hold between
        # them. A panel sweep holds one panel of each matrix for as many steps
        # as the producer runs ahead, and the rest cycle through disk.
        plan = testplan(scratch_dir=dir, panel_width=w,
                        host_budget=panelbudget(8, N, w, T))
        X = PanelMatrix(A; plan=plan, w=w)
        Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)

        panelmul!(Y, deviceoperator(G), X)
        @test Matrix(Y) ≈ G * A rtol=blastol(T, N, k)
        @test disk_reads(X) > 0
        @test disk_reads(Y) > 0
        @test disk_writes(Y) > 0
    end
end

@testset "row sweeps over a disk backed matrix" begin
    T = defaulteltype()
    N, k, w = 40, 24, 4
    A = randmatrix(T, N, k)
    B = randmatrix(T, N, k)
    mktempdir() do dir
        # A row block is a slice of every panel, so gram and cholqr2! hold the
        # whole matrix at once. Seven panels covers one of these two and not
        # both, so the one that is not being swept cycles out to disk.
        plan = testplan(scratch_dir=dir, panel_width=w,
                        host_budget=panelbudget(7, N, w, T))
        X = PanelMatrix(A; plan=plan, w=w)
        Y = PanelMatrix(B; plan=plan, w=w)

        R = cholqr2!(Y)
        @test gram(X) ≈ A' * A rtol=blastol(T, N, k)
        @test disk_reads(X) > 0
        @test disk_writes(Y) > 0

        Q = Matrix(Y)
        @test disk_reads(Y) > 0
        @test opnorm(Q' * Q - I) <= orthogonality_bound(T, N, k)
        @test Q * R ≈ B rtol=blastol(T, N, k)
        @test Matrix(X) == A
    end
end

@testset "disk staging under jitter" begin
    T = defaulteltype()
    N, k, w = 12, 20, 4
    A = randmatrix(T, N, k)
    for _ in 1:min(repetitions(), 10)
        mktempdir() do dir
            plan = testplan(backend=jittery_backend(), scratch_dir=dir,
                            panel_width=w, host_budget=panelbudget(4, N, w, T))
            pm = PanelMatrix(A; plan=plan, w=w)
            foreachpanel(pm) do _, block
                block .*= 2
            end
            @test Matrix(pm) == 2 .* A
            @test all(panel -> panel.pins == 0, pm.panels)
            free!(pm)
        end
    end
end

@testset "eviction and prefetch" begin
    T = defaulteltype()
    N, k, w = 8, 12, 4
    A = randmatrix(T, N, k)
    mktempdir() do dir
        plan = testplan(scratch_dir=dir, panel_width=w,
                        host_budget=panelbudget(3, N, w, T))
        pm = PanelMatrix(A; plan=plan, w=w)
        @test !any(ondisk, pm.panels)

        evict!(pm.panels[2], plan)
        @test ondisk(pm.panels[2])
        @test pm.panels[2].tier isa DiskTier
        @test evict!(pm.panels[2], plan) === nothing
        @test_throws ArgumentError materialize!(zerobuffer(T, N, w), pm.panels[2], plan)

        prefetch!(pm, 2)
        @test !ondisk(pm.panels[2])
        buffer = zerobuffer(T, N, w)
        materialize!(buffer, pm.panels[2], plan)
        @test host(buffer) == A[:, panelrange(pm, 2)]
        @test Matrix(pm) == A
    end
end

@testset "save and load" begin
    T = defaulteltype()
    N, k, w = 11, 26, 5
    A = randmatrix(T, N, k)
    mktempdir() do dir
        path = joinpath(dir, "basis.h5")
        pm = PanelMatrix(A; plan=testplan(), w=w)
        @test save(pm, path) == path
        @test Matrix(pm) == A

        reloaded = load(PanelMatrix, path; plan=testplan())
        @test eltype(reloaded) === T
        @test size(reloaded) == (N, k)
        @test panelwidth(reloaded) == w
        @test all(ondisk, reloaded.panels)
        @test Matrix(reloaded) == A
        @test gram(reloaded) ≈ A' * A rtol=blastol(T, N, k)
        free!(reloaded)
        @test isfile(path)

        typed = load(PanelMatrix{T}, path; plan=testplan())
        @test eltype(typed) === T
        free!(typed)
    end
end

@testset "a read only matrix refuses to spill a change" begin
    T = defaulteltype()
    N, k, w = 8, 16, 4
    A = randmatrix(T, N, k)
    mktempdir() do dir
        path = joinpath(dir, "frozen.h5")
        save(PanelMatrix(A; plan=testplan(), w=w), path)

        pm = load(PanelMatrix, path; plan=testplan(panel_width=w), readonly=true)
        @test Matrix(pm) == A
        prefetch!(pm, 1)
        pm.panels[1].dirty = true
        @test_throws ArgumentError evict!(pm.panels[1], pm.plan)
        free!(pm)

        cramped = testplan(panel_width=w, host_budget=panelbudget(2, N, w, T))
        doubling = load(PanelMatrix, path; plan=cramped, readonly=true)
        @test_throws STAGING_FAILURE foreachpanel((_, block) -> block .*= 2, doubling)
        free!(doubling)

        again = load(PanelMatrix, path; plan=testplan())
        @test Matrix(again) == A
        free!(again)
    end
end

@testset "disk tier failure paths" begin
    T = defaulteltype()
    mktempdir() do dir
        @test_throws ArgumentError load(PanelMatrix, joinpath(dir, "absent.h5"); plan=testplan())

        S = narrowed(T)
        mismatched = testplan(scratch_dir=dir, disk_eltype=S)
        @test_throws ArgumentError PanelMatrix{T}(undef, 8, 8; plan=mismatched, w=4)

        # Half precision has no HDF5 datatype, so the cold tier refuses it
        # rather than letting the extension fail on the way out.
        halved = testplan(scratch_dir=dir, host_eltype=Float16)
        @test_throws ArgumentError PanelMatrix{Float32}(undef, 8, 8; plan=halved, w=4)

        path = joinpath(dir, "wide.h5")
        save(PanelMatrix(randmatrix(T, 8, 8); plan=testplan(), w=4), path)
        @test_throws ArgumentError load(PanelMatrix, path; plan=testplan(host_eltype=S))

        # Every panel of a row sweep has to be resident at once, and a budget
        # too small for that says so rather than thrashing.
        C = randmatrix(T, 8, 24)
        cramped = testplan(scratch_dir=dir, panel_width=4,
                           host_budget=panelbudget(2, 8, 4, T))
        tight = PanelMatrix(C; plan=cramped, w=4)
        @test_throws STAGING_FAILURE gram(tight)
        # A sweep that gave up part way through unpins what it staged and
        # leaves the matrix as it was.
        @test all(panel -> panel.pins == 0, tight.panels)
        @test Matrix(tight) == C
    end
end

@testset "concurrent readers of one store" begin
    T = defaulteltype()
    N, k, w = 9, 24, 3
    A = randmatrix(T, N, k)
    mktempdir() do dir
        path = joinpath(dir, "shared.h5")
        save(PanelMatrix(A; plan=testplan(), w=w), path)

        readers = map(1:4) do _
            Threads.@spawn begin
                plan = testplan(panel_width=w, host_budget=panelbudget(3, N, w, T))
                pm = load(PanelMatrix, path; plan=plan, readonly=true)
                try
                    all(Matrix(pm) == A for _ in 1:5)
                finally
                    free!(pm)
                end
            end
        end
        @test all(fetch.(readers))
    end
end

@testset "mini RSVD with the data larger than the host tier" begin
    T = defaulteltype()
    N, k, w, rank = 120, 32, 4, 6
    Random.seed!(0xD15C)
    A = lowrank_plus_noise(T, N, rank, noisefloor(T))
    G = deviceoperator(A)
    reference = svdvals(A)

    Random.seed!(0xD15CE)
    _, resident = minirsvd(G, T, N, k, testplan(), w)

    mktempdir() do dir
        # Ten panels of room for the sixteen the two matrices hold between them,
        # so panels cycle through disk while a row sweep still has all eight of
        # one matrix at once.
        plan = testplan(scratch_dir=dir, panel_width=w,
                        host_budget=panelbudget(10, N, w, T))
        Random.seed!(0xD15CE)
        Q, S = minirsvd(G, T, N, k, plan, w)

        @test disk_reads(Q) > 0
        @test disk_writes(Q) > 0
        @test opnorm(Matrix(Q)' * Matrix(Q) - I) <= orthogonality_bound(T, N, k)
        @test svdvals(S)[1:rank] ≈ reference[1:rank] rtol=spectrum_tol(T)
        @test svdvals(S) ≈ svdvals(resident) rtol=blastol(T, N, k)
    end
end
