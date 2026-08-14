@testset "a ghost matrix regenerates the same entries w=$w" for w in (8, 7, 1)
    T = defaulteltype()
    N, k = 24, 8
    plan = testplan()
    Ω = GhostPanels(T, N, k; plan=plan, seed=0x5EED, w=w)

    @test size(Ω) == (N, k)
    @test eltype(Ω) === T
    @test npanels(Ω) == cld(k, w)
    A = Matrix(Ω)
    @test A == Matrix(Ω)
    @test A == Matrix(GhostPanels(T, N, k; plan=testplan(), seed=0x5EED, w=w))
    @test A != Matrix(GhostPanels(T, N, k; plan=testplan(), seed=0x5EEE, w=w))
    # Standard normals, so nothing is degenerate and the columns are independent.
    @test 0.5 < norm(A) / sqrt(N * k) < 2
    @test rank(A) == min(N, k)
end

@testset "the same seed gives the same matrix at every panel width" begin
    T = defaulteltype()
    N, k = 40, 23
    reference = Matrix(GhostPanels(T, N, k; plan=testplan(), seed=0x5EED, w=23))
    for w in (23, 12, 7, 1)
        Ω = GhostPanels(T, N, k; plan=testplan(), seed=0x5EED, w=w)
        @test npanels(Ω) == cld(k, w)
        @test Matrix(Ω) == reference
    end
end

@testset "the generator sees each column's index in the full matrix" begin
    T = defaulteltype()
    N, k, w = 12, 7, 3
    plan = testplan()
    Ω = GhostPanels(T, N, k; plan=plan, w=w) do dst, rng, col
        dst .= T(col)
    end
    @test Matrix(Ω) == T.(repeat((1:k)', N))
end

@testset "a per column generator is free of the panel boundaries w=$w" for w in (23, 7, 1)
    T = defaulteltype()
    N, k = 40, 23
    plan = testplan()
    # Normalizing a column needs the whole column, and that is what the
    # generator is handed however the columns are cut into panels.
    Ω = GhostPanels(T, N, k; plan=plan, seed=0xC0FFEE, w=w) do dst, rng, col
        randn!(rng, dst)
        dst ./= norm(dst)
    end
    A = Matrix(Ω)
    @test all(j -> norm(A[:, j]) ≈ 1, 1:k)
    @test A == Matrix(GhostPanels(T, N, k; plan=testplan(), seed=0xC0FFEE, w=23) do dst, rng, col
        randn!(rng, dst)
        dst ./= norm(dst)
    end)
end

@testset "a Rademacher generator round trips" begin
    T = defaulteltype()
    N, k, w = 24, 9, 4
    plan = testplan()
    Ω = GhostPanels(T, N, k; plan=plan, seed=7, w=w) do dst, rng, col
        rand!(rng, dst, (-one(T), one(T)))
    end
    A = Matrix(Ω)
    @test all(x -> abs(x) == 1, A)
    @test A == Matrix(Ω)
end

@testset "a narrowed host tier hands the generator its own eltype" begin
    T = defaulteltype()
    S = narrowed(T)
    N, k, w = 12, 5, 2
    seen = Tuple{DataType,Int}[]
    plan = testplan(host_eltype=S)
    Ω = GhostPanels(T, N, k; plan=plan, w=w) do dst, rng, col
        push!(seen, (eltype(dst), length(dst)))
        randn!(rng, dst)
    end
    Matrix(Ω)
    @test length(seen) == k
    # The generator writes into the host tier's storage, so it sees S rather
    # than the compute eltype, and it sees a whole column however wide the panel
    # holding that column is.
    @test all(entry -> entry == (S, N), seen)
end

@testset "a ghost panel is regenerated after it leaves host memory" begin
    T = defaulteltype()
    N, k, w = 24, 8, 2
    plan = testplan(host_budget=panelbudget(2, N, w, T))
    Ω = GhostPanels(T, N, k; plan=plan, seed=3, w=w)
    A = Matrix(Ω)
    # Two blocks for four panels, so collecting it dropped panels it then had to
    # make again, and the second pass sees none of the first one's memory. A
    # ghost matrix needs no scratch_dir for this: there is nothing to write.
    @test count(isresident, Ω.panels) <= 2
    @test plan.scratch_dir === nothing
    @test Matrix(Ω) == A
end

@testset "a ghost matrix feeds the operator sweeps w=$w" for w in (4, 3)
    T = defaulteltype()
    N, k = 20, 8
    plan = testplan()
    Ω = GhostPanels(T, N, k; plan=plan, seed=0xF00D, w=w)
    A = Matrix(Ω)
    dense = randmatrix(T, N, N)
    G = deviceoperator(dense)

    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    panelmul!(Y, G, Ω)
    @test Matrix(Y) ≈ dense * A rtol=blastol(T, N, k)
    @test gram(Ω, Y) ≈ A' * (dense * A) rtol=blastol(T, N, k)
    @test norm(Ω) ≈ norm(A) rtol=blastol(T, N, k)

    seen = Matrix{T}(undef, N, k)
    foreachpanel(Ω; write=false) do j, panel
        copyto!(view(seen, :, panelrange(Ω, j)), host(panel))
    end
    @test seen == A
end

@testset "a ghost matrix refuses to be written to" begin
    T = defaulteltype()
    N, k, w = 16, 6, 4
    plan = testplan()
    Ω = GhostPanels(T, N, k; plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    G = deviceoperator(randmatrix(T, N, N))

    @test_throws ArgumentError cholqr2!(Ω)
    @test_throws ArgumentError scale!(Ω, 2)
    @test_throws ArgumentError copyto!(Ω, randmatrix(T, N, k))
    @test_throws ArgumentError panelmul!(Ω, G, Y)
    @test_throws ArgumentError axpy!(one(T), Y, Ω)
    @test_throws ArgumentError foreachpanel((j, panel) -> nothing, Ω)
    # The refusal comes before anything is staged, so no panel is left pinned.
    @test all(panel -> panel.pins == 0, Ω.panels)
end

@testset "a ghost matrix narrows its host storage" begin
    T = defaulteltype()
    S = narrowed(T)
    N, k, w = 20, 7, 3
    plan = testplan(host_eltype=S)
    Ω = GhostPanels(T, N, k; plan=plan, seed=11, w=w)
    prefetch!(Ω, 1)
    @test storage_eltype(Ω.panels[1]) === S
    @test Matrix(Ω) ≈ Matrix(GhostPanels(T, N, k; plan=testplan(), seed=11, w=w)) rtol=8 * eps(real(S))
end

@testset "ghost panels evict without writing anything" begin
    T = defaulteltype()
    N, k, w = 16, 6, 3
    plan = testplan()
    Ω = GhostPanels(T, N, k; plan=plan, w=w)
    A = Matrix(Ω)
    for panel in Ω.panels
        @test isresident(panel)
        evict!(panel, plan)
        @test !isresident(panel)
        @test !ondisk(panel)
    end
    @test Matrix(Ω) == A
end

@testset "a read only sweep leaves a ghost regenerable w=$w" for w in (23, 7, 1)
    T = defaulteltype()
    m, n, k = 40, 25, 23
    reference = randmatrix(T, m, n)
    plan = testplan()
    Ω = GhostPanels(T, n, k; plan=plan, seed=0x5EED, w=w)
    A = Matrix(Ω)
    Y = PanelMatrix{T}(undef, m, k; plan=plan, w=w)

    panelmul!(Y, deviceoperator(reference), Ω)
    @test Matrix(Y) ≈ reference * A rtol=blastol(T, m, k)
    @test occursin("ghost", sprint(show, Ω))
    @test !any(panel -> panel.dirty, Ω.panels)
    @test all(panel -> panel.pins == 0, Ω.panels)
    for panel in Ω.panels
        evict!(panel, plan)
    end
    @test Matrix(Ω) == A
end

# The chain a randomized method runs on every iteration. The budget leaves the
# ghost fewer blocks than it has panels, so panels of it are dropped and made
# again while the row sweeps inside cholqr2! hold every panel of Y at once.
function sketchchain(plan, ::Type{T}, reference, m, n, k, w) where {T}
    Ω = GhostPanels(T, n, k; plan=plan, seed=0x5EED, w=w)
    Y = PanelMatrix{T}(undef, m, k; plan=plan, w=w)
    panelmul!(Y, deviceoperator(reference), Ω)
    R = cholqr2!(Y)
    Matrix(Ω), Matrix(Y), R
end

@testset "the sketch chain matches the dense reference under a tight budget" begin
    T = defaulteltype()
    m, n, k, w = 40, 24, 24, 3
    reference = randmatrix(T, m, n)
    tol = blastol(T, m, k)
    budget = panelbudget(cld(k, w), m, w, T) + panelbudget(6, n, w, T)

    A, Q, R = sketchchain(testplan(host_budget=budget), T, reference, m, n, k, w)
    @test opnorm(Q' * Q - I) <= orthogonality_bound(T, m, k)
    @test Q * R ≈ reference * A rtol=tol

    if HAS_HDF5
        mktempdir() do dir
            spilled, Qd, Rd = sketchchain(testplan(scratch_dir=dir, host_budget=budget), T, reference, m, n, k, w)
            @test spilled == A
            @test opnorm(Qd' * Qd - I) <= orthogonality_bound(T, m, k)
            @test Qd * Rd ≈ reference * spilled rtol=tol
        end
    end
end

@testset "ghost construction is validated" begin
    T = defaulteltype()
    plan = testplan()
    @test_throws ArgumentError GhostPanels(T, 0, 4; plan=plan)
    @test_throws ArgumentError GhostPanels(T, 4, 0; plan=plan)
    @test_throws ArgumentError GhostPanels(Complex, 4, 4; plan=plan)
    @test_throws ArgumentError GhostPanels(T, 4, 4; plan=plan, w=5)
    @test occursin("ghost", sprint(show, GhostPanels(T, 8, 4; plan=plan, w=2)))
end
