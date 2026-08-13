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

@testset "the generator sees the panel's own columns" begin
    T = defaulteltype()
    N, k, w = 12, 7, 3
    plan = testplan()
    Ω = GhostPanels(T, N, k; plan=plan, w=w) do dst, rng, cols
        for (c, col) in enumerate(cols)
            dst[:, c] .= T(col)
        end
    end
    @test Matrix(Ω) == T.(repeat((1:k)', N))
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

@testset "ghost construction is validated" begin
    T = defaulteltype()
    plan = testplan()
    @test_throws ArgumentError GhostPanels(T, 0, 4; plan=plan)
    @test_throws ArgumentError GhostPanels(T, 4, 0; plan=plan)
    @test_throws ArgumentError GhostPanels(Complex, 4, 4; plan=plan)
    @test_throws ArgumentError GhostPanels(T, 4, 4; plan=plan, w=5)
    @test occursin("ghost", sprint(show, GhostPanels(T, 8, 4; plan=plan, w=2)))
end
