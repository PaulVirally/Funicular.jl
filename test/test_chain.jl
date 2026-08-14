# A randomized factorization from end to end, written only in terms of what
# Funicular exports: a ghost test matrix through a rectangular operator, an
# orthonormalization on each side of the adjoint pass, a truncation, and the
# result collected and spliced into a wider matrix. Every stage is checked
# against the dense arithmetic, so a failure names the operation that drifted
# rather than the end of the chain.

@testset "a randomized factorization chain matches the dense reference $T" for T in testeltypes()
    # The operator is rectangular and the panels are cut narrow, so the chain
    # carries two different row counts and a ragged last panel throughout.
    m, n, s, r, w = 120, 96, 24, 9, 5
    Random.seed!(0x5EED)
    plan = testplan()
    reference = randmatrix(T, m, n)
    G = deviceoperator(reference)
    tol = blastol(T, max(m, n), s)

    Ω = GhostPanels(T, n, s; plan=plan, seed=0xB0A7, w=w)
    probe = Matrix(Ω)

    Y = PanelMatrix{T}(undef, m, s; plan=plan, w=w)
    panelmul!(Y, G, Ω)
    sketch = reference * probe
    @test Matrix(Y) ≈ sketch rtol=tol

    R = cholqr2!(Y)
    Q = Matrix(Y)
    @test opnorm(Q' * Q - I) <= orthogonality_bound(T, m, s)
    @test Q * R ≈ sketch rtol=tol

    # similar carries the panel width across the change of row count, which is
    # what lets the adjoint pass write into a matrix of n rows from one of m.
    Z = similar(Y, T, (n, s))
    panelmul!(Z, adjoint(G), Y)
    image = reference' * Q
    @test Matrix(Z) ≈ image rtol=tol

    S = cholqr2!(Z)
    P = Matrix(Z)
    @test opnorm(P' * P - I) <= orthogonality_bound(T, n, s)
    @test P * S ≈ image rtol=tol

    C = randmatrix(T, s, r)
    V = similar(Z, T, (n, r))
    rightmul!(V, Z, C)
    truncated = P * C
    @test Matrix(V) ≈ truncated rtol=tol

    collected = Matrix(V)
    roundtrip = PanelMatrix(collected; plan=plan, w=w)
    @test Matrix(roundtrip) == collected

    wide = PanelMatrix{T}(undef, n, 2r; plan=plan, w=w)
    copycols!(wide, 1:r, collected)
    copycols!(wide, (r + 1):(2r), roundtrip, 1:r)
    @test Matrix(wide) == hcat(collected, collected)

    @test Matrix(Ω) == probe
end
