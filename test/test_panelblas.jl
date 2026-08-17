const BLAS_WIDTHS = (23, 12, 7, 1)

@testset "gram against dense $T w=$w nbuffers=$nb" for T in testeltypes(), w in BLAS_WIDTHS, nb in (1, 2)
    N, k = 40, 23
    A, B = randmatrix(T, N, k), randmatrix(T, N, k)
    plan = testplan()
    X = PanelMatrix(A; plan=plan, w=w)
    Y = PanelMatrix(B; plan=plan, w=w)
    tol = blastol(T, N, k)

    @test gram(X, Y; nbuffers=nb) ≈ A' * B rtol=tol
    @test gram(X; nbuffers=nb) ≈ A' * A rtol=tol
    @test gram(X, X; nbuffers=nb) == gram(X; nbuffers=nb)
    @test Matrix(X) == A
end

@testset "gram is the same to the bit at nbuffers 1 and 2 w=$w" for w in BLAS_WIDTHS
    T = defaulteltype()
    N, k = 40, 23
    A, B = randmatrix(T, N, k), randmatrix(T, N, k)
    plan = testplan()
    X = PanelMatrix(A; plan=plan, w=w)
    Y = PanelMatrix(B; plan=plan, w=w)
    @test gram(X, Y; nbuffers=1) == gram(X, Y; nbuffers=2)
    @test gram(X; nbuffers=1) == gram(X; nbuffers=2)
end

@testset "gram with narrowed host storage" begin
    T = defaulteltype()
    S = narrowed(T)
    N, k, w = 40, 23, 7
    A = randmatrix(T, N, k)
    plan = testplan(host_eltype=S)
    X = PanelMatrix(A; plan=plan, w=w)
    stored = T.(S.(A))
    @test gram(X) ≈ stored' * stored rtol=blastol(T, N, k)
end

@testset "gram rejects mismatched operands" begin
    T = defaulteltype()
    plan = testplan()
    X = PanelMatrix{T}(undef, 10, 20; plan=plan, w=7)
    @test_throws ArgumentError gram(X, PanelMatrix{T}(undef, 10, 20; plan=plan, w=5))
    @test_throws ArgumentError gram(X, PanelMatrix{narrowed(T)}(undef, 10, 20; plan=plan, w=7))
    @test_throws ArgumentError gram(X, PanelMatrix{T}(undef, 10, 20; plan=testplan(), w=7))
end

@testset "panelmul! against dense $T w=$w nbuffers=$nb" for T in testeltypes(), w in BLAS_WIDTHS, nb in (1, 2)
    N, k = 40, 23
    A = randmatrix(T, N, k)
    reference = randmatrix(T, N, N)
    dense = deviceoperator(reference)
    plan = testplan()
    X = PanelMatrix(A; plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    tol = blastol(T, N, k)

    @test Matrix(panelmul!(Y, dense, X; nbuffers=nb)) ≈ reference * A rtol=tol
    @test Matrix(panelmul!(Y, ColumnOperator(dense), X; nbuffers=nb)) ≈ reference * A rtol=tol
end

@testset "panelmul! with a matrix free operator w=$w" for w in BLAS_WIDTHS
    T = defaulteltype()
    N, k = 40, 23
    A = randmatrix(T, N, k)
    G = CircularConvolution(randmatrix(T, N, 1)[:, 1])
    plan = testplan()
    X = PanelMatrix(A; plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)

    panelmul!(Y, G, X)
    @test Matrix(Y) ≈ densematrix(G) * A rtol=blastol(T, N, k)
end

@testset "panelmul! rejects an operator that does not fit" begin
    T = defaulteltype()
    plan = testplan()
    X = PanelMatrix{T}(undef, 10, 20; plan=plan, w=7)
    Y = PanelMatrix{T}(undef, 10, 20; plan=plan, w=7)
    @test_throws ArgumentError panelmul!(Y, deviceoperator(randmatrix(T, 10, 9)), X)
    @test_throws ArgumentError panelmul!(Y, deviceoperator(randmatrix(T, 11, 11)), X)
    @test_throws ArgumentError panelmul!(X, deviceoperator(randmatrix(T, 10, 10)), X)
end

@testset "panelmul! with a rectangular operator $T $(m)×$(n) w=$w nbuffers=$nb" for T in testeltypes(), (m, n) in ((40, 25), (25, 40)), w in (23, 7, 1), nb in (1, 2)
    k = 23
    A = randmatrix(T, n, k)
    B = randmatrix(T, m, k)
    reference = randmatrix(T, m, n)
    dense = deviceoperator(reference)
    plan = testplan()
    X = PanelMatrix(A; plan=plan, w=w)
    Y = PanelMatrix{T}(undef, m, k; plan=plan, w=w)
    tol = blastol(T, max(m, n), k)

    @test Matrix(panelmul!(Y, dense, X; nbuffers=nb)) ≈ reference * A rtol=tol
    @test Matrix(panelmul!(Y, ColumnOperator(dense), X; nbuffers=nb)) ≈ reference * A rtol=tol

    # The adjoint maps the other way, so the two matrices swap roles.
    W = PanelMatrix(B; plan=plan, w=w)
    Z = PanelMatrix{T}(undef, n, k; plan=plan, w=w)
    @test Matrix(panelmul!(Z, adjoint(dense), W; nbuffers=nb)) ≈ reference' * B rtol=tol
end

@testset "panelmul! applies a rectangular operator to a ghost w=$w" for w in BLAS_WIDTHS
    T = defaulteltype()
    m, n, k = 40, 25, 23
    reference = randmatrix(T, m, n)
    plan = testplan()
    Ω = GhostPanels(T, n, k; plan=plan, seed=0xdeadbeef, w=w)
    Y = PanelMatrix{T}(undef, m, k; plan=plan, w=w)

    panelmul!(Y, deviceoperator(reference), Ω)
    @test Matrix(Y) ≈ reference * Matrix(Ω) rtol=blastol(T, m, k)
end

@testset "panelmul! rejects a rectangular operator that does not fit" begin
    T = defaulteltype()
    plan = testplan()
    X = PanelMatrix{T}(undef, 12, 20; plan=plan, w=7)
    Y = PanelMatrix{T}(undef, 15, 20; plan=plan, w=7)
    G = deviceoperator(randmatrix(T, 15, 12))

    @test_throws ArgumentError panelmul!(Y, deviceoperator(randmatrix(T, 12, 15)), X)
    @test_throws ArgumentError panelmul!(Y, G, PanelMatrix{T}(undef, 12, 20; plan=plan, w=5))
    @test_throws ArgumentError panelmul!(Y, G, PanelMatrix{T}(undef, 12, 18; plan=plan, w=7))
    @test panelmul!(Y, G, X) === Y
end

@testset "rightmul! out of place against dense $T r=$r w=$w nbuffers=$nb" for T in testeltypes(), r in (12, 23, 31), w in BLAS_WIDTHS, nb in (1, 2)
    N, s = 40, 23
    A = randmatrix(T, N, s)
    C = randmatrix(T, s, r)
    plan = testplan()
    src = PanelMatrix(A; plan=plan, w=w)
    dest = PanelMatrix{T}(undef, N, r; plan=plan, w=clamp(w - 3, 1, r))
    tol = blastol(T, N, max(s, r))

    @test rightmul!(dest, src, C; nbuffers=nb) === dest
    @test Matrix(dest) ≈ A * C rtol=tol
    @test Matrix(src) == A
end

@testset "rightmul! in place against dense $T w=$w nbuffers=$nb" for T in testeltypes(), w in BLAS_WIDTHS, nb in (1, 2)
    N, k = 40, 23
    A = randmatrix(T, N, k)
    C = randmatrix(T, k, k)
    plan = testplan()
    Y = PanelMatrix(A; plan=plan, w=w)
    tol = blastol(T, N, k)

    @test rightmul!(Y, C; nbuffers=nb) === Y
    @test Matrix(Y) ≈ A * C rtol=tol

    dest = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    rightmul!(dest, PanelMatrix(A; plan=plan, w=w), C; nbuffers=nb)
    @test Matrix(dest) ≈ Matrix(Y) rtol=tol
end

@testset "rightmul! is the same to the bit at nbuffers 1 and 2 w=$w" for w in BLAS_WIDTHS
    T = defaulteltype()
    N, s, r = 40, 23, 12
    A = randmatrix(T, N, s)
    C = randmatrix(T, s, r)
    D = randmatrix(T, s, s)

    produced = map((1, 2)) do nb
        plan = testplan()
        src = PanelMatrix(A; plan=plan, w=w)
        dest = PanelMatrix{T}(undef, N, r; plan=plan, w=clamp(w - 3, 1, r))
        Matrix(rightmul!(dest, src, C; nbuffers=nb))
    end
    @test produced[1] == produced[2]

    inplace = map((1, 2)) do nb
        plan = testplan()
        Matrix(rightmul!(PanelMatrix(A; plan=plan, w=w), D; nbuffers=nb))
    end
    @test inplace[1] == inplace[2]
end

@testset "rightmul! takes a wrapped factor w=$w" for w in BLAS_WIDTHS
    T = defaulteltype()
    N, k = 40, 23
    A = randmatrix(T, N, k)
    R = UpperTriangular(randmatrix(T, k, k) + 4I)
    D = adjoint(randmatrix(T, k, k))
    plan = testplan()
    tol = blastol(T, N, k)

    @test Matrix(rightmul!(PanelMatrix(A; plan=plan, w=w), R)) ≈ A * R rtol=tol
    @test Matrix(rightmul!(PanelMatrix(A; plan=plan, w=w), inv(R))) ≈ A * Matrix(inv(R)) rtol=tol
    @test Matrix(rightmul!(PanelMatrix(A; plan=plan, w=w), D)) ≈ A * D rtol=tol
end

@testset "rightmul! with narrowed host storage" begin
    T = defaulteltype()
    S = narrowed(T)
    N, s, r, w = 40, 23, 12, 7
    A = randmatrix(T, N, s)
    C = randmatrix(T, s, r)
    plan = testplan(host_eltype=S)
    src = PanelMatrix(A; plan=plan, w=w)
    dest = PanelMatrix{T}(undef, N, r; plan=plan, w=5)
    stored = T.(S.(A))

    rightmul!(dest, src, C)
    # Both tiers hold S, so the product is narrowed on the way back down as well
    # as on the way up, and the tolerance has to be the storage eltype's.
    @test Matrix(dest) ≈ stored * C rtol=blastol(S, N, s)
end

@testset "rightmul! reads a ghost source w=$w" for w in BLAS_WIDTHS
    T = defaulteltype()
    N, s, r = 40, 23, 12
    C = randmatrix(T, s, r)
    plan = testplan()
    Ω = GhostPanels(T, N, s; plan=plan, seed=0xdeadbeef, w=w)
    dest = PanelMatrix{T}(undef, N, r; plan=plan, w=clamp(w - 3, 1, r))
    reference = Matrix(Ω)

    @test Matrix(rightmul!(dest, Ω, C)) ≈ reference * C rtol=blastol(T, N, s)
    @test Matrix(Ω) == reference
end

@testset "rightmul! rejects factors and matrices that do not fit" begin
    T = defaulteltype()
    N, s, r, w = 20, 12, 8, 5
    plan = testplan()
    Y = PanelMatrix(randmatrix(T, N, s); plan=plan, w=w)
    dest = PanelMatrix{T}(undef, N, r; plan=plan, w=4)
    Ω = GhostPanels(T, N, s; plan=plan, w=w)

    @test_throws ArgumentError rightmul!(Y, randmatrix(T, s, r))
    @test_throws ArgumentError rightmul!(Y, randmatrix(T, r, s))
    @test_throws ArgumentError rightmul!(dest, Y, randmatrix(T, s, s))
    @test_throws ArgumentError rightmul!(Y, Y, randmatrix(T, s, s))
    @test_throws ArgumentError rightmul!(Ω, randmatrix(T, s, s))
    @test_throws ArgumentError rightmul!(Ω, Y, randmatrix(T, s, s))
    @test_throws ArgumentError rightmul!(dest, PanelMatrix{T}(undef, N + 1, s; plan=plan, w=w), randmatrix(T, s, r))
    @test_throws ArgumentError rightmul!(dest, PanelMatrix{T}(undef, N, s; plan=testplan(), w=w), randmatrix(T, s, r))
end

@testset "axpy!, scale!, dot and norm $T w=$w nbuffers=$nb" for T in testeltypes(), w in BLAS_WIDTHS, nb in (1, 2)
    N, k = 40, 23
    A, B = randmatrix(T, N, k), randmatrix(T, N, k)
    α = T <: Complex ? T(0.5, -1.25) : T(0.75)
    plan = testplan()
    X = PanelMatrix(A; plan=plan, w=w)
    Y = PanelMatrix(B; plan=plan, w=w)
    tol = blastol(T, N, k)

    @test dot(X, Y; nbuffers=nb) ≈ dot(A, B) rtol=tol
    @test dot(X, X; nbuffers=nb) ≈ dot(A, A) rtol=tol
    @test norm(X; nbuffers=nb) ≈ norm(A) rtol=tol
    @test scale!(Y, α; nbuffers=nb) === Y
    @test Matrix(Y) ≈ α .* B rtol=tol
    @test axpy!(α, X, Y; nbuffers=nb) === Y
    @test Matrix(Y) ≈ α .* A .+ α .* B rtol=tol
end

@testset "reductions are the same to the bit at nbuffers 1 and 2 w=$w" for w in BLAS_WIDTHS
    T = defaulteltype()
    N, k = 40, 23
    A, B = randmatrix(T, N, k), randmatrix(T, N, k)
    plan = testplan()
    X = PanelMatrix(A; plan=plan, w=w)
    Y = PanelMatrix(B; plan=plan, w=w)
    @test dot(X, Y; nbuffers=1) === dot(X, Y; nbuffers=2)
    @test norm(X; nbuffers=1) === norm(X; nbuffers=2)
end

@testset "the elementwise operations reject mismatched operands" begin
    T = defaulteltype()
    plan = testplan()
    X = PanelMatrix{T}(undef, 10, 20; plan=plan, w=7)
    Y = PanelMatrix{T}(undef, 10, 20; plan=plan, w=5)
    @test_throws ArgumentError axpy!(T(2), X, Y)
    @test_throws ArgumentError axpy!(T(2), X, X)
    @test_throws ArgumentError dot(X, Y)
    @test_throws ArgumentError norm(X, 1)
end

conditioned(::Type{T}, N, k, κ) where {T} = Matrix(qr(randmatrix(T, N, k)).Q) * Diagonal(real(T).(exp10.(range(0, -log10(κ); length=k)))) * Matrix(qr(randmatrix(T, k, k)).Q)'

# Past κ ≈ eps^(-1/2) the Gram matrix loses positivity and the shifted fallback
# takes over, so the last case of each list is the one that has to survive it.
conditioning(::Type{Float64}) = (1e2, 1e6, 1e10)
conditioning(::Type{Float32}) = (1e1, 1e2, 1e5)

shifted_bound(::Type{Float64}) = 1e-8
shifted_bound(::Type{Float32}) = 1e-3

reconstruction_bound(::Type{Float64}) = 1e-10
reconstruction_bound(::Type{Float32}) = 1e-4

@testset "cholqr2! orthonormalizes w=$w nbuffers=$nb" for w in BLAS_WIDTHS, nb in (1, 2)
    T = defaulteltype()
    N, k = 40, 23
    A = randmatrix(T, N, k)
    plan = testplan()
    Y = PanelMatrix(A; plan=plan, w=w)

    R = cholqr2!(Y; nbuffers=nb)
    Q = Matrix(Y)
    @test size(R) == (k, k)
    @test R ≈ triu(R)
    @test opnorm(Q' * Q - I) <= orthogonality_bound(T, N, k)
    @test Q * R ≈ A rtol=blastol(T, N, k)
end

@testset "cholqr2! survives κ = $κ" for κ in conditioning(real(defaulteltype()))
    T = defaulteltype()
    R = real(T)
    N, k, w = 40, 23, 7
    A = conditioned(T, N, k, κ)
    plan = testplan()
    Y = PanelMatrix(A; plan=plan, w=w)

    factor = cholqr2!(Y)
    Q = Matrix(Y)
    shifted = κ > 1 / sqrt(eps(R))
    @test opnorm(Q' * Q - I) <= (shifted ? shifted_bound(R) : orthogonality_bound(T, N, k))
    @test norm(Q * factor - A) <= reconstruction_bound(R) * norm(A)
end

@testset "the shifted fallback fires only when it has to" begin
    T = defaulteltype()
    N, k = 40, 23
    healthy = randmatrix(T, N, k)
    G = healthy' * healthy
    _, shifted = Funicular.cholesky_upper(G, N, k)
    @test !shifted

    # A dependent column leaves the Gram positive semidefinite, and whether
    # LAPACK calls that a failure comes down to rounding. A zero column puts an
    # exact zero on the diagonal, which LAPACK always refuses.
    singular = copy(G)
    singular[:, end] .= 0
    singular[end, :] .= 0
    factor, shifted = Funicular.cholesky_upper(singular, N, k)
    @test shifted
    @test factor ≈ triu(factor)
    @test_throws ArgumentError Funicular.cholesky_upper(-G, N, k)
end

@testset "project against dense w=$w nbuffers=$nb" for w in BLAS_WIDTHS, nb in (1, 2)
    T = defaulteltype()
    N, k = 40, 23
    reference = randmatrix(T, N, N)
    dense = deviceoperator(reference)
    A = randmatrix(T, N, k)
    plan = testplan()
    Q = PanelMatrix(A; plan=plan, w=w)
    tol = blastol(T, N, k)

    @test project(Q, dense; nbuffers=nb) ≈ A' * reference * A rtol=tol
    @test project(Q, ColumnOperator(dense); nbuffers=nb) ≈ A' * reference * A rtol=tol
    @test Matrix(Q) == A
end

@testset "project matches panelmul! followed by gram" begin
    T = defaulteltype()
    N, k, w = 40, 23, 7
    dense = deviceoperator(randmatrix(T, N, N))
    A = randmatrix(T, N, k)
    plan = testplan()
    Q = PanelMatrix(A; plan=plan, w=w)
    Z = PanelMatrix{T}(undef, N, k; plan=plan, w=w)

    panelmul!(Z, dense, Q)
    @test project(Q, dense) ≈ gram(Q, Z) rtol=blastol(T, N, k)
end

@testset "project follows the operator's adjoint" begin
    T = defaulteltype()
    N, k, w = 40, 23, 7
    reference = randmatrix(T, N, N)
    dense = deviceoperator(reference)
    adjoint_dense = deviceoperator(Matrix(reference'))
    hermitian = deviceoperator(Matrix(Hermitian(reference)))
    A = randmatrix(T, N, k)
    plan = testplan()
    Q = PanelMatrix(A; plan=plan, w=w)
    tol = blastol(T, N, k)

    @test project(Q, adjoint_dense) ≈ project(Q, dense)' rtol=tol
    S = project(Q, hermitian)
    @test S ≈ S' rtol=tol
end

# The arithmetic above runs on a quiet backend. A row block sweep issues one
# copy per panel per step, and the CPU backend's jitter sleeps once per queued
# operation, about 2 ms each time. Jittering the parameterized sets above would
# put them into the minutes, so the schedule is stressed here instead, and in
# the executor's own row sweep tests.
@testset "the panel BLAS holds up under jitter nbuffers=$nb" for nb in (1, 2)
    T = defaulteltype()
    N, k, w = 30, 23, 7
    A = randmatrix(T, N, k)
    reference = randmatrix(T, N, N)
    tol = blastol(T, N, k)
    for _ in 1:10
        plan = testplan(backend=jittery_backend())
        X = PanelMatrix(A; plan=plan, w=w)
        dense = todevice(plan.backend, reference)
        @test gram(X; nbuffers=nb) ≈ A' * A rtol=tol
        @test project(X, dense; nbuffers=nb) ≈ A' * reference * A rtol=tol

        R = cholqr2!(X; nbuffers=nb)
        Q = Matrix(X)
        @test opnorm(Q' * Q - I) <= orthogonality_bound(T, N, k)
        @test Q * R ≈ A rtol=tol
    end
end

@testset "rightmul! holds up under jitter nbuffers=$nb" for nb in (1, 2)
    T = defaulteltype()
    N, s, r, w = 30, 23, 12, 7
    A = randmatrix(T, N, s)
    C = randmatrix(T, s, r)
    D = randmatrix(T, s, s)
    tol = blastol(T, N, s)
    for _ in 1:repetitions()
        plan = testplan(backend=jittery_backend())
        src = PanelMatrix(A; plan=plan, w=w)
        dest = PanelMatrix{T}(undef, N, r; plan=plan, w=5)
        @test Matrix(rightmul!(dest, src, C; nbuffers=nb)) ≈ A * C rtol=tol
        @test Matrix(rightmul!(src, D; nbuffers=nb)) ≈ A * D rtol=tol
    end
end

@testset "the panel BLAS gives its buffers back" begin
    T = defaulteltype()
    N, k, w = 40, 23, 7
    plan = testplan()
    X = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    Y = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    Z = PanelMatrix{T}(undef, N, 12; plan=plan, w=5)
    dense = deviceoperator(randmatrix(T, N, N))
    C = randmatrix(T, k, 12)
    D = randmatrix(T, k, k)

    gram(X, Y)
    norm(X)
    project(X, dense)
    rightmul!(Z, X, C)
    rightmul!(Y, D)
    cholqr2!(Y)
    before = device_bytes_allocated(plan)

    gram(X, Y)
    norm(X)
    project(X, dense)
    rightmul!(Z, X, C)
    rightmul!(Y, D)
    cholqr2!(Y)
    @test device_bytes_allocated(plan) == before
end
