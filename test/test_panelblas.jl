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

@testset "the panel BLAS gives its buffers back" begin
    T = defaulteltype()
    N, k, w = 40, 23, 7
    plan = testplan()
    X = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    Y = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    dense = deviceoperator(randmatrix(T, N, N))

    gram(X, Y)
    norm(X)
    project(X, dense)
    cholqr2!(Y)
    before = device_bytes_allocated(plan)

    gram(X, Y)
    norm(X)
    project(X, dense)
    cholqr2!(Y)
    @test device_bytes_allocated(plan) == before
end
