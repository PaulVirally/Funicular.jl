# The panel BLAS in its intended use: a randomized subspace iteration written
# only in terms of what Funicular exports, against a dense reference.

function minirsvd(G, ::Type{T}, N, k, plan, w; iterations=2, nbuffers=nothing) where {T}
    Q = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    scratch = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    cholqr2!(Q; nbuffers=nbuffers)
    for _ in 1:iterations
        panelmul!(scratch, G, Q; nbuffers=nbuffers)
        cholqr2!(scratch; nbuffers=nbuffers)
        Q, scratch = scratch, Q
    end
    S = project(Q, G; nbuffers=nbuffers)
    Q, S
end

# The noise floor sits just above what the arithmetic can resolve, so the
# retained singular values stand clear of it in either precision.
noisefloor(::Type{T}) where {T} = sqrt(eps(real(T)))
spectrum_tol(::Type{T}) where {T} = real(T) === Float64 ? 1e-6 : 1e-4

@testset "mini RSVD matches the dense reference w=$w" for w in (24, 12, 17, 1)
    T = defaulteltype()
    N, k, rank = 120, 24, 6
    Random.seed!(0xdeadbeef + w)
    A = lowrank_plus_noise(T, N, rank, noisefloor(T))
    reference = svdvals(A)
    plan = testplan()

    Q, S = minirsvd(deviceoperator(A), T, N, k, plan, w)
    @test opnorm(Matrix(Q)' * Matrix(Q) - I) <= orthogonality_bound(T, N, k)
    @test svdvals(S)[1:rank] ≈ reference[1:rank] rtol=spectrum_tol(T)
end

@testset "mini RSVD is unchanged by the buffer count" begin
    T = defaulteltype()
    N, k, w, rank = 120, 24, 7, 6
    Random.seed!(0xdeadbeef)
    A = lowrank_plus_noise(T, N, rank, noisefloor(T))
    G = deviceoperator(A)
    values = map((1, 2)) do nb
        Random.seed!(0xbeef)
        plan = testplan()
        _, S = minirsvd(G, T, N, k, plan, w; nbuffers=nb)
        svdvals(S)
    end
    @test values[1] ≈ values[2] rtol=blastol(T, N, k)
end

@testset "mini RSVD through a matrix free operator" begin
    T = defaulteltype()
    N, k, w, rank = 120, 24, 7, 6
    Random.seed!(0xdeadbeef)
    # A spectrum that halves all the way down would give the subspace iteration
    # a matrix of condition number 2^k to orthonormalize, and CholeskyQR2 cannot
    # hold that in single precision. Here the retained values halve and the rest
    # sit on a floor. That is the shape a randomized method is meant for anyway.
    G = CircularConvolution(T.(circulant_kernel([m <= rank ? 2.0^-(m - 1) : 1e-6 for m in 1:N])))
    @test check_operator(G; backend=current_backend())

    plan = testplan()
    _, S = minirsvd(G, T, N, k, plan, w)
    @test svdvals(S)[1:rank] ≈ svdvals(densematrix(G))[1:rank] rtol=spectrum_tol(T)
end
