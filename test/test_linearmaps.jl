# Both directions of the adapter. A LinearMap has to satisfy the contract, and
# an operator that satisfies the contract has to come back out as a LinearMap
# that means the same thing.

@testset "a LinearMap satisfies the operator contract" begin
    T = defaulteltype()
    N, k, w = 20, 7, 3
    dense = randmatrix(T, N, N)
    A = LinearMap(deviceoperator(dense))

    @test Funicular.panel_capable(A)
    # LinearMaps declines to inspect a matrix it wraps, so a map is Hermitian
    # only when it was declared to be, and the trait reports that declaration.
    @test !Funicular.ishermitian_op(A)
    @test Funicular.ishermitian_op(LinearMap(deviceoperator(Matrix(Hermitian(dense))); ishermitian=true))
    @test check_operator(A; backend=current_backend())

    plan = testplan()
    X = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    panelmul!(Y, A, X)
    @test Matrix(Y) ≈ dense * Matrix(X) rtol=blastol(T, N, k)
    @test project(X, A) ≈ Matrix(X)' * dense * Matrix(X) rtol=blastol(T, N, k)
end

# A chain of vector-only FunctionMaps is a CompositeMap, whose matrix mul!
# materializes an intermediate factor into host arrays instead of looping over
# the columns. Note, however, that on the CPU backend this succeeds silently, so
# the apply counter is what catches it: materializing an N×N factor costs N
# applies, while a column loop costs one per column.
@testset "a matrix-free composition is applied one column at a time" begin
    T = defaulteltype()
    N, k, w, n = 20, 6, 3, 4
    dense = randmatrix(T, N, N)
    device = deviceoperator(dense)
    deviceadj = conjugate_transpose(device)

    applies = Ref(0)
    forward!(y, x) = (applies[] += 1; mul!(y, device, x))
    adjointly!(y, x) = (applies[] += 1; mul!(y, deviceadj, x))
    factor = LinearMap{T}(forward!, adjointly!, N, N; ismutating=true)
    composite = factor * factor * factor

    @test composite isa LinearMaps.CompositeMap
    @test !Funicular.panel_capable(factor)
    @test !Funicular.panel_capable(composite)
    @test !Funicular.panel_capable(adjoint(composite))
    @test !Funicular.panel_capable(composite + factor * factor * factor)
    # A map that wraps a matrix keeps the claim, since its mul! is a real GEMM.
    @test Funicular.panel_capable(LinearMap(device))
    @test Funicular.panel_capable(adjoint(LinearMap(device)))

    # Three factors, once per probe column of the composition and once per
    # probe column of its adjoint. A panel probe on top would add another 3n.
    applies[] = 0
    @test check_operator(composite; backend=current_backend(), n=n)
    @test applies[] == 2 * 3 * n

    plan = testplan()
    X = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    applies[] = 0
    panelmul!(Y, composite, X)
    @test Matrix(Y) ≈ dense^3 * Matrix(X) rtol=blastol(T, N, k)
    @test applies[] == 3 * k
end

@testset "an operator comes back out as a LinearMap" begin
    T = defaulteltype()
    N, k, w = 20, 7, 3
    dense = randmatrix(T, N, N)
    G = ColumnOperator(deviceoperator(dense))
    A = Funicular.linearmap(G)

    @test A isa LinearMap
    @test size(A) == (N, N)
    @test eltype(A) === T
    @test check_operator(A; backend=current_backend())

    x = todevice(current_backend(), randmatrix(T, N, 1))
    y = zerobuffer(T, N, 1)
    mul!(view(y, :, 1), A, view(x, :, 1))
    @test host(y) ≈ dense * host(x) rtol=blastol(T, N, 1)
    mul!(view(y, :, 1), adjoint(A), view(x, :, 1))
    @test host(y) ≈ dense' * host(x) rtol=blastol(T, N, 1)

    # A Hermitian operator declares itself through the map, because a LinearMaps
    # caller dispatches on the map's own trait.
    H = DeclaredHermitian(deviceoperator(Matrix(Hermitian(dense))))
    @test ishermitian(Funicular.linearmap(H))

    plan = testplan()
    X = PanelMatrix(randmatrix(T, N, k); plan=plan, w=w)
    Y = PanelMatrix{T}(undef, N, k; plan=plan, w=w)
    panelmul!(Y, A, X)
    @test Matrix(Y) ≈ dense * Matrix(X) rtol=blastol(T, N, k)

    @test Funicular.linearmap(A) === A
end
