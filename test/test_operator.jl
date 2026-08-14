checkop(G; kwargs...) = check_operator(G; backend=current_backend(), kwargs...)

@testset "trait defaults" begin
    T = defaulteltype()
    G = deviceoperator(randmatrix(T, 6, 6))
    @test workspace_bytes(G) == 0
    @test panel_capable(G)
    @test !ishermitian_op(G)
    @test ishermitian_op(deviceoperator(Matrix(Hermitian(randmatrix(T, 6, 6)))))

    column = ColumnOperator(G)
    @test workspace_bytes(column) == 0
    @test !panel_capable(column)
    @test !ishermitian_op(column)
    @test workspace_bytes(Workspaced(G, 1 << 20)) == 1 << 20
end

function contract_abiding_operators()
    T = defaulteltype()
    dense = deviceoperator(randmatrix(T, 12, 12))
    (dense,
     deviceoperator(Matrix(Hermitian(randmatrix(T, 12, 12)))),
     ColumnOperator(dense),
     CircularConvolution(randmatrix(T, 12, 1)[:, 1]),
     Workspaced(dense, 4096))
end

@testset "check_operator accepts $(nameof(typeof(G)))" for G in contract_abiding_operators()
    @test checkop(G)
    @test checkop(G; n=1)
    @test checkop(adjoint(G))
end

@testset "check_operator accepts a rectangular operator" begin
    @test checkop(deviceoperator(randmatrix(defaulteltype(), 9, 5)))
end

@testset "check_operator accepts a real operator" begin
    @test checkop(deviceoperator(randmatrix(real(defaulteltype()), 8, 8)))
end

@testset "check_operator catches a wrong adjoint" begin
    T = defaulteltype()
    @test_throws ArgumentError checkop(TransposedAdjoint(deviceoperator(randmatrix(T, 10, 10))))
    # A real operator's transpose is its adjoint, so that one has to pass.
    @test checkop(TransposedAdjoint(deviceoperator(randmatrix(real(T), 10, 10))))
end

@testset "check_operator catches a wrong shape" begin
    T = defaulteltype()
    @test_throws ArgumentError checkop(MisshapenAdjoint(deviceoperator(randmatrix(T, 10, 10))))
    @test_throws ArgumentError checkop(MisshapenAdjoint(deviceoperator(randmatrix(T, 9, 5))))
end

@testset "an operator has to match both row counts" begin
    T = defaulteltype()
    G = deviceoperator(randmatrix(T, 9, 5))
    @test Funicular.assert_operator_shape(G, 9, 5) === nothing
    @test_throws ArgumentError Funicular.assert_operator_shape(G, 5, 9)
    @test_throws ArgumentError Funicular.assert_operator_shape(G, 9, 9)
    @test_throws ArgumentError Funicular.assert_square_operator(G, 9)
end

@testset "check_operator catches a panel and column mismatch" begin
    @test_throws ArgumentError checkop(WrongPanel(deviceoperator(randmatrix(defaulteltype(), 10, 10))))
end

@testset "check_operator catches a false Hermitian claim" begin
    T = defaulteltype()
    @test_throws ArgumentError checkop(FalselyHermitian(deviceoperator(randmatrix(T, 10, 10))))
    hermitian = deviceoperator(Matrix(Hermitian(randmatrix(T, 10, 10))))
    @test checkop(FalselyHermitian(hermitian))
end

@testset "check_operator rejects nonsense arguments" begin
    A = deviceoperator(randmatrix(defaulteltype(), 6, 6))
    @test_throws ArgumentError checkop(A; n=0)
    @test_throws ArgumentError check_operator(randmatrix(ComplexF64, 6, 6); backend=NarrowBackend())
end

@testset "the circular convolution operator is what it claims" begin
    T = defaulteltype()
    G = CircularConvolution(randmatrix(T, 9, 1)[:, 1])
    dense = densematrix(G)
    x = view(todevice(current_backend(), randmatrix(T, 9, 1)), :, 1)
    y = similar(x)
    mul!(y, G, x)
    @test host(y) ≈ dense * host(x) rtol=blastol(T, 9, 9)
    @test densematrix(adjoint(G)) ≈ dense'
end
