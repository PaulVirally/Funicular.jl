# The shared KernelAbstractions kernels, on every backend. Each is checked
# against the LinearAlgebra operation it stands in for, on whole buffers and on
# the ragged views a sweep hands it.

@testset "convert_copy! $T" for T in testeltypes()
    b = current_backend()
    S = T <: Complex ? Complex{Float16} : Float16
    A = randmatrix(T, 9, 5)
    src = todevice(b, A)
    dst = alloc_device(b, T, (9, 5))

    @test convert_copy!(b, dst, src) === dst
    @test host(dst) == A

    narrow = alloc_device(b, S, (9, 5))
    convert_copy!(b, narrow, src)
    @test host(narrow) == S.(A)
    convert_copy!(b, dst, narrow)
    @test host(dst) == T.(S.(A))

    # The shapes a sweep produces: a column slice for a ragged panel and a row
    # slice for the last block of a row traversal.
    fill!(dst, zero(T))
    convert_copy!(b, view(dst, :, 1:2), view(src, :, 4:5))
    convert_copy!(b, view(dst, 1:3, 3:5), view(src, 7:9, 1:3))
    @test host(dst)[:, 1:2] == A[:, 4:5]
    @test host(dst)[1:3, 3:5] == A[7:9, 1:3]

    @test_throws DimensionMismatch convert_copy!(b, dst, view(src, :, 1:3))
end

@testset "gram_accumulate! $T" for T in testeltypes()
    b = current_backend()
    A, B = randmatrix(T, 12, 5), randmatrix(T, 12, 4)
    tol = blastol(T, 12, 5)
    Ad, Bd = todevice(b, A), todevice(b, B)
    C = alloc_device(b, T, (5, 4))
    fill!(C, zero(T))

    @test host(gram_accumulate!(b, C, Ad, Bd, one(T), zero(T))) ≈ A' * B rtol=tol
    gram_accumulate!(b, C, Ad, Bd, one(T), one(T))
    @test host(C) ≈ 2 .* (A' * B) rtol=tol

    kernelled = alloc_device(b, T, (5, 4))
    fill!(kernelled, zero(T))
    gram_accumulate_ka!(b, kernelled, Ad, Bd, one(T), zero(T))
    @test host(kernelled) ≈ A' * B rtol=tol

    # A buffer out of the pool has whatever its last owner left in it, so a zero
    # β has to overwrite rather than scale.
    fill!(kernelled, T(NaN))
    gram_accumulate!(b, kernelled, Ad, Bd, one(T), zero(T))
    @test host(kernelled) ≈ A' * B rtol=tol

    # A row block of the traversal is a view of the buffer, and the panel it
    # meets is a view too.
    block = view(C, 1:3, 1:2)
    gram_accumulate!(b, block, view(Ad, 1:6, 1:3), view(Bd, 1:6, 1:2), one(T), zero(T))
    @test host(C)[1:3, 1:2] ≈ A[1:6, 1:3]' * B[1:6, 1:2] rtol=tol

    @test_throws DimensionMismatch gram_accumulate_ka!(b, C, Ad, view(Bd, :, 1:2), one(T), zero(T))
end

@testset "rightmul_gemm! $T" for T in testeltypes()
    b = current_backend()
    A, C = randmatrix(T, 12, 5), randmatrix(T, 5, 4)
    tol = blastol(T, 12, 5)
    Ad, Cd = todevice(b, A), todevice(b, C)
    dst = alloc_device(b, T, (12, 4))
    fill!(dst, zero(T))

    @test host(rightmul_gemm!(b, dst, Ad, Cd)) ≈ A * C rtol=tol

    kernelled = alloc_device(b, T, (12, 4))
    fill!(kernelled, T(NaN))
    rightmul_gemm_ka!(b, kernelled, Ad, Cd)
    @test host(kernelled) ≈ A * C rtol=tol

    # The ragged last block of a row traversal, which cuts the rows of the
    # destination and of the block it multiplies.
    fill!(dst, zero(T))
    rightmul_gemm!(b, view(dst, 1:7, :), view(Ad, 1:7, :), Cd)
    @test host(dst)[1:7, :] ≈ A[1:7, :] * C rtol=tol
    @test all(iszero, host(dst)[8:12, :])

    @test_throws DimensionMismatch rightmul_gemm_ka!(b, dst, Ad, view(Cd, :, 1:3))
end

@testset "rdiv_upper! $T" for T in testeltypes()
    b = current_backend()
    Y = randmatrix(T, 7, 5)
    R = Matrix(UpperTriangular(randmatrix(T, 5, 5) + 4I))
    tol = blastol(T, 7, 5)
    Rd = todevice(b, R)

    Yd = todevice(b, Y)
    @test host(rdiv_upper!(b, Yd, Rd)) ≈ Y / UpperTriangular(R) rtol=tol

    kernelled = todevice(b, Y)
    rdiv_upper_ka!(b, kernelled, Rd)
    @test host(kernelled) ≈ Y / UpperTriangular(R) rtol=tol

    # The ragged last block of a row traversal, a view with a row stride.
    strided = todevice(b, Y)
    rdiv_upper!(b, view(strided, 1:3, :), Rd)
    @test host(strided)[1:3, :] ≈ Y[1:3, :] / UpperTriangular(R) rtol=tol
    @test host(strided)[4:7, :] == Y[4:7, :]

    @test_throws DimensionMismatch rdiv_upper_ka!(b, Yd, view(Rd, 1:4, 1:4))
end
