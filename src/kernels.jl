# Device kernels written once, in KernelAbstractions, and launched on whatever
# backend the plan carries. A vendor extension implements the device API and gets
# these for free. It overrides one of them only when the vendor's own library
# leaves a gap, as the Metal extension does for the GEMM and for the triangular
# solve.

@kernel function convert_copy_kernel!(dst, @Const(src))
    I = @index(Global, Cartesian)
    @inbounds dst[I]=convert(eltype(dst), src[I])
end

# Widens or narrows precision. Both arrays are device resident, so on a backend
# with unified memory this is also how a host panel reaches a device buffer.
function convert_copy!(backend::DeviceBackend, dst, src)
    size(dst) == size(src) || throw(DimensionMismatch("copy between a $(size(dst)) destination and a $(size(src)) source"))
    isempty(dst) && return dst
    kernel = convert_copy_kernel!(ka_backend(backend))
    kernel(dst, src; ndrange=size(dst))
    dst
end

@kernel function gram_accumulate_kernel!(C, @Const(A), @Const(B), α, β)
    i, j = @index(Global, NTuple)
    acc = zero(eltype(C))
    @inbounds for l in axes(A, 1)
        acc += conj(A[l, i]) * B[l, j]
    end
    # A zero β overwrites rather than scales, since the sweep's first step meets
    # a buffer straight out of the pool and whatever it happens to hold.
    @inbounds C[i, j]=iszero(β) ? α * acc : α * acc + β * C[i, j]
end

"""
    gram_accumulate!(backend, C, A, B, α, β)

Computes `C ← α A' B + β C`. The default is the vendor's GEMM through
`LinearAlgebra.mul!`. A backend whose GEMM does not reach every combination of
views Funicular hands it overrides this with [`gram_accumulate_ka!`](@ref).
"""
gram_accumulate!(::DeviceBackend, C, A, B, α, β) = mul!(C, adjoint(A), B, α, β)

"""
    gram_accumulate_ka!(backend, C, A, B, α, β)

Computes the same product as `gram_accumulate!` through a kernel, one thread per
entry of `C`, each summing down the shared dimension. It has none of a tuned
GEMM's blocking and is meant for backends that cannot offer one.
"""
function gram_accumulate_ka!(backend::DeviceBackend, C, A, B, α, β)
    (size(A, 1) == size(B, 1) && size(C) == (size(A, 2), size(B, 2))) || throw(DimensionMismatch("cannot accumulate the $(size(A))' times $(size(B)) product into a $(size(C)) matrix"))
    isempty(C) && return C
    kernel = gram_accumulate_kernel!(ka_backend(backend))
    kernel(C, A, B, α, β; ndrange=size(C))
    C
end

@kernel function rightmul_gemm_kernel!(dst, @Const(A), @Const(C))
    i, j = @index(Global, NTuple)
    acc = zero(eltype(dst))
    @inbounds for l in axes(A, 2)
        acc += A[i, l] * C[l, j]
    end
    @inbounds dst[i, j]=acc
end

"""
    rightmul_gemm!(backend, dst, A, C)

Computes `dst ← A C`. The default is the vendor's GEMM through
`LinearAlgebra.mul!`. A backend whose GEMM does not reach every combination of
views Funicular hands it overrides this with [`rightmul_gemm_ka!`](@ref).
"""
rightmul_gemm!(::DeviceBackend, dst, A, C) = mul!(dst, A, C)

"""
    rightmul_gemm_ka!(backend, dst, A, C)

Computes the same product as `rightmul_gemm!` through a kernel, one thread per
entry of `dst`, each summing along the shared dimension. It has none of a tuned
GEMM's blocking and is meant for backends that cannot offer one.
"""
function rightmul_gemm_ka!(backend::DeviceBackend, dst, A, C)
    (size(A, 2) == size(C, 1) && size(dst) == (size(A, 1), size(C, 2))) || throw(DimensionMismatch("cannot multiply a $(size(A)) block by a $(size(C)) factor into a $(size(dst)) matrix"))
    isempty(dst) && return dst
    kernel = rightmul_gemm_kernel!(ka_backend(backend))
    kernel(dst, A, C; ndrange=size(dst))
    dst
end

@kernel function rdiv_upper_kernel!(Y, @Const(R))
    row = @index(Global)
    @inbounds for j in axes(Y, 2)
        acc = Y[row, j]
        for i in 1:(j - 1)
            acc -= Y[row, i] * R[i, j]
        end
        Y[row, j] = acc / R[j, j]
    end
end

"""
    rdiv_upper!(backend, Y, R)

Computes `Y ← Y R⁻¹` in place for upper triangular `R`. The default goes through
`LinearAlgebra.rdiv!`, which reaches the vendor's TRSM. A backend whose arrays
have no triangular solve overrides this with [`rdiv_upper_ka!`](@ref).
"""
rdiv_upper!(::DeviceBackend, Y::AbstractMatrix, R::AbstractMatrix) = rdiv!(Y, UpperTriangular(R))

"""
    rdiv_upper_ka!(backend, Y, R)

Performs the same solve as [`rdiv_upper!`](@ref) through a kernel, one thread
per row of `Y`, each substituting forward along the `k` columns. It reads all of
`R` for every row and is meant for backends that have no TRSM rather than as
competition for one.
"""
function rdiv_upper_ka!(backend::DeviceBackend, Y::AbstractMatrix, R::AbstractMatrix)
    size(R, 1) == size(R, 2) == size(Y, 2) || throw(DimensionMismatch("a $(size(Y)) block cannot be divided by a $(size(R)) triangular factor"))
    isempty(Y) && return Y
    kernel = rdiv_upper_kernel!(ka_backend(backend))
    kernel(Y, R; ndrange=size(Y, 1))
    Y
end
