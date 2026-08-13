# A diagnostic for strided host copies. Run it on a CUDA machine:
#
#     julia --project=. test/manual/row_block_copies.jl
#
# gram and cholqr2! traverse blocks of rows. A row block is a slice of every
# column panel, so staging one means copying view(host_panel, rows, :), an
# N-strided 2D region of pinned host memory, into a contiguous column range of
# the device buffer. Everything else Funicular copies is contiguous, so this is
# the only place the CUDA backend has to copy a strided region, and the CPU
# reference backend makes that copy look free.
#
# The questions, in order:
#
#   1. Does copyto! on such a view reach cudaMemcpy2DAsync, or does CUDA.jl
#      materialize the view into a fresh host Array first? A materialization
#      allocates a panel per step and, being pageable, copies synchronously.
#      That would break both the allocation discipline and the pipeline.
#   2. Is the copy actually asynchronous against the issuing stream?
#   3. What does it cost next to a contiguous copy of the same bytes?
#
# The CUDA extension does not rely on copyto! for this: h2d! and d2h! call
# CUDA.unsafe_copy2d! directly for every copy, contiguous or strided. Section 2,
# which checks that call and its keywords, is the load-bearing one, and
# test/gpu_smoke.jl exercises the same call through Funicular. Sections 1, 3 and
# 4 are still worth knowing: what a strided copy costs against a contiguous one
# sets the price of the row traversal, and if copyto! turns out to be fine then
# the direct call could go.

using CUDA, LinearAlgebra, Printf, Random

CUDA.functional() || error("no functional CUDA device")

const T = ComplexF64
const N = 1 << 20        # rows of one panel
const W = 32             # columns of one panel
const ROWS = 1 << 16     # rows in one block

host = Array{T}(undef, N, W)
rand!(host)
pinned = CUDA.pin(host)
@printf("pinned host panel: %d×%d %s, %.1f MiB\n", N, W, T, sizeof(host) / 2^20)
println("CUDA.pin returned a ", typeof(pinned))

device = CuArray{T}(undef, ROWS, W)
block = 1:ROWS
src = view(host, block, :)
@printf("\nsource view: %s\nstrides %s against a contiguous %s\n", typeof(src), strides(src), strides(similar(src)))

stream = CuStream()

println("\n--- 1. does the strided copy allocate on the host? ---")
CUDA.@sync copyto!(device, src)
bytes = @allocated copyto!(device, src)
@printf("host allocation for one strided copy: %d B (a materialized view would be about %d B)\n", bytes, sizeof(src))
@printf("correct: %s\n", Array(device) == host[block, :])

println("\n--- 2. is CUDA.unsafe_copy2d! reachable, and does it agree? ---")
fill!(device, zero(T))
try
    GC.@preserve host begin
        CUDA.unsafe_copy2d!(pointer(device), CUDA.DeviceMemory,
                            pointer(host, first(block)), CUDA.HostMemory,
                            ROWS, W;
                            srcPos=(1, 1), dstPos=(1, 1),
                            srcPitch=N * sizeof(T),
                            dstPitch=ROWS * sizeof(T),
                            async=true, stream=stream)
    end
    synchronize(stream)
    @printf("unsafe_copy2d! ran, correct: %s\n", Array(device) == host[block, :])
catch err
    println("unsafe_copy2d! did not work as called here:")
    showerror(stdout, err)
    println("\nIf the signature has moved, please paste `methods(CUDA.unsafe_copy2d!)`.")
end

println("\n--- 3. is the strided copy asynchronous on an explicit stream? ---")
fill!(device, zero(T))
issue = @elapsed CUDA.stream!(stream) do
    copyto!(device, src)
end
wait = @elapsed synchronize(stream)
@printf("issue %.3f ms, then synchronize %.3f ms\n", 1e3 * issue, 1e3 * wait)
println("(issue much smaller than synchronize means the copy really was queued)")

println("\n--- 4. what does the stride cost? ---")
flat = Array{T}(undef, ROWS, W)
rand!(flat)
CUDA.pin(flat)
CUDA.@sync copyto!(device, flat)
strided = CUDA.@elapsed CUDA.@sync copyto!(device, src)
contiguous = CUDA.@elapsed CUDA.@sync copyto!(device, flat)
@printf("strided %.3f ms (%.1f GB/s), contiguous %.3f ms (%.1f GB/s)\n",
        1e3 * strided, sizeof(flat) / strided / 2^30,
        1e3 * contiguous, sizeof(flat) / contiguous / 2^30)

println("\n--- 5. the writeback direction ---")
back = Array{T}(undef, N, W)
fill!(back, zero(T))
CUDA.pin(back)
CUDA.@sync copyto!(view(back, block, :), device)
@printf("device to strided host view correct: %s\n", back[block, :] == Array(device))
@printf("host allocation for one strided writeback: %d B\n", @allocated copyto!(view(back, block, :), device))

println("\n--- 6. the CUBLAS calls the row traversal makes ---")
buffer = CuArray{T}(undef, ROWS, 4 * W)
rand!(buffer)
partial = view(buffer, 1:(ROWS ÷ 2), :)
gram = CuArray{T}(undef, 4 * W, 4 * W)
fill!(gram, zero(T))
CUDA.stream!(stream) do
    mul!(gram, adjoint(partial), partial, one(T), one(T))
end
synchronize(stream)
@printf("mul! on a row-strided view accumulated, allocation: %d B\n",
        CUDA.@allocated CUDA.stream!(() -> mul!(gram, adjoint(partial), partial, one(T), one(T)), stream))

factor = CuArray(triu(rand(T, 4 * W, 4 * W)) + 4I)
CUDA.stream!(stream) do
    rdiv!(partial, UpperTriangular(factor))
end
synchronize(stream)
@printf("rdiv! on a row-strided view ran, allocation: %d B\n",
        CUDA.@allocated CUDA.stream!(() -> rdiv!(partial, UpperTriangular(factor)), stream))

println("\ndone")
