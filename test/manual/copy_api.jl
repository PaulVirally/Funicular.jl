# A probe of the copy API the CUDA extension rests on. Run it on a CUDA machine:
#
#     julia --project=test/cuda --startup-file=no test/manual/copy_api.jl
#
# When every device copy in the extension throws something that is neither an
# ArgumentError nor an ErrorException, the failure is not one of the extension's
# own checks. That leaves CUDA.unsafe_copy2d! and the two memory type names it
# is passed. This script asks the installed CUDA.jl what it offers, then tries
# the copy four ways so that the extension can be written against whichever way
# works.
#
# Nothing here aborts on failure: each attempt prints its own error and the
# script carries on.

using CUDA, LinearAlgebra, Printf

CUDA.functional() || error("no functional CUDA device")

const T = ComplexF64
const N = 64      # rows of the host panel
const W = 4       # columns
const ROWS = 3:6  # the row block of the strided case

function attempt(name, f)
    print(rpad(name, 52))
    try
        result = f()
        println(result === true ? "ok" : "ok, returned $(repr(result))")
    catch err
        println("FAILED: ", sprint(showerror, err))
    end
end

println("--- 0. what is installed ---")
attempt("versions", () -> string("CUDA.jl ", pkgversion(CUDA), ", device ", CUDA.name(CUDA.device())))
for name in (:unsafe_copy2d!, :unsafe_copyto!, :DeviceMemory, :HostMemory, :UnifiedMemory,
             :pin, :is_pinned, :stream, :stream!, :device_synchronize, :Mem)
    @printf("%-22s %s\n", name, isdefined(CUDA, name) ? "defined" : "MISSING")
end

if isdefined(CUDA, :unsafe_copy2d!)
    println("\nmethods(CUDA.unsafe_copy2d!):")
    show(stdout, MIME"text/plain"(), methods(CUDA.unsafe_copy2d!))
    println()
end
if isdefined(CUDA, :unsafe_copyto!)
    println("\nmethods(CUDA.unsafe_copyto!) with a CuPtr destination:")
    for m in methods(CUDA.unsafe_copyto!)
        occursin("CuPtr", string(m.sig)) && println("  ", m)
    end
end

println("\n--- 1. the pointers the extension takes ---")
host = rand(T, N, W)
CUDA.pin(host)
device = CuArray{T}(undef, N, W)
attempt("pointer(::CuArray)", () -> pointer(device) isa CuPtr{T})
attempt("pointer(::Array)", () -> pointer(host) isa Ptr{T})
attempt("stride(::CuArray, 2)", () -> stride(device, 2) == N)
attempt("stride of a device row view", () -> stride(view(device, 1:4, :), 2) == N)
attempt("pointer + offset on a CuPtr", () -> (pointer(device) + 16) isa CuPtr{T})

println("\n--- 2. exactly what the extension calls ---")
stream = CuStream(flags=CUDA.STREAM_NON_BLOCKING)

# The contiguous case: a whole host panel into a whole device buffer.
function extension_copy2d(dst, dsttyp, src, srctyp, width, height, dstpitch, srcpitch, async)
    GC.@preserve dst src begin
        CUDA.unsafe_copy2d!(dst, dsttyp, src, srctyp, width, height;
                            dstPitch=dstpitch * sizeof(T),
                            srcPitch=srcpitch * sizeof(T),
                            async=async, stream=stream)
    end
end

attempt("unsafe_copy2d! H2D contiguous, async", function ()
    fill!(device, zero(T))
    extension_copy2d(pointer(device), CUDA.DeviceMemory, pointer(host), CUDA.HostMemory,
                     N, W, N, N, true)
    synchronize(stream)
    Array(device) == host
end)

attempt("unsafe_copy2d! H2D contiguous, sync", function ()
    fill!(device, zero(T))
    extension_copy2d(pointer(device), CUDA.DeviceMemory, pointer(host), CUDA.HostMemory,
                     N, W, N, N, false)
    Array(device) == host
end)

# The strided case: a block of rows of the host panel into a device block whose
# columns are shorter. This is the row traversal of gram and cholqr2!.
block = CuArray{T}(undef, length(ROWS), W)
attempt("unsafe_copy2d! H2D strided rows", function ()
    fill!(block, zero(T))
    extension_copy2d(pointer(block), CUDA.DeviceMemory,
                     pointer(host) + (first(ROWS) - 1) * sizeof(T), CUDA.HostMemory,
                     length(ROWS), W, length(ROWS), N, true)
    synchronize(stream)
    Array(block) == host[ROWS, :]
end)

attempt("unsafe_copy2d! D2H strided rows", function ()
    back = zeros(T, N, W)
    CUDA.pin(back)
    extension_copy2d(pointer(back) + (first(ROWS) - 1) * sizeof(T), CUDA.HostMemory,
                     pointer(block), CUDA.DeviceMemory,
                     length(ROWS), W, N, length(ROWS), true)
    synchronize(stream)
    back[ROWS, :] == host[ROWS, :]
end)

attempt("unsafe_copy2d! from pageable host memory", function ()
    pageable = rand(T, N, W)
    fill!(device, zero(T))
    extension_copy2d(pointer(device), CUDA.DeviceMemory, pointer(pageable), CUDA.HostMemory,
                     N, W, N, N, false)
    Array(device) == pageable
end)

println("\n--- 3. the same copies through copyto! ---")
attempt("copyto!(CuArray, Array)", function ()
    fill!(device, zero(T))
    CUDA.stream!(() -> copyto!(device, host), stream)
    synchronize(stream)
    Array(device) == host
end)

attempt("copyto!(column view of CuArray, Array)", function ()
    fill!(device, zero(T))
    CUDA.stream!(() -> copyto!(view(device, :, 1:2), host[:, 1:2]), stream)
    synchronize(stream)
    Array(device)[:, 1:2] == host[:, 1:2]
end)

attempt("copyto!(row view of CuArray, row view of Array)", function ()
    fill!(block, zero(T))
    CUDA.stream!(() -> copyto!(view(block, :, :), view(host, ROWS, :)), stream)
    synchronize(stream)
    Array(block) == host[ROWS, :]
end)

attempt("host allocation of that strided copyto!", function ()
    CUDA.stream!(() -> copyto!(view(block, :, :), view(host, ROWS, :)), stream)
    bytes = CUDA.stream!(stream) do
        @allocated copyto!(view(block, :, :), view(host, ROWS, :))
    end
    @sprintf("%d B (a materialized view would be about %d B)", bytes, length(ROWS) * W * sizeof(T))
end)

println("\n--- 4. the flat pointer copy, as a fallback for the contiguous case ---")
attempt("CUDA.unsafe_copyto!(CuPtr, Ptr, n; async, stream)", function ()
    fill!(device, zero(T))
    GC.@preserve host device begin
        CUDA.unsafe_copyto!(pointer(device), pointer(host), N * W; async=true, stream=stream)
    end
    synchronize(stream)
    Array(device) == host
end)

attempt("CUDA.unsafe_copyto!(CuPtr, Ptr, n) with no keywords", function ()
    fill!(device, zero(T))
    GC.@preserve host device begin
        CUDA.unsafe_copyto!(pointer(device), pointer(host), N * W)
    end
    Array(device) == host
end)

println("\n--- 5. the CUBLAS calls the row traversal makes ---")
buffer = CuArray(rand(T, 32, 12))
partial = view(buffer, 1:16, :)
target = CuArray(zeros(T, 12, 12))
attempt("mul! into a k×k from a row-strided view", function ()
    CUDA.stream!(() -> mul!(target, adjoint(partial), partial, one(T), zero(T)), stream)
    synchronize(stream)
    isapprox(Array(target), Array(partial)' * Array(partial); rtol=1e-10)
end)

factor = CuArray(Matrix(UpperTriangular(rand(T, 12, 12) + 4I)))
attempt("rdiv! of a row-strided view by an UpperTriangular", function ()
    before = Array(partial)
    CUDA.stream!(() -> rdiv!(partial, UpperTriangular(factor)), stream)
    synchronize(stream)
    isapprox(Array(partial), before / UpperTriangular(Array(factor)); rtol=1e-10)
end)

println("\ndone")
