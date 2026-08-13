# The script the reload test runs in a process of its own. It knows nothing of
# this one beyond the file it is pointed at.
const RELOAD_SCRIPT = """
    using Funicular, HDF5, Serialization
    path, out = ARGS
    plan = ResidencyPlan(device_budget = 1 << 24, host_budget = 1 << 24)
    pm = load(PanelMatrix, path; plan = plan, readonly = true)
    serialize(out, (size(pm), panelwidth(pm), gram(pm)))
    Funicular.free!(pm)
"""

@testset "a saved matrix reloads in another process" begin
    T = ComplexF64
    N, k, w = 14, 30, 6
    A = randmatrix(T, N, k)
    mktempdir() do dir
        path = joinpath(dir, "restart.h5")
        save(PanelMatrix(A; plan=testplan(backend=CPUBackend()), w=w), path)

        script = joinpath(dir, "reload.jl")
        write(script, RELOAD_SCRIPT)
        answer = joinpath(dir, "gram.jls")
        run(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no $script $path $answer`)

        shape, width, G = deserialize(answer)
        @test shape == (N, k)
        @test width == w
        @test G ≈ A' * A rtol=blastol(T, N, k)
    end
end

if HAS_DISKARRAYS
    @testset "a saved matrix reads as a DiskArrays array" begin
        T = ComplexF64
        N, k, w = 13, 29, 6
        A = randmatrix(T, N, k)
        mktempdir() do dir
            path = joinpath(dir, "interop.h5")
            save(PanelMatrix(A; plan=testplan(backend=CPUBackend()), w=w), path)

            da = Funicular.diskarray(path)
            try
                @test da isa DiskArrays.AbstractDiskArray
                @test size(da) == (N, k)
                @test eltype(da) === T
                @test DiskArrays.haschunks(da) isa DiskArrays.Chunked
                @test first(DiskArrays.eachchunk(da)) == (1:N, 1:w)
                @test length(DiskArrays.eachchunk(da)) == cld(k, w)

                # The wrapper exists so that generic chunked-array code works.
                @test Array(da) == A
                @test da[:, 3:9] == A[:, 3:9]
                @test sum(da) ≈ sum(A) rtol=blastol(T, N, k)
                @test collect(da[7, :]) == A[7, :]
            finally
                close(da)
            end
        end
    end
end
