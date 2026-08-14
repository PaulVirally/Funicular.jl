include("setup.jl")

@testset "Funicular" begin
    @testset "residency plan" begin
        include("test_plan.jl")
    end
    @testset "CPU reference backend" begin
        include("test_device_api.jl")
    end
    for backend in available_backends()
        withbackend(backend) do
            @testset "$(backendlabel(backend))" begin
                @testset "device API" begin
                    include("test_backend.jl")
                end
                @testset "kernels" begin
                    include("test_kernels.jl")
                end
                @testset "panels" begin
                    include("test_panels.jl")
                end
                @testset "ghost panels" begin
                    include("test_ghosts.jl")
                end
                @testset "movement" begin
                    include("test_movement.jl")
                end
                @testset "executor" begin
                    include("test_executor.jl")
                end
                @testset "operator contract" begin
                    include("test_operator.jl")
                end
                if HAS_LINEARMAPS
                    @testset "LinearMaps adapter" begin
                        include("test_linearmaps.jl")
                    end
                end
                @testset "panel BLAS" begin
                    include("test_panelblas.jl")
                end
                @testset "mini RSVD" begin
                    include("test_rsvd.jl")
                end
                @testset "factorization chain" begin
                    include("test_chain.jl")
                end
                if HAS_HDF5
                    @testset "disk tier" begin
                        include("test_disk.jl")
                    end
                end
                if backendlabel(backend) == "CUDABackend"
                    @testset "CUDA discipline" begin
                        include("test_cuda.jl")
                    end
                end
            end
        end
    end
    if HAS_HDF5
        @testset "disk interop" begin
            include("test_diskinterop.jl")
        end
        @testset "documented demo" begin
            include("test_demo.jl")
        end
    end
    @testset "API surface" begin
        include("test_api_surface.jl")
    end
    @testset "discipline" begin
        include("test_discipline.jl")
    end
end
