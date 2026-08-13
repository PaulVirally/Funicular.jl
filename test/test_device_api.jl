# The parts particular to the CPU reference backend. The vendor API contract
# every backend has to meet is in test_backend.jl.

@testset "CPU backend traits" begin
    b = CPUBackend()
    @test devicetype(b, ComplexF64) === Matrix{ComplexF64}
    @test devicetype(b, ComplexF32) === Matrix{ComplexF32}
    @test alloc_device(b, ComplexF64, (7, 3)) isa Matrix{ComplexF64}
    @test supports_eltype(b, ComplexF64)
    @test !transfers_are_real(b)
    @test ka_backend(b) === KernelAbstractions.CPU()
    @test_throws ArgumentError CPUBackend(jitter=-1.0)
end

@testset "queued work is deferred" begin
    b = CPUBackend()
    queue = make_queue(b)
    gate = Base.Event()
    ran = Ref(false)
    submit!(b, queue, () -> wait(gate))
    submit!(b, queue, () -> ran[] = true)
    @test !ran[]
    notify(gate)
    sync_queue(b, queue)
    @test ran[]
end

@testset "an event with nothing behind it" begin
    @test sync_event(CPUBackend(), TaskEvent(nothing)) === nothing
end

@testset "cuda_backend" begin
    if any(backend -> backendlabel(backend) == "CUDABackend", available_backends())
        @test Funicular.cuda_backend() isa DeviceBackend
    else
        @test_throws ArgumentError Funicular.cuda_backend()
    end
end

@testset "metal_backend" begin
    if any(backend -> backendlabel(backend) == "MetalBackend", available_backends())
        @test Funicular.metal_backend() isa DeviceBackend
    else
        @test_throws ArgumentError Funicular.metal_backend()
    end
end

@testset "human_readable_bytes" begin
    @test human_readable_bytes(512) == "512 B"
    @test human_readable_bytes(1024) == "1.0 KiB"
    @test human_readable_bytes(3 * 2^30) == "3.0 GiB"
    @test endswith(human_readable_bytes(5 * 2^50), "TiB")
end
