# The demo in docs/demo/minirsvd.jl has to keep working, so it runs here rather
# than only on a CUDA machine. The substitutions below are exactly the ones its
# last section tells a reader without a GPU to make, plus smaller sizes so it
# takes a second instead of a minute. Each substitution is checked to have
# applied, so a rename in the demo fails this test rather than quietly skipping
# half of it.

const DEMO = joinpath(@__DIR__, "..", "docs", "demo", "minirsvd.jl")

const SUBSTITUTIONS = ["using CUDA\n" => "",
                       "CuArray(" => "Array(",
                       "Funicular.cuda_backend()" => "CPUBackend()",
                       "const N = 1 << 20" => "const N = 1 << 10",
                       "device_budget=2 * 2^30" => "device_budget=64 * 2^20",
                       "host_budget=2 * 2^30" => "host_budget=16 * 1024 * 8 * 16"]

@testset "the documented demo runs on the reference backend" begin
    source = read(DEMO, String)
    for (from, to) in SUBSTITUTIONS
        @test occursin(from, source)
        source = replace(source, from => to)
    end
    demo = Module(:MiniRSVDDemo)
    redirect_stdout(devnull) do
        Base.include_string(demo, source, DEMO)
    end
    # The demo's own assertions carry the correctness. The check here is that
    # the run went through the disk tier: exercising that half of the demo is
    # why host_budget is made smaller.
    @test demo.Funicular.disk_reads(demo.Y) + demo.Funicular.disk_reads(demo.Z) > 0
end
