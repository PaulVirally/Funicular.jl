@testset "plan construction" begin
    plan = testplan(scratch_dir="/tmp/funicular-test", workspace_bytes=2^20)
    @test plan.device_budget == 64 * 2^20
    @test plan.devicepool.budget == 64 * 2^20 - 2^20
    @test plan.scratch_dir == "/tmp/funicular-test"
    @test plan.backend isa CPUBackend
    @test occursin("ResidencyPlan", sprint(show, plan))

    @test_throws ArgumentError testplan(nbuffers=0)
    @test_throws ArgumentError testplan(host_budget=0)
    @test_throws ArgumentError testplan(workspace_bytes=-1)
    @test_throws ArgumentError testplan(device_budget=2^20, workspace_bytes=2^20)
    @test_throws ArgumentError testplan(panel_width=0)
    @test_throws ArgumentError testplan(target_panel_bytes=0)
    @test_throws ArgumentError testplan(host_eltype=Complex)
end

@testset "panel width arithmetic" begin
    @test panel_bytes(1000, 8, ComplexF64) == 1000 * 8 * 16
    @test device_bytes_required(1000, 8, ComplexF64; nbuffers=2, workspace_bytes=7) ==
          2 * 1000 * 8 * 16 + 7

    # One panel when everything fits, and the panel count grows as the budget shrinks.
    @test choose_panel_width(100, 64, ComplexF64; device_budget=1 << 30) == 64
    @test choose_panel_width(100, 64, ComplexF64; device_budget=2 * 2 * 100 * 16) == 2

    # Panels are evened out instead of leaving a width-1 remainder.
    @test choose_panel_width(100, 1000, ComplexF64; device_budget=2 * 999 * 100 * 16) == 500

    @test choose_panel_width(100, 64, ComplexF64; device_budget=1 << 30,
                             target_panel_bytes=8 * 100 * 16) == 8
    @test choose_panel_width(100, 64, ComplexF64; device_budget=1 << 30,
                             target_panel_bytes=1) == 1

    @test_throws ArgumentError choose_panel_width(100, 64, ComplexF64; device_budget=100 * 16)
    @test_throws ArgumentError choose_panel_width(0, 64, ComplexF64; device_budget=1 << 30)
    @test_throws ArgumentError choose_panel_width(100, 0, ComplexF64; device_budget=1 << 30)
end

@testset "panel width properties" begin
    rng = Xoshiro(0xdeadbeef)
    for _ in 1:400
        N = rand(rng, 1:5000)
        k = rand(rng, 1:2000)
        T = rand(rng, (ComplexF32, ComplexF64, Float64))
        nbuffers = rand(rng, 1:2)
        extra = rand(rng, 0:2)
        workspace = rand(rng, 0:(1 << 20))
        column = N * sizeof(T)
        budget = workspace + (nbuffers + extra) * column * rand(rng, 1:2000)
        target = rand(rng, (1, 1 << 10, 1 << 20, 2 * 1024^3))

        w = choose_panel_width(N, k, T; device_budget=budget, nbuffers=nbuffers,
                               extra_buffers=extra, workspace_bytes=workspace,
                               target_panel_bytes=target)
        per_buffer = fld(budget - workspace, nbuffers + extra)
        uncapped = min(k, fld(per_buffer, column))
        capped = min(k, max(1, fld(min(target, per_buffer), column)))

        @test 1 <= w <= uncapped
        @test device_bytes_required(N, w, T; nbuffers=nbuffers, extra_buffers=extra,
                                    workspace_bytes=workspace) <= budget
        @test cld(k, w) == cld(k, capped)
        @test w == 1 || panel_bytes(N, w, T) <= max(target, column)
    end
end

@testset "plan width resolution" begin
    plan = testplan(device_budget=2 * 64 * 100 * 16)
    @test resolve_panel_width(plan, 100, 256, ComplexF64) == 64
    @test resolve_panel_width(plan, 100, 256, ComplexF64; override=17) == 17
    @test resolve_panel_width(plan, 100, 256, ComplexF64; extra_buffers=2) == 32
    @test_throws ArgumentError resolve_panel_width(plan, 100, 256, ComplexF64; override=0)
    @test_throws ArgumentError resolve_panel_width(plan, 100, 256, ComplexF64; override=257)
    @test_throws ArgumentError resolve_panel_width(plan, 100, 256, ComplexF64; override=128)

    fixed = testplan(device_budget=1 << 30, panel_width=13)
    @test resolve_panel_width(fixed, 100, 256, ComplexF64) == 13
    @test resolve_panel_width(fixed, 100, 256, ComplexF64; override=8) == 8

    message = try
        resolve_panel_width(plan, 100, 256, ComplexF64; override=128)
    catch err
        err.msg
    end
    @test occursin("device_budget", message)
    @test occursin("panel_width ≤ 64", message)
end

@testset "device pool" begin
    plan = testplan(device_budget=5 * 64 * 16 + 64)
    buffers = checkout_device_buffers!(plan, ComplexF64, (64, 1), 2)
    @test length(buffers) == 2
    @test buffers[1] !== buffers[2]
    @test device_bytes_allocated(plan) == 2 * 64 * 16

    checkin_device_buffers!(plan, buffers)
    again = checkout_device_buffers!(plan, ComplexF64, (64, 1), 2)
    @test Set(objectid.(again)) == Set(objectid.(buffers))
    @test device_bytes_allocated(plan) == 2 * 64 * 16

    # A nested sweep must get buffers of its own, not the outer sweep's.
    nested = checkout_device_buffers!(plan, ComplexF64, (64, 1), 3)
    @test isempty(intersect(objectid.(nested), objectid.(again)))
    @test device_bytes_allocated(plan) == 5 * 64 * 16
    @test_throws ArgumentError checkout_device_buffers!(plan, ComplexF64, (64, 1), 1)

    narrow = checkout_device_buffers!(plan, ComplexF32, (4, 1), 1)
    @test eltype(narrow[1]) === ComplexF32
end

@testset "host pool" begin
    plan = testplan(host_budget=4 * 2^20)
    pool = plan.hostpool
    blocks = [alloc_host_storage!(plan, ComplexF64, (2^16, 1)) for _ in 1:4]
    @test all(s -> size(s.array) == (2^16, 1), blocks)
    @test host_bytes_reserved(pool) == pool.budget
    @test host_bytes_in_use(pool) == 4 * 2^20
    @test host_bytes_free(pool) == 0
    @test_throws ArgumentError alloc_host_storage!(plan, ComplexF64, (2^16, 1))

    Funicular.release_host_block!(plan, blocks[2].block)
    @test host_bytes_free(pool) == 2^20
    @test host_bytes_in_use(pool) == 3 * 2^20
    reused = alloc_host_storage!(plan, ComplexF64, (2^16, 1))
    @test reused.block == blocks[2].block
    @test host_bytes_free(pool) == 0

    # Blocks sharing a slab must not overlap.
    fill!(blocks[3].array, 1)
    fill!(blocks[4].array, 2)
    @test blocks[3].block.slab == blocks[4].block.slab
    @test all(==(1), blocks[3].array)
end

@testset "host pool grows in slabs" begin
    plan = testplan(host_budget=8 * 2^20)
    pool = plan.hostpool
    @test host_bytes_reserved(pool) == 0
    alloc_host_storage!(plan, ComplexF64, (2^16, 1))
    @test host_bytes_reserved(pool) == 2^20
    for _ in 1:7
        alloc_host_storage!(plan, ComplexF64, (2^16, 1))
    end
    @test host_bytes_reserved(pool) == 8 * 2^20
    @test length(pool.slabs) == 4
end
