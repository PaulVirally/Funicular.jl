@testset "synchronous movement w=$w" for w in (20, 7, 1)
    T = defaulteltype()
    N, k = 11, 20
    A = randmatrix(T, N, k)
    plan = testplan()
    pm = PanelMatrix(A; plan=plan, w=w)
    buffer = devicebuffer(T, N, w)

    for j in 1:npanels(pm)
        wj = panelwidth(pm, j)
        view_j = view(buffer, :, 1:wj)
        materialize!(view_j, pm.panels[j], plan)
        @test host(view_j) == A[:, panelrange(pm, j)]
        view_j .*= 2
        writeback!(pm.panels[j], view_j, plan)
    end
    @test Matrix(pm) == 2 .* A
end

@testset "movement rejects mismatched buffers" begin
    T = defaulteltype()
    plan = testplan()
    pm = PanelMatrix{T}(undef, 11, 20; plan=plan, w=7)
    last = pm.panels[end]
    @test_throws DimensionMismatch materialize!(zerobuffer(T, 11, 7), last, plan)
    @test_throws DimensionMismatch writeback!(last, zerobuffer(T, 11, 7), plan)
    @test sync_event(current_backend(), materialize!(zerobuffer(T, 11, 6), last, plan)) === nothing
end

@testset "movement with narrowed storage" begin
    T = defaulteltype()
    S = narrowed(T)
    N, k = 9, 15
    A = randmatrix(T, N, k)
    plan = testplan(host_eltype=S)
    pm = PanelMatrix(A; plan=plan, w=4)
    buffer = zerobuffer(T, N, 4)

    for j in 1:npanels(pm)
        wj = panelwidth(pm, j)
        view_j = view(buffer, :, 1:wj)
        materialize!(view_j, pm.panels[j], plan)
        @test host(view_j) == T.(S.(A[:, panelrange(pm, j)]))
        writeback!(pm.panels[j], view_j, plan)
    end
    @test Matrix(pm) == T.(S.(A))
end

@testset "queued movement under jitter" begin
    T = defaulteltype()
    N, k, w = 12, 23, 5
    A = randmatrix(T, N, k)
    for rep in 1:repetitions()
        backend = jittery_backend()
        plan = testplan(backend=backend)
        pm = PanelMatrix(A; plan=plan, w=w)
        up, compute, down = make_queue(backend), make_queue(backend), make_queue(backend)
        buffers = checkout_device_buffers!(plan, T, (N, w), 2)

        for j in 1:npanels(pm)
            wj = panelwidth(pm, j)
            slot = view(buffers[mod1(j, 2)], :, 1:wj)
            wait_event(backend, up, record_event(backend, down))
            materialize!(slot, pm.panels[j], plan; queue=up)
            wait_event(backend, compute, record_event(backend, up))
            submit!(backend, compute, () -> slot .*= 2)
            wait_event(backend, down, record_event(backend, compute))
            writeback!(pm.panels[j], slot, plan; queue=down)
        end
        for queue in (up, compute, down)
            sync_queue(backend, queue)
        end

        @test Matrix(pm) == 2 .* A
        @test all(p -> p.epoch == 2 && p.dirty, pm.panels)
    end
end

@testset "queued movement matches synchronous movement" begin
    T = defaulteltype()
    N, k, w = 10, 17, 6
    A = randmatrix(T, N, k)
    reference = similar(A)
    plan = testplan()
    pm = PanelMatrix(A; plan=plan, w=w)
    buffer = zerobuffer(T, N, w)
    for j in 1:npanels(pm)
        slot = view(buffer, :, 1:panelwidth(pm, j))
        materialize!(slot, pm.panels[j], plan)
        reference[:, panelrange(pm, j)] = host(slot)
    end

    backend = jittery_backend()
    queued_plan = testplan(backend=backend)
    queued = PanelMatrix(A; plan=queued_plan, w=w)
    queued_buffer = zerobuffer(T, N, w)
    out = similar(A)
    queue = make_queue(backend)
    for j in 1:npanels(queued)
        slot = view(queued_buffer, :, 1:panelwidth(queued, j))
        materialize!(slot, queued.panels[j], queued_plan; queue=queue)
        submit!(backend, queue, () -> out[:, panelrange(queued, j)] = host(slot))
    end
    sync_queue(backend, queue)
    @test out == reference
end
