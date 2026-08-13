# Feature-creep tripwire: the export list is fixed, and adding to it means
# editing this test on purpose.
const PUBLIC_API = [:CPUBackend, :Funicular, :GhostPanels, :PanelMatrix,
                    :ResidencyPlan, :check_operator, :cholqr2!, :foreachpanel,
                    :gram, :load, :npanels, :panelmul!, :panelrange,
                    :panelwidth, :project, :save, :scale!]

@testset "public API surface" begin
    @test sort(names(Funicular)) == PUBLIC_API
end

@testset "exports are documented" begin
    docs = Base.Docs.meta(Funicular)
    for name in PUBLIC_API
        name === :Funicular && continue
        @test haskey(docs, Base.Docs.Binding(Funicular, name))
    end
end

@testset "src never imports a vendor package" begin
    forbidden = ("using CUDA", "import CUDA", "using Metal", "import Metal",
                 "using HDF5", "import HDF5", "using LinearMaps",
                 "import LinearMaps", "CUDA.", "MtlArray")
    for (root, _, files) in walkdir(joinpath(@__DIR__, "..", "src"))
        for file in files
            endswith(file, ".jl") || continue
            source = read(joinpath(root, file), String)
            for pattern in forbidden
                @test !occursin(pattern, source)
            end
        end
    end
end
