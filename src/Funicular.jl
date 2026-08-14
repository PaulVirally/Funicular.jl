module Funicular

using KernelAbstractions
using LinearAlgebra
using Random

include("device_api.jl")
include("kernels.jl")
include("tiers.jl")
include("plan.jl")
include("panels.jl")
include("ghosts.jl")
include("io.jl")
include("executor.jl")
include("operator.jl")
include("panelblas.jl")

export CPUBackend, ResidencyPlan, PanelMatrix, GhostPanels, npanels, panelwidth,
       panelrange, copycols!, foreachpanel, check_operator, gram, panelmul!,
       rightmul!, cholqr2!, project, scale!, save, load

end # module Funicular
