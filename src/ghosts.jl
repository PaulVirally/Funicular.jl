# Panels that are computed instead of stored. A ghost matrix has the same
# partitioning and the same interface as any other PanelMatrix, and its panels
# have a home the way disk-backed ones do; the difference is that the home is a
# generator and a seed, so bringing a panel to the host tier runs a function and
# giving its memory back writes nothing.

"""
    GhostPanels([f!], T, N, k; plan, seed=0, w=nothing) -> PanelMatrix{T}

An `N × k` matrix of compute eltype `T` whose panels are generated on demand and
never stored. Panel `j` is filled by `f!(dst, rng, cols)`, where `dst` is the
panel's host buffer, `rng` is a `Random.Xoshiro` seeded from `seed` and `j`
alone, and `cols` are the columns of the full matrix that panel holds. `f!`
defaults to standard normal entries.

The result is an ordinary [`PanelMatrix`](@ref) apart from being read only, so
`panelmul!(Y, G, Ω)`, `gram(Ω)` and `foreachpanel(f, Ω; write=false)` all work
on it. Anything that would write to it, including [`cholqr2!`](@ref), raises an
error: a written panel would be thrown away the moment it left host memory.

This is meant for the test matrix of a randomized method. A `10⁷ × 1000`
Gaussian takes 160 GB to store and nothing at all when it is regenerated
instead. The generation runs on the sweep's producer task, a step or two ahead
of the pipeline, so it overlaps the transfers the way a disk read does. Panels
stay in host memory while there is room for them and are dropped rather than
written when there is not, so a ghost matrix needs no `scratch_dir` and no
`host_budget` beyond what a sweep holds at once.

Regeneration is exact within one Julia version and one panel width. Both `hash`
and the `Xoshiro` stream are Julia implementation details, and the stream is per
panel, so a different `w` cuts the columns differently and gives a different
matrix. Two runs that have to agree need the same Julia version and the same
`w`. Otherwise, write the matrix out once with [`save`](@ref) and load it
thereafter.

# Arguments
- `f!`: Generator for one panel, called as `f!(dst, rng, cols)`
- `N::Integer`: Number of rows
- `k::Integer`: Number of columns
- `plan::ResidencyPlan`: The plan that owns the tiers the panels live on
- `seed=0`: Seed the panel's random stream is derived from
- `w=nothing`: Panel width for this matrix, or `nothing` to take the plan's choice from the device budget (see [`ResidencyPlan`](@ref))

```julia
Ω = GhostPanels(ComplexF64, N, k; plan = plan, seed = 0x5EED)
panelmul!(Y, G, Ω)
```
"""
function GhostPanels(f!, ::Type{T}, N::Integer, k::Integer; plan::ResidencyPlan, seed=0, w=nothing) where {T}
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    isconcretetype(T) || throw(ArgumentError("GhostPanels eltype must be concrete, got $T"))
    width = resolve_panel_width(plan, N, k, T; override=w)
    source = GhostSource(f!, hash(seed), Int(N), Int(k), width)
    panels = [ghost_panel(T, source, j) for j in 1:cld(Int(k), width)]
    PanelMatrix{T,typeof(plan)}(Int(N), Int(k), width, panels, plan, nothing)
end

GhostPanels(::Type{T}, N::Integer, k::Integer; kwargs...) where {T} = GhostPanels(randn_panel!, T, N, k; kwargs...)

function ghost_panel(::Type{T}, source::GhostSource, j::Integer) where {T}
    cols = ((Int(j) - 1) * source.w + 1):min(Int(j) * source.w, source.k)
    Panel{T}(nothing, GhostTier(), GhostHome(source, Int(j), cols), length(cols), false, 0, 0, nothing)
end

"""
    Funicular.randn_panel!(dst, rng, cols)

Fills `dst` with standard normal entries. For a complex eltype that means real
and imaginary parts of variance one half each. [`GhostPanels`](@ref) uses this
when no generator is given, and any generator has to have this signature.
"""
randn_panel!(dst::AbstractMatrix, rng::AbstractRNG, ::AbstractUnitRange) = randn!(rng, dst)

isghost(pm::PanelMatrix) = first(pm.panels).home isa GhostHome

function assert_writable(pm::PanelMatrix)
    isghost(pm) || return nothing
    throw(ArgumentError("this is a ghost matrix: its panels are regenerated from a seed rather than stored, so anything written to one would be lost the moment that panel left host memory. Write into a panel matrix of its own instead, which for a $(pm.N)×$(pm.k) ghost that fits in host memory is PanelMatrix(Matrix(ghost); plan = plan, w = $(pm.w))"))
end

# Nothing to write: the panel's contents are a function of its index, so the
# host block goes straight back to the pool and the next reader regenerates it.
cool!(::GhostHome, ::Panel) = nothing

coldtier(::GhostHome) = GhostTier()

# The stream depends on the panel index and the seed and on nothing else, so a
# sweep is repeatable and panels can be regenerated in any order.
function generate_panel!(home::GhostHome, dst::AbstractMatrix)
    home.source.generator(dst, Xoshiro(hash(home.index, home.source.seed)), home.cols)
    dst
end

function stage_host!(panel::Panel{T}, ::GhostTier, plan::ResidencyPlan) where {T}
    home = panel.home
    source = home.source
    pin!(plan, panel)
    S = host_storage_eltype(plan, T)
    storage = alloc_host_storage!(plan, S, (source.N, length(home.cols)), panel_bytes(source.N, source.w, S))
    try
        generate_panel!(home, storage.array)
    catch
        release_host_block!(plan, storage.block)
        unpin!(plan, panel)
        rethrow()
    end
    panel.data = storage
    panel.tier = HostTier()
    register_resident!(plan, panel)
    panel
end

materialize!(::AbstractMatrix, ::Panel, ::GhostTier, ::ResidencyPlan; kwargs...) = cold_panel_error()
writeback!(::Panel, ::AbstractMatrix, ::GhostTier, ::ResidencyPlan; kwargs...) = cold_panel_error()
materialize!(::AbstractMatrix, ::Panel, ::AbstractUnitRange, ::GhostTier, ::ResidencyPlan; kwargs...) = cold_panel_error()
writeback!(::Panel, ::AbstractMatrix, ::AbstractUnitRange, ::GhostTier, ::ResidencyPlan; kwargs...) = cold_panel_error()
