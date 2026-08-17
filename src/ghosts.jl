# Panels that are computed instead of stored. A ghost matrix has the same
# partitioning and the same interface as any other PanelMatrix, and its panels
# have a home the way disk-backed ones do; the difference is that the home is a
# generator and a seed, so bringing a panel to the host tier runs a function and
# giving its memory back writes nothing.

"""
    GhostPanels([f!], T, N, k; plan, seed=0, w=nothing) -> PanelMatrix{T}

An `N × k` matrix of compute eltype `T` whose panels are generated on demand and
never stored. Column `col` is filled by `f!(dst, rng, col)`, where `dst` is a
host array holding one whole column, `rng` is a `Random.Xoshiro` seeded from
`seed` and `col` alone, and `col` is the column's index in the full matrix. `f!`
defaults to standard normal entries.

`dst` is host memory and never a device array, and its eltype is the one the
host tier stores: under a narrowing `host_eltype` that is the narrowed type
rather than `T`. It holds all `N` rows of the column, so a generator that
normalizes what it draws needs nothing from the panels around it.

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

A column's stream comes from its index and the seed, so panels regenerate in any
order and one seed means one matrix whatever the panel width. Regeneration is
exact within one Julia version: both `hash` and the `Xoshiro` stream are Julia
implementation details. Two runs on different versions that have to agree need
the matrix written out once with [`save`](@ref) and loaded thereafter.

# Arguments
- `f!`: Generator for one column, called as `f!(dst, rng, col)`
- `N::Integer`: Number of rows
- `k::Integer`: Number of columns
- `plan::ResidencyPlan`: The plan that owns the tiers the panels live on
- `seed=0`: Seed the column's random stream is derived from
- `w=nothing`: Panel width for this matrix, or `nothing` to take the plan's choice from the device budget (see [`ResidencyPlan`](@ref))

```julia
Ω = GhostPanels(ComplexF64, N, k; plan = plan, seed = 0xdeadbeef)
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

GhostPanels(::Type{T}, N::Integer, k::Integer; kwargs...) where {T} = GhostPanels(randn_column!, T, N, k; kwargs...)

function ghost_panel(::Type{T}, source::GhostSource, j::Integer) where {T}
    cols = ((Int(j) - 1) * source.w + 1):min(Int(j) * source.w, source.k)
    Panel{T}(nothing, GhostTier(), GhostHome(source, cols), length(cols), false, 0, 0, nothing)
end

"""
    Funicular.randn_column!(dst, rng, col)

Fills `dst` with standard normal entries. For a complex eltype that means real
and imaginary parts of variance one half each. [`GhostPanels`](@ref) uses this
when no generator is given, and any generator has to have this signature.
"""
randn_column!(dst::AbstractVector, rng::AbstractRNG, ::Integer) = randn!(rng, dst)

isghost(pm::PanelMatrix) = first(pm.panels).home isa GhostHome

function assert_writable(pm::PanelMatrix)
    isghost(pm) || return nothing
    throw(ArgumentError("this is a ghost matrix: its panels are regenerated from a seed rather than stored, so anything written to one would be lost the moment that panel left host memory. Write into a panel matrix of its own instead, which for a $(pm.N)×$(pm.k) ghost that fits in host memory is PanelMatrix(Matrix(ghost); plan = plan, w = $(pm.w))"))
end

# Nothing to write: the panel's contents are a function of the columns it holds,
# so the host block goes straight back to the pool and the next reader
# regenerates it.
cool!(::GhostHome, ::Panel) = nothing

coldtier(::GhostHome) = GhostTier()

# One stream per column, keyed on the column's index in the full matrix and on
# the seed, so the width the columns are cut at cannot reach the values and
# panels can be regenerated in any order. One Xoshiro per column costs nothing
# next to filling the N rows it draws.
function generate_panel!(home::GhostHome, dst::AbstractMatrix)
    source = home.source
    for (c, col) in enumerate(home.cols)
        source.generator(view(dst, :, c), Xoshiro(hash(col, source.seed)), col)
    end
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
