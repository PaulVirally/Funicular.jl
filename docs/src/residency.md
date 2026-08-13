# Residency

## The three tiers

A panel is on exactly one tier at a time.

The device tier is where the work happens. Panels are never resident there:
they are staged through buffers the [`ResidencyPlan`](@ref) allocated once and
reuses for every sweep. Nothing inside a sweep loop allocates device memory.

The host tier is page-locked memory the plan owns as slabs, with one block per
panel viewing into a slab. The copies stay asynchronous because the memory is
page-locked. Slabs are allocated on demand and double in size, so a
`host_budget` of hundreds of gigabytes does not cost minutes at startup.

The disk tier is one HDF5 file per matrix, with one dataset chunked at exactly
one panel, so a read or a write always covers a whole chunk rather than a
strided slice of several. It exists when the plan has a `scratch_dir` and
HDF5.jl is loaded. Panels go there when the host tier runs out of room, and come
back on the sweep's producer task a step or two ahead of the pipeline, so the
read overlaps the transfers rather than stalling them.

A fourth kind of panel has no storage at all: [`GhostPanels`](@ref) regenerates
its panels from a seed and the panel index whenever one is asked for, and drops
them rather than writing them when host memory runs short. That is the right
representation for the random test matrix of a randomized method.

## Panel width

`w` is uniform across a matrix, with a ragged last panel when it does not divide
`k`. Left alone, [`ResidencyPlan`](@ref) picks the widest panel whose buffers fit
in `device_budget`, capped at `target_panel_bytes` (2 GiB by default), and then
evens the width out over the panels it implies: a `k = 1000` matrix whose budget
allows `w = 999` is cut into two panels of 500 rather than one of 999 and one
of 1.

Override it with `panel_width` on the plan or `w` on one matrix. Two panel
matrices in one operation must share `(N, k, w)` exactly; there is no
repartitioning.

Wider panels mean fewer, larger transfers and better use of the bus. They also
mean more device memory per staging buffer and a coarser unit of work for the
pipeline to hide. Somewhere between one and four gigabytes per panel is the
range this was designed around.

## Sizing the budgets

`device_budget` has to cover, at once:

  * `nbuffers` staging buffers per panel matrix taking part in a sweep, each
    `N × w` in the compute eltype,
  * a second set of those in the storage eltype when the host tier is narrowed,
  * `workspace_bytes(G)`, the operator's own device memory,
  * the small `k × k` accumulators `gram` and `project` use, and one extra panel
    buffer for `project`.

Take it from the device's total memory minus a reserve rather than from
`CUDA.free_memory()`, which under-reports once the memory pool holds cached
blocks. Ten percent held back is a reasonable reserve. When the budget runs out
the error names the buffer that did not fit, what is already allocated, and the
cap, so read the message rather than guessing.

`host_budget` caps the page-locked tier. With a `scratch_dir` it no longer has
to hold the whole matrix, but it does have to hold what a sweep touches at once:

  * a panel sweep (`panelmul!`, `axpy!`, `scale!`, `foreachpanel`) stages up to
    four steps ahead of the pipeline, one panel per matrix per step, so four
    panels per matrix in the sweep;
  * a row sweep (`gram`, `cholqr2!`) works on blocks of rows that span every
    panel, so it holds every panel of its matrix in host memory at once.

That second one is the real floor: `host_budget` cannot go below one whole
matrix while `gram` and `cholqr2!` traverse rows. A sweep that cannot get host
memory raises an error saying so rather than silently degrading.

## Precision

`PanelMatrix{T}` carries the compute eltype, that is, what device kernels and
the operator see. The cold tiers can be narrower. `host_eltype = ComplexF32`
under a `ComplexF64` compute eltype halves both the host memory and the bytes
crossing the bus, and the conversion is a device kernel fused into the transfer
rather than a separate pass. `disk_eltype` must equal `host_eltype`, since a
panel moves between those two tiers unconverted.

Narrowing is opt-in and never happens by default. Gram matrices, `R` factors and
projections are computed in the compute eltype whatever the storage does.

## Moving panels by hand

Sweeps handle residency themselves. Outside a sweep:

  * `Funicular.prefetch!(pm, j)` brings panel `j` to the host tier.
  * `Funicular.evict!(pm.panels[j], plan)` sends it back to its cold tier,
    writing it out first if it changed. A panel with no colder tier, and no
    ghost home to regenerate from, has nowhere to go, and the call says so.
  * `Funicular.free!(pm)` gives every block back and closes the matrix's scratch
    file. A process that exits without it leaves the file behind.
  * [`save`](@ref) writes a matrix to a named file and [`load`](@ref) opens one
    with its panels still cold.
