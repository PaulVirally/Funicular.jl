# Funicular.jl

Tiered-residency storage and a streaming pipeline for tall-skinny complex
matrices, plus the panel-granular linear algebra that matrix-free randomized
methods need on top of it.

A randomized method applied to a large matrix-free operator keeps a basis of
`N × k`, with `N` in the tens of millions and `k` in the hundreds or thousands.
At `N = 10⁷` one column is 160 MB in ComplexF64, so a thousand of them are
160 GB, past any single GPU and often past the node's RAM as well. One column,
or a panel of a few dozen, does fit on the GPU. As such, the matrix is cut into
column panels, the panels live on disk or in page-locked host memory, and they
stream through the GPU behind the compute that is already running there.

Applying such an operator to a column typically costs several times what moving
that column across the host link costs, so with two staging buffers and a
schedule that keeps both of them busy the transfers cost essentially nothing.
Providing that schedule, and keeping it correct, is the job of this package.
[When to use this](when-to-use.md) discusses whether your problem is of that
kind.

## Installing it

```julia
using Pkg
Pkg.add(url = "https://github.com/PaulVirally/Funicular.jl")
```

`Funicular` itself depends only on the standard library and
KernelAbstractions.jl. Everything else arrives with a package you load:

| load this      | and you get                                              |
| -------------- | -------------------------------------------------------- |
| CUDA.jl        | the real device backend: pinned host memory, streams, the pipeline |
| Metal.jl       | an Apple GPU backend for correctness work, ComplexF32 only |
| HDF5.jl        | the disk tier, `save` and `load`                          |
| DiskArrays.jl  | reading a Funicular file from generic chunked-array code  |
| LinearMaps.jl  | the [operator](operators.md) adapter, both directions     |

The CPU reference backend needs nothing loaded at all. It runs the whole logic
layer on plain arrays, and the test suite exercises it everywhere.

## A first look

```julia
using CUDA, Funicular, HDF5, LinearAlgebra

backend = Funicular.cuda_backend()
plan = ResidencyPlan(backend = backend,
                     device_budget = 0.7 * CUDA.total_memory(),
                     host_budget = 200 * 2^30,
                     scratch_dir = "/scratch/$(ENV["USER"])/funicular")

Ω = GhostPanels(ComplexF64, N, k; plan = plan, seed = 0xdeadbeef)
Y = PanelMatrix{ComplexF64}(undef, N, k; plan = plan)

panelmul!(Y, G, Ω)      # stream Ω up, apply G, stream Y down
R = cholqr2!(Y)         # orthonormalize in place, Y becomes Q
S = project(Y, G)       # the k × k projected operator Q' G Q
```

[The demo](demo.md) is a complete runnable version of this, with a known
spectrum that it checks itself against.

## What it is not

A `PanelMatrix` is not an `AbstractMatrix` and has no `getindex`. A scalar read
of a cold panel costs a panel-sized disk read for sixteen bytes, and
cheap-looking syntax should not hide a cost like that. Use `Matrix(pm)` when the
whole matrix fits in host memory, `foreachpanel` to get one panel at a time on
the device, and the operations on this page for everything else.

Funicular is also not a chunked-array standard, not a linear operator standard,
and not a solver library. It stores, moves, and multiplies panels. The
randomized algorithms that use it live elsewhere.

## Where to go next

  * [When to use this](when-to-use.md), for deciding whether the package fits
    your problem, and the arithmetic behind why it can help.
  * [Residency](residency.md), for panel width, the three tiers, and how to size
    the budgets.
  * [Operators](operators.md), for what `G` has to provide.
  * [The demo](demo.md), for a complete program.
  * [Porting a randomized method](porting.md), for turning an existing solver
    into sweeps.
  * [Reference](reference.md), for the exported names.
