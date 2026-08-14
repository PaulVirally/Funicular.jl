# Reference

Everything Funicular exports. The internal names used elsewhere in these pages
are reached as `Funicular.name` and are listed at the bottom.

## Storage

```@docs
ResidencyPlan
PanelMatrix
GhostPanels
npanels
panelwidth
panelrange
copycols!
```

`Base.copyto!` and `Base.similar` also take panel matrices, as methods on the
existing generics rather than new names. `copyto!(pm, A)` fills a matrix from a
dense one, and `copyto!(dest, src)` copies one panel matrix into another of the
same `N` and `k`, which is how a matrix is cut into different panels. Both are
[`copycols!`](@ref) over every column. `similar(pm)`, `similar(pm, T)` and
`similar(pm, T, dims)` build an uninitialized matrix from `pm`'s plan, with
`pm`'s panel width.

## Sweeps

```@docs
foreachpanel
```

## Panel linear algebra

```@docs
gram
panelmul!
rightmul!
cholqr2!
project
scale!
```

`LinearAlgebra.axpy!`, `dot` and `norm` also take panel matrices. They are
methods on the existing generics rather than new names: `axpy!(α, X, Y)`,
`dot(X, Y)` and `norm(X)`, the last being the Frobenius norm and the only `p`
it accepts.

## Operators

[`check_operator`](@ref) and the three operator traits have a page of their
own: see [The operator contract](operators.md).

## Files

```@docs
save
load
```

## Backends

```@docs
CPUBackend
Funicular.cuda_backend
Funicular.metal_backend
```

## Not exported

| name                          | what it does                                   |
| ----------------------------- | ---------------------------------------------- |
| `Funicular.workspace_bytes`   | operator trait, device memory it holds          |
| `Funicular.panel_capable`     | operator trait, `mul!` takes a matrix           |
| `Funicular.ishermitian_op`    | operator trait, `G == G'`                       |
| `Funicular.linearmap`         | wrap an operator as a `LinearMap`               |
| `Funicular.plan`              | the plan a matrix was built from                |
| `Funicular.prefetch!`         | bring one panel to the host tier                |
| `Funicular.evict!`            | send one panel back to its cold tier            |
| `Funicular.free!`             | release a matrix's blocks and close its file    |
| `Funicular.diskarray`         | read a Funicular file through DiskArrays.jl     |
| `Funicular.disk_reads`        | panels read from a matrix's store               |
| `Funicular.disk_writes`       | panels written to it                            |
| `Funicular.randn_column!`     | the default `GhostPanels` generator             |
