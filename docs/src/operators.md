# The operator contract

`panelmul!` and `project` take any `G` that implements four operations, plus
three more that are optional. There is no abstract type to subtype and nothing
to register. The methods listed below are the whole contract.

## What is required

```julia
Base.size(G)                       # (rows, cols), both positive
Base.size(G, d)
Base.eltype(G)                     # a concrete number type
Base.adjoint(G)                    # an object satisfying this same contract
LinearAlgebra.mul!(y, G, x)        # y and x are device vectors
```

`mul!` is handed device-resident arrays: `CuArray` views under CUDA, `MtlArray`
views under Metal, plain `Array` views on the reference backend. Write it
against `AbstractVector` and let the arrays decide, the way the operators in
`test/operators.jl` do.

Funicular applies an operator to whole columns, so an `m × n` `G` reads columns
of an `n`-row matrix and writes columns of an `m`-row one. `panelmul!` takes any
such shape. `project` is the exception and needs a square `G`, since `Q' G Q`
makes sense only if `G` maps a space into itself.

## What is optional

```@docs
Funicular.workspace_bytes
Funicular.panel_capable
Funicular.ishermitian_op
```

Of the three, `workspace_bytes` is the one worth implementing carefully.
Funicular's budget arithmetic holds that many bytes back from the device pool
before allocating any staging buffer, so an operator that under-reports it turns
an honest error at plan construction into an out-of-memory failure part way
through a run. This matters for an operator holding an FFT plan with a large
work area, or any other factor it keeps resident between applications.

`panel_capable` says `mul!(Y, G, X)` takes a matrix. When it is false, which is
the default, `panelmul!` loops the columns of the resident panel itself. That
costs nothing extra in transfers and only gives up whatever batching the
operator might have done.

## Checking it

```@docs
check_operator
```

Run it once, when you write an operator, against the backend the operator will
be applied on. It probes shape, eltype, adjoint consistency `⟨y, Gx⟩ = ⟨G'y, x⟩`,
and both claims when they are made. It allocates several probe blocks outside
any plan's pool, so it is a diagnostic rather than something to call in a loop.

## LinearMaps.jl

Once LinearMaps.jl is loaded, a `LinearMap` satisfies the contract with no
further work, with `ishermitian_op` reading the map's own trait. Note that
LinearMaps declines to inspect a matrix it wraps, so `LinearMap(A)` is Hermitian
only if you say `LinearMap(A; ishermitian = true)`.

`panel_capable` is true only for a map that wraps a matrix, such as
`LinearMap(A)` and its adjoint, where `mul!(Y, A, X)` is a real GEMM. Other maps
do define a matrix `mul!`, but a `CompositeMap` (which is what `A * B` builds)
implements it by materializing an intermediate factor into host arrays, and that
fails on a device backend. As such, Funicular applies these maps one column at a
time itself, which is the same work the column loop inside LinearMaps would have
done, but on the caller's arrays.

Going the other way:

```@docs
Funicular.linearmap
```

## An example

The [demo](demo.md) defines a Hermitian, panel-capable, matrix-free operator
with an honest `workspace_bytes` in about twenty lines.
