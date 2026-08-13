# Porting a randomized method

Randomized range finders, subspace iterations, RSVD and the trace estimators
all do the same four things in a loop: apply the operator to a block of vectors,
orthonormalize, repeat, and finish with a small dense factorization. Each has a
Funicular counterpart, and porting one is mostly a matter of deciding which
matrices become a `PanelMatrix` and which stay a `Matrix`.

## The translation table

| dense code                     | Funicular                       | traversal |
| ------------------------------ | ------------------------------- | --------- |
| `Ω = randn(T, N, k)`           | `GhostPanels(T, N, k; plan, seed)` | none, regenerated |
| `Y = G * X`                    | `panelmul!(Y, G, X)`            | one column-panel sweep |
| `Q, R = qr(Y)`                 | `R = cholqr2!(Y)`, `Y` becomes `Q` | four row-block sweeps |
| `B = X' * Y`                   | `gram(X, Y)`                    | one row-block sweep |
| `S = Q' * G * Q`               | `project(Q, G)`                 | `p (p + 1)` panel uploads |
| `Y .+= α .* X`                 | `axpy!(α, X, Y)`                | one sweep |
| `Y .*= α`                      | `scale!(Y, α)`                  | one sweep |
| `dot(X, Y)`, `norm(X)`         | the same names                  | one sweep |
| anything else                  | `foreachpanel(f!, pm)`          | one sweep |

Everything `k × k` stays an ordinary `Matrix` on the host: `gram` and `project`
return one, `cholqr2!` returns one, and the eigen- or SVD-solve at the end of a
randomized method is a plain LAPACK call.

## Rules of thumb

Count sweeps rather than flops. Once the matrix no longer fits on the device,
the cost of an operation is how many times it moves the matrix past the GPU.
`panelmul!` is one pass, `gram` is one, `cholqr2!` is four, and `project` is
`p + 1` of them for `p` panels. A loop body that was free in the dense version
because it was a broadcast over a resident array becomes a full sweep here, so
fold it into the panel function of a sweep that is happening anyway.

Prefer `panelmul!` then `gram` over `project`. The nested version exists for the
case where you cannot afford a second `N × k` matrix. If you can afford it,
`panelmul!(Z, G, Q)` followed by `gram(Q, Z)` is two sweeps instead of `p + 1`.

Orthonormalize between every application. CholeskyQR2 squares the condition
number before factoring, so it wants `κ(Y) ≲ 1/sqrt(eps)`, about `1e8` in double
precision. Power iteration squares `κ` each time. Past that limit, `cholqr2!`
falls back to a shifted factorization and a third pass, which recovers
orthogonality but costs more than staying below it would have.

Do not reach for indexing. There is no `getindex`. When you want a column, use
either a sweep or, for a matrix that fits in host memory, `Matrix(pm)`, and
which of those it is depends on whether the answer is `k × k` or `N × k`.

Two matrices in one operation must be conformal: same `N`, same `k`, same `w`,
same plan. Build them from one plan and let it choose the width.

## A worked shape

The randomized range finder, as the demo writes it:

```julia
function rangefinder(G, Ω, plan; iterations = 2)
    Y = PanelMatrix{eltype(Ω)}(undef, size(Ω)...; plan = plan)
    Z = PanelMatrix{eltype(Ω)}(undef, size(Ω)...; plan = plan)
    panelmul!(Y, G, Ω)
    cholqr2!(Y)
    for _ in 1:iterations
        panelmul!(Z, G, Y)
        cholqr2!(Z)
        Y, Z = Z, Y
    end
    Y, Z
end
```

The two panel matrices are allocated once and swapped rather than reallocated
each iteration, because allocating a panel matrix means page-locked host memory
and, under a `scratch_dir`, a file, and neither is cheap enough to do in a loop.

The trace estimators port the same way. Their random probes become
`GhostPanels`, their `A * Ω` becomes `panelmul!`, and their traces become `gram`
or `dot`.

## What has to stay outside

Funicular provides storage, movement and panel BLAS. The methods themselves
stay where they are. `cholqr2!` is the single exception, and it lives here
because its second pass is a sweep and it cannot be written efficiently from
outside the executor.
