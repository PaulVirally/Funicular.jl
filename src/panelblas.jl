function shared_plan(X::PanelMatrix, Y::PanelMatrix)
    assert_conformal(X, Y)
    eltype(X) === eltype(Y) || throw(ArgumentError("panel matrices in one operation must share their compute eltype, got $(eltype(X)) and $(eltype(Y)). Narrowing belongs on the storage tiers, not here"))
    X.plan === Y.plan || throw(ArgumentError("panel matrices in one operation must belong to the same ResidencyPlan, since the plan is what owns the staging buffers the operation runs through. Build both matrices from one plan"))
    X.plan
end

"""
    gram(X[, Y]; nbuffers=nothing) -> Matrix

Computes the `k × k` matrix `X' * Y`, accumulating on the device one row block
at a time and copying the result back to the host once at the end. `gram(X)` is
`gram(X, X)` and costs one upload instead of two.

The sweep goes over blocks of rows rather than over column panels, since a
column panel of `X` only meets the matching columns of `Y` and that is not
enough for `X' Y`. One row block spans every panel, so the whole matrix moves
exactly once, but every panel has to be host resident at the same time.

The sum accumulates in the compute eltype of `X`. Narrowing storage through
[`ResidencyPlan`](@ref) does not narrow it, since that keyword sets the eltype
of the host and disk tiers and leaves the compute eltype alone.

With `X === Y` the result is Hermitian only up to rounding, because the two
triangles are computed independently rather than one mirrored onto the other.
Wrap it in `Hermitian` if you need the symmetry exactly.

# Arguments
- `nbuffers=nothing`: Staging buffers per operand, defaulting to the plan's `nbuffers`
"""
function gram(X::PanelMatrix, Y::PanelMatrix=X; nbuffers=nothing)
    plan = shared_plan(X, Y)
    T = eltype(X)
    accumulator = only(checkout_device_buffers!(plan, T, (X.k, X.k), 1))
    try
        traversal = rowtraversal(X)
        if X === Y
            sweep!(readpanels(X); nbuffers=nbuffers, traversal=traversal) do block, Xb
                gram_accumulate!(plan.backend, accumulator, Xb, Xb, one(T), carry(T, block))
            end
        else
            sweep!(readpanels(X), readpanels(Y); nbuffers=nbuffers, traversal=traversal) do block, Xb, Yb
                gram_accumulate!(plan.backend, accumulator, Xb, Yb, one(T), carry(T, block))
            end
        end
        collect_small(accumulator, plan)
    finally
        checkin_device_buffers!(plan, (accumulator,))
    end
end

# The first step of a sweep overwrites the accumulator and later steps add to
# it, so it never has to be zeroed beforehand. Zeroing would be a device write
# with no sweep to order it against, so on a real backend it would race the
# first step.
carry(::Type{T}, step::Integer) where {T} = step == 1 ? zero(T) : one(T)

function collect_small(device::AbstractMatrix{T}, plan::ResidencyPlan) where {T}
    host = Matrix{T}(undef, size(device))
    d2h!(host, device, plan.backend)
    host
end

"""
    panelmul!(Y, G, X; nbuffers=nothing) -> Y

Applies the operator `G` to every column of `X`, writing the result into the
matching column of `Y`. Panels of `X` stream up to the device and panels of `Y`
stream back down, both overlapped with the operator's work on the panel in
between.

`G` must be square and match the row count of `X` and `Y`, and must satisfy the
operator contract: run [`check_operator`](@ref) on it once. An operator that
declares `Funicular.panel_capable` is handed the whole resident panel, otherwise
`Y`'s columns are filled one `mul!` at a time from the resident panel.

# Arguments
- `nbuffers=nothing`: Staging buffers per operand, defaulting to the plan's `nbuffers`
"""
function panelmul!(Y::PanelMatrix, G, X::PanelMatrix; nbuffers=nothing)
    shared_plan(X, Y)
    X === Y && throw(ArgumentError("panelmul! cannot write its output over its input: the operator is applied panel by panel and would read columns it has already overwritten"))
    assert_square_operator(G, X.N)
    sweep!(readpanels(X), writepanels(Y); nbuffers=nbuffers) do _, Xb, Yb
        apply_operator!(Yb, G, Xb)
    end
    Y
end

"""
    cholqr2!(Y; nbuffers=nothing) -> R

Orthonormalizes the columns of `Y` in place by CholeskyQR2, returning the
`k × k` upper triangular `R` with `Q * R` equal to the original `Y`, where `Q`
is what `Y` now holds.

Each of the two passes forms the Gram matrix `Y' Y`, takes its Cholesky factor
on the host, and divides `Y` by that factor with a triangular solve over row
blocks. Two passes cost four sweeps of the whole matrix.

CholeskyQR2 squares the condition number before factoring, so `Y` must satisfy
`κ(Y) ≲ eps(real(T))^(-1/2)`, about `1e8` in double precision, for the Cholesky
to succeed outright. Past that the Gram matrix loses positive definiteness and
the pass falls back to factoring `G + s I` with
`s = 11 N k eps(real(T)) ‖G‖`, and a third pass is then needed to recover
orthogonality. That is the shifted CholeskyQR3 of Fukaya et al., and it holds
up to `κ(Y)` near `1/eps`. Orthonormalizing between power iterations, rather
than after several of them, keeps you out of this regime.

An indefinite Gram matrix raises an error, since the shift cannot rescue it. A
merely rank deficient one does survive the shift, and the columns of `Y` that
were dependent come back as columns of `Q` that are not meaningfully
orthonormal. Check `norm(gram(Q) - I)` if you cannot rule that out.

# Arguments
- `nbuffers=nothing`: Staging buffers per operand, defaulting to the plan's `nbuffers`
"""
function cholqr2!(Y::PanelMatrix; nbuffers=nothing)
    R, shifted = cholqr_pass!(Y, nbuffers)
    R2, shifted2 = cholqr_pass!(Y, nbuffers)
    R = R2 * R
    if shifted || shifted2
        R3, _ = cholqr_pass!(Y, nbuffers)
        R = R3 * R
    end
    R
end

function cholqr_pass!(Y::PanelMatrix, nbuffers)
    G = gram(Y; nbuffers=nbuffers)
    R, shifted = cholesky_upper(G, Y.N, Y.k)
    rdiv_rows!(Y, R; nbuffers=nbuffers)
    R, shifted
end

function cholesky_upper(G::Matrix{T}, N::Integer, k::Integer) where {T}
    plain = cholesky(Hermitian(G); check=false)
    issuccess(plain) && return Matrix(plain.U), false
    shift = 11 * N * k * eps(real(T)) * norm(G)
    shifted = cholesky(Hermitian(G + shift * I); check=false)
    issuccess(shifted) || throw(ArgumentError("the $(k)×$(k) Gram matrix is not positive definite even after a shift of $shift, so the columns are numerically dependent: a zero column, a duplicated one, or a condition number past $(1 / eps(real(T))). Drop the dependent columns before orthonormalizing"))
    Matrix(shifted.U), true
end

function rdiv_rows!(Y::PanelMatrix{T}, R::Matrix{T}; nbuffers=nothing) where {T}
    plan = Y.plan
    factor = only(checkout_device_buffers!(plan, T, size(R), 1))
    try
        h2d!(factor, R, plan.backend)
        sweep!(updatepanels(Y); nbuffers=nbuffers, traversal=rowtraversal(Y)) do _, Yb
            rdiv_upper!(plan.backend, Yb, factor)
        end
    finally
        checkin_device_buffers!(plan, (factor,))
    end
    Y
end

"""
    project(Q, G; nbuffers=nothing) -> Matrix

Forms the `k × k` projected operator `Q' * G * Q` without ever storing `G Q`.

This is the one operation here that does not move the matrix a constant number
of times. For each panel `j` it applies `G` to that panel and then sweeps all of
`Q` against the result to fill one block column of the answer, so it uploads
`p (p + 1)` panels for `p = npanels(Q)` while applying `G` only `p` times. When
you can afford another `N × k` matrix, `panelmul!(Z, G, Q)` followed by
`gram(Q, Z)` gets the same answer in two sweeps instead of `p + 1` of them.

Needs one device panel buffer beyond the sweep's own, to hold `G Q_j` while the
rest of `Q` streams past it.

# Arguments
- `nbuffers=nothing`: Staging buffers per operand, defaulting to the plan's `nbuffers`
"""
function project(Q::PanelMatrix{T}, G; nbuffers=nothing) where {T}
    plan = Q.plan
    assert_square_operator(G, Q.N)
    p = npanels(Q)
    panels = Int[]
    labels = Tuple{Bool,Int,Int}[]
    for j in 1:p
        push!(panels, j)
        push!(labels, (true, j, j))
        for i in 1:p
            push!(panels, i)
            push!(labels, (false, i, j))
        end
    end
    image = only(checkout_device_buffers!(plan, T, (Q.N, Q.w), 1))
    accumulator = only(checkout_device_buffers!(plan, T, (Q.k, Q.k), 1))
    try
        # Every block of the answer is assigned by exactly one step, so the
        # accumulator needs no zeroing.
        sweep!(readpanels(Q); nbuffers=nbuffers, traversal=PanelSteps(panels, labels)) do label, Qb
            applying, i, j = label
            GQj = view(image, :, 1:panelwidth(Q, j))
            if applying
                apply_operator!(GQj, G, Qb)
            else
                block = view(accumulator, panelrange(Q, i), panelrange(Q, j))
                gram_accumulate!(plan.backend, block, Qb, GQj, one(T), zero(T))
            end
        end
        collect_small(accumulator, plan)
    finally
        checkin_device_buffers!(plan, (image, accumulator))
    end
end

"""
    scale!(Y, α; nbuffers=nothing) -> Y

Multiplies every entry of `Y` by `α` in place, in one sweep.

# Arguments
- `nbuffers=nothing`: Staging buffers per operand, defaulting to the plan's `nbuffers`
"""
function scale!(Y::PanelMatrix, α::Number; nbuffers=nothing)
    sweep!(updatepanels(Y); nbuffers=nbuffers) do _, Yb
        rmul!(Yb, α)
    end
    Y
end

# axpy!, dot and norm extend LinearAlgebra rather than taking names of their
# own. Each is one sweep over column panels. norm is the Frobenius norm and is
# the plain square root of a sum of squares, without the rescaling that
# LinearAlgebra does for arrays, so entries near the square root of floatmax
# overflow it.
function LinearAlgebra.axpy!(α::Number, X::PanelMatrix, Y::PanelMatrix; nbuffers=nothing)
    shared_plan(X, Y)
    X === Y && throw(ArgumentError("axpy!(α, X, X) doubles a matrix against itself. Use scale!(X, 1 + α)"))
    sweep!(readpanels(X), updatepanels(Y); nbuffers=nbuffers) do _, Xb, Yb
        axpy!(α, Xb, Yb)
    end
    Y
end

function LinearAlgebra.dot(X::PanelMatrix, Y::PanelMatrix; nbuffers=nothing)
    plan = shared_plan(X, Y)
    T = eltype(X)
    accumulator = only(checkout_device_buffers!(plan, T, (X.N, X.w), 1))
    try
        if X === Y
            sweep!(readpanels(X); nbuffers=nbuffers) do j, Xb
                block = view(accumulator, :, axes(Xb, 2))
                j == 1 ? block .= conj.(Xb) .* Xb : block .+= conj.(Xb) .* Xb
            end
        else
            sweep!(readpanels(X), readpanels(Y); nbuffers=nbuffers) do j, Xb, Yb
                block = view(accumulator, :, axes(Xb, 2))
                j == 1 ? block .= conj.(Xb) .* Yb : block .+= conj.(Xb) .* Yb
            end
        end
        sum(collect_small(accumulator, plan))
    finally
        checkin_device_buffers!(plan, (accumulator,))
    end
end

function LinearAlgebra.norm(X::PanelMatrix, p::Real=2; nbuffers=nothing)
    p == 2 || throw(ArgumentError("a PanelMatrix has only its Frobenius norm, p = 2, got p = $p"))
    plan = X.plan
    R = real(eltype(X))
    accumulator = only(checkout_device_buffers!(plan, R, (X.N, X.w), 1))
    try
        sweep!(readpanels(X); nbuffers=nbuffers) do j, Xb
            block = view(accumulator, :, axes(Xb, 2))
            j == 1 ? block .= abs2.(Xb) : block .+= abs2.(Xb)
        end
        sqrt(sum(collect_small(accumulator, plan)))
    finally
        checkin_device_buffers!(plan, (accumulator,))
    end
end
