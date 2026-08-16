# The operator contract. Any G passed to panelmul! or project must provide
# size(G), eltype(G), adjoint(G) and LinearAlgebra.mul!(y, G, x) for device
# vectors x and y. The three traits below are optional, and an operator that
# does not define them gets a default that is safe rather than fast.
# check_operator checks the whole contract at once.

"""
    Funicular.workspace_bytes(G) -> Int

Returns the device memory `G` needs for itself while it is applied, for
instance the work area of an FFT plan. [`ResidencyPlan`](@ref) holds that much
back from the buffer pool, so an operator that under-reports it overflows the
device. Defaults to 0.
"""
workspace_bytes(::Any) = 0

"""
    Funicular.panel_capable(G) -> Bool

Whether `mul!(Y, G, X)` accepts whole panels, that is matrices of several
columns at once, rather than one vector at a time. Defaults to false, which
makes [`panelmul!`](@ref) loop over the columns of the resident panel.
"""
panel_capable(::Any) = false
panel_capable(::AbstractMatrix) = true

"""
    Funicular.ishermitian_op(G) -> Bool

Whether `G` is its own adjoint. Defaults to false, which is always safe.
Funicular does not yet take a shortcut on the strength of it. It is declared so
that callers and [`check_operator`](@ref) can ask, and so that an operator that
answers wrongly is caught before something starts trusting it.
"""
ishermitian_op(::Any) = false
ishermitian_op(G::AbstractMatrix) = ishermitian(G)
# A' is Hermitian exactly when A is, and asking the wrapper reads a device
# matrix one entry at a time.
ishermitian_op(G::Adjoint{<:Any,<:AbstractMatrix}) = ishermitian_op(parent(G))

"""
    Funicular.linearmap(G) -> LinearMap

Wraps an operator that satisfies the Funicular contract as a
`LinearMaps.LinearMap`, so code written against LinearMaps.jl can consume it.
The map applies `G` and its `adjoint`, carries `G`'s size and eltype, and
declares Hermitian symmetry when [`Funicular.ishermitian_op`](@ref) does.

The other direction needs nothing: a `LinearMap` already satisfies the contract,
and loading LinearMaps.jl tells Funicular when one is Hermitian and when one
takes whole panels. Neither crossing carries [`Funicular.workspace_bytes`](@ref),
since a `LinearMap` has no way to hold it, so pass the operator's workspace to
[`ResidencyPlan`](@ref) yourself.

Needs LinearMaps.jl loaded.
"""
function linearmap(args...; kwargs...)
    throw(ArgumentError("Funicular.linearmap arrives with the LinearMaps package, which has to be added to the environment and loaded first"))
end

function operator_size(G)
    dims = size(G)
    dims isa Tuple{Integer,Integer} || throw(ArgumentError("size(G) must return a pair of integers, got $(repr(dims))"))
    all(>(0), dims) || throw(ArgumentError("operator dimensions must be positive, got $dims"))
    Int.(dims)
end

function assert_square_operator(G, N::Integer)
    rows, cols = operator_size(G)
    rows == cols == N || throw(ArgumentError("operator is $(rows)×$(cols) but the panel matrices have $N rows. Funicular applies an operator to whole columns, so it must be square and match N"))
    nothing
end

function assert_operator_shape(G, m::Integer, n::Integer)
    rows, cols = operator_size(G)
    (rows, cols) == (Int(m), Int(n)) || throw(ArgumentError("operator is $(rows)×$(cols) but it reads columns of an $n-row matrix and writes columns of an $m-row one, so it has to be $(m)×$(n). Either the operator or one of the two matrices has the wrong row count"))
    nothing
end

apply_operator!(dst::AbstractMatrix, G, src::AbstractMatrix) = panel_capable(G) ? mul!(dst, G, src) : columnwise_apply!(dst, G, src)

function columnwise_apply!(dst::AbstractMatrix, G, src::AbstractMatrix)
    for c in axes(src, 2)
        mul!(view(dst, :, c), G, view(src, :, c))
    end
    dst
end

"""
    check_operator(G; n=3, backend=CPUBackend(), rtol=nothing) -> true

Holds `G` to the contract [`panelmul!`](@ref) and [`project`](@ref) expect,
using `n` random probe vectors, and throws an `ArgumentError` naming the first
thing that fails. Checks that `size`, `eltype` and `adjoint` answer sensibly,
and that `G` and its adjoint agree: `⟨y, Gx⟩ = ⟨G'y, x⟩`. An operator claiming
`Funicular.panel_capable` is additionally checked to give the same answer for a
whole panel as it does column by column, and one claiming
`Funicular.ishermitian_op` to satisfy `⟨y, Gx⟩ = ⟨Gy, x⟩`.

A rectangular `G` is checked the same way. For an `m × n` operator the probes
`x` have `n` rows and the probes `y` have `m`, so each side of
`⟨y, Gx⟩ = ⟨G'y, x⟩` is an inner product in the space it belongs to, and an
`adjoint` that maps the wrong way round fails the shape check before the probes
run. Only the Hermitian claim needs a square operator, and a rectangular
operator that claims it is refused.

The probes are allocated outside any plan's pool, four blocks of `n` columns
and two more when the panel and Hermitian claims are made, so a check on a full
size operator costs several vectors of device memory.

# Arguments
- `n::Integer=3`: The number of probe vectors
- `backend::DeviceBackend=CPUBackend()`: Where the probes are allocated. Must be the backend the operator expects to be applied on
- `rtol=nothing`: Relative tolerance for the probe comparisons, defaulting to a multiple of `eps` scaled by the operator size
"""
function check_operator(G; n::Integer=3, backend::DeviceBackend=CPUBackend(), rtol=nothing)
    probes = Int(n)
    probes >= 1 || throw(ArgumentError("n must be at least 1, got $n"))
    rows, cols = operator_size(G)
    T = eltype(G)
    T isa DataType && T <: Number && isconcretetype(T) || throw(ArgumentError("eltype(G) must be a concrete number type, got $(repr(T))"))
    supports_eltype(backend, T) || throw(ArgumentError("backend $(nameof(typeof(backend))) cannot compute in the operator's eltype $T, so the probes have nowhere to live. Check the operator against a backend that supports $T, or build it in one the backend does"))
    workspace_bytes(G) >= 0 || throw(ArgumentError("Funicular.workspace_bytes(G) returned $(workspace_bytes(G)), which must be nonnegative"))

    adj = adjoint(G)
    adjdims = operator_size(adj)
    adjdims == (cols, rows) || throw(ArgumentError("adjoint(G) is $(adjdims[1])×$(adjdims[2]) but G is $(rows)×$(cols), so the two cannot be adjoints"))

    tol = rtol === nothing ? 64 * eps(real(float(T))) * sqrt(max(rows, cols)) : rtol
    X = todevice(backend, randprobe(T, cols, probes))
    Y = todevice(backend, randprobe(T, rows, probes))
    GX = alloc_device(backend, T, (rows, probes))
    AY = alloc_device(backend, T, (cols, probes))
    fill!(GX, zero(T))
    fill!(AY, zero(T))
    columnwise_apply!(GX, G, X)
    columnwise_apply!(AY, adj, Y)

    for c in 1:probes
        left = dot(view(Y, :, c), view(GX, :, c))
        right = dot(view(AY, :, c), view(X, :, c))
        isapprox(left, right; rtol=tol, atol=tol * max(abs(left), abs(right), one(real(T)))) ||
            throw(ArgumentError("G and adjoint(G) disagree on probe $c: ⟨y, Gx⟩ = $left but ⟨G'y, x⟩ = $right, a relative difference of $(abs(left - right) / max(abs(left), abs(right))) against a tolerance of $tol. Either adjoint(G) is not the adjoint of G or mul! is wrong for one of them"))
    end

    if panel_capable(G)
        panelled = alloc_device(backend, T, (rows, probes))
        fill!(panelled, zero(T))
        # An operator that claims the trait but cannot actually take a matrix
        # fails somewhere inside its own mul!, so name the claim here instead of
        # propagating whatever it threw.
        try
            mul!(panelled, G, X)
        catch err
            throw(ArgumentError("G claims Funicular.panel_capable but mul!(Y, G, X) on a whole $(cols)×$probes panel threw: $(sprint(showerror, err)). Either define mul! for a matrix right hand side or leave Funicular.panel_capable(G) at its default of false, and Funicular will apply G one column at a time"))
        end
        isapprox(panelled, GX; rtol=tol, atol=tol) ||
            throw(ArgumentError("G claims Funicular.panel_capable but applying it to a whole $(cols)×$probes panel does not match applying it column by column, to a relative tolerance of $tol"))
    end

    if ishermitian_op(G)
        rows == cols || throw(ArgumentError("G claims Funicular.ishermitian_op but is $(rows)×$(cols), which cannot be Hermitian"))
        GY = alloc_device(backend, T, (rows, probes))
        fill!(GY, zero(T))
        columnwise_apply!(GY, G, Y)
        for c in 1:probes
            left = dot(view(Y, :, c), view(GX, :, c))
            right = dot(view(GY, :, c), view(X, :, c))
            isapprox(left, right; rtol=tol, atol=tol * max(abs(left), abs(right), one(real(T)))) ||
                throw(ArgumentError("G claims Funicular.ishermitian_op but ⟨y, Gx⟩ = $left and ⟨Gy, x⟩ = $right differ on probe $c, beyond a tolerance of $tol"))
        end
    end
    true
end

randprobe(::Type{T}, m::Integer, n::Integer) where {T<:Complex} = T.(randn(real(T), m, n), randn(real(T), m, n))
randprobe(::Type{T}, m::Integer, n::Integer) where {T<:Real} = randn(T, m, n)

function todevice(backend::DeviceBackend, A::Matrix{T}) where {T}
    device = alloc_device(backend, T, size(A))
    h2d!(device, A, backend)
    device
end
