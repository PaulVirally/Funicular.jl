# Test operators. A plain matrix already satisfies the contract and reports
# itself panel capable, so these cover the other cases: an operator that only
# knows one column at a time, a matrix free one, and four that break the
# contract in the ways check_operator has to catch. Each holds its coefficients
# wherever the backend under test wants them, since an operator is applied to
# device resident panels.

using KernelAbstractions

deviceoperator(A::AbstractMatrix) = todevice(current_backend(), Matrix(A))

function devicevector(v::AbstractVector{T}) where {T}
    backend = current_backend()
    device = alloc_device(backend, T, (length(v),))
    h2d!(device, Vector(v), backend)
    device
end

# Formed on the host and sent back. A test operator can take that liberty.
function conjugate_transpose(A::AbstractMatrix)
    dense = Matrix(Matrix(A)')
    A isa Matrix ? dense : copyto!(similar(A, size(dense)), dense)
end

# Dense, but hands out no matrix, so panelmul! has to loop over columns.
struct ColumnOperator{T,M<:AbstractMatrix{T}}
    A::M
end

Base.size(G::ColumnOperator) = size(G.A)
Base.size(G::ColumnOperator, d::Integer) = size(G.A, d)
Base.eltype(::ColumnOperator{T}) where {T} = T
Base.adjoint(G::ColumnOperator) = ColumnOperator(conjugate_transpose(G.A))
LinearAlgebra.mul!(y::AbstractVector, G::ColumnOperator, x::AbstractVector) = mul!(y, G.A, x)

# Circular convolution: matrix free, translation invariant, and its adjoint is
# the conjugate reversed kernel. The Green operator of GilaElectromagnetics has
# this structure, at 3D FFT speed rather than this quadratic kernel.
struct CircularConvolution{T,V<:AbstractVector{T}}
    kernel::Vector{T}
    coefficients::V
    conjugated::Bool
end

CircularConvolution(kernel::Vector) = CircularConvolution(kernel, devicevector(kernel), false)

Base.size(G::CircularConvolution) = (length(G.kernel), length(G.kernel))
Base.size(G::CircularConvolution, d::Integer) = d <= 2 ? length(G.kernel) : 1
Base.eltype(::CircularConvolution{T}) where {T} = T
Base.adjoint(G::CircularConvolution) = CircularConvolution(G.kernel, G.coefficients, !G.conjugated)

coefficient(G::CircularConvolution, i::Int, j::Int) = G.conjugated ? conj(G.kernel[mod(j - i, length(G.kernel)) + 1]) : G.kernel[mod(i - j, length(G.kernel)) + 1]

@kernel function circulant_apply!(y, @Const(c), @Const(x), conjugated)
    i = @index(Global)
    N = length(x)
    acc = zero(eltype(y))
    for j in 1:N
        entry = conjugated ? conj(c[mod(j - i, N) + 1]) : c[mod(i - j, N) + 1]
        acc += entry * x[j]
    end
    y[i] = acc
end

function LinearAlgebra.mul!(y::AbstractVector, G::CircularConvolution, x::AbstractVector)
    kernel = circulant_apply!(get_backend(y))
    kernel(y, G.coefficients, x, G.conjugated; ndrange=length(y))
    y
end

densematrix(G::CircularConvolution) = [coefficient(G, i, j) for i in 1:length(G.kernel), j in 1:length(G.kernel)]

# A circulant's eigenvalues are the DFT of its kernel, so prescribing the
# spectrum means taking one inverse DFT. Written out by hand rather than fetched
# from FFTW, which is not in the test environment.
circulant_kernel(λ::Vector) = [sum(λ[m] * cis(2π * (m - 1) * (l - 1) / length(λ)) for m in eachindex(λ)) / length(λ) for l in eachindex(λ)]

struct TransposedAdjoint{T,M<:AbstractMatrix{T}}
    A::M
end

Base.size(G::TransposedAdjoint) = size(G.A)
Base.size(G::TransposedAdjoint, d::Integer) = size(G.A, d)
Base.eltype(::TransposedAdjoint{T}) where {T} = T
Base.adjoint(G::TransposedAdjoint) = TransposedAdjoint(transposed(G.A))
LinearAlgebra.mul!(y::AbstractVector, G::TransposedAdjoint, x::AbstractVector) = mul!(y, G.A, x)

function transposed(A::AbstractMatrix)
    dense = Matrix(transpose(Matrix(A)))
    A isa Matrix ? dense : copyto!(similar(A, size(dense)), dense)
end

struct MisshapenAdjoint{T,M<:AbstractMatrix{T}}
    A::M
end

Base.size(G::MisshapenAdjoint) = size(G.A)
Base.size(G::MisshapenAdjoint, d::Integer) = size(G.A, d)
Base.eltype(::MisshapenAdjoint{T}) where {T} = T
Base.adjoint(G::MisshapenAdjoint) = MisshapenAdjoint(clipped(G.A))
LinearAlgebra.mul!(y::AbstractVector, G::MisshapenAdjoint, x::AbstractVector) = mul!(y, G.A, x)

function clipped(A::AbstractMatrix)
    dense = Matrix(A)'[:, 1:end - 1]
    A isa Matrix ? Matrix(dense) : copyto!(similar(A, size(dense)), Matrix(dense))
end

# Claims it takes whole panels, then gets the panel case wrong.
struct WrongPanel{T,M<:AbstractMatrix{T}}
    A::M
end

Base.size(G::WrongPanel) = size(G.A)
Base.size(G::WrongPanel, d::Integer) = size(G.A, d)
Base.eltype(::WrongPanel{T}) where {T} = T
Base.adjoint(G::WrongPanel) = WrongPanel(conjugate_transpose(G.A))
Funicular.panel_capable(::WrongPanel) = true
LinearAlgebra.mul!(y::AbstractVector, G::WrongPanel, x::AbstractVector) = mul!(y, G.A, x)
LinearAlgebra.mul!(Y::AbstractMatrix, G::WrongPanel, X::AbstractMatrix) = rmul!(mul!(Y, G.A, X), 2)

# Declares the Hermitian symmetry it does have, so a trait can be watched
# crossing an adapter.
struct DeclaredHermitian{T,M<:AbstractMatrix{T}}
    A::M
end

Base.size(G::DeclaredHermitian) = size(G.A)
Base.size(G::DeclaredHermitian, d::Integer) = size(G.A, d)
Base.eltype(::DeclaredHermitian{T}) where {T} = T
Base.adjoint(G::DeclaredHermitian) = G
Funicular.ishermitian_op(::DeclaredHermitian) = true
LinearAlgebra.mul!(y::AbstractVector, G::DeclaredHermitian, x::AbstractVector) = mul!(y, G.A, x)

# Claims Hermitian symmetry it does not have.
struct FalselyHermitian{T,M<:AbstractMatrix{T}}
    A::M
end

Base.size(G::FalselyHermitian) = size(G.A)
Base.size(G::FalselyHermitian, d::Integer) = size(G.A, d)
Base.eltype(::FalselyHermitian{T}) where {T} = T
Base.adjoint(G::FalselyHermitian) = ColumnOperator(conjugate_transpose(G.A))
Funicular.ishermitian_op(::FalselyHermitian) = true
LinearAlgebra.mul!(y::AbstractVector, G::FalselyHermitian, x::AbstractVector) = mul!(y, G.A, x)

struct Workspaced{T,M<:AbstractMatrix{T}}
    A::M
    bytes::Int
end

Base.size(G::Workspaced) = size(G.A)
Base.size(G::Workspaced, d::Integer) = size(G.A, d)
Base.eltype(::Workspaced{T}) where {T} = T
Base.adjoint(G::Workspaced) = Workspaced(conjugate_transpose(G.A), G.bytes)
Funicular.workspace_bytes(G::Workspaced) = G.bytes
LinearAlgebra.mul!(y::AbstractVector, G::Workspaced, x::AbstractVector) = mul!(y, G.A, x)

# Hermitian positive definite, rank r up to a noise floor of ε.
function lowrank_plus_noise(::Type{T}, N::Integer, r::Integer, ε::Real) where {T}
    B = randmatrix(T, N, r)
    A = B * B'
    A ./= opnorm(A)
    E = randmatrix(T, N, N)
    A .+= ε .* (E .+ E')
    Matrix(Hermitian(A))
end
