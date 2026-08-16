module FunicularLinearMapsExt

using Funicular
using LinearAlgebra
using LinearMaps

# A LinearMap already answers size, eltype, adjoint and mul!, so the operator
# contract holds for one without any adapter. It does not answer the optional
# traits, and both of those can be defined for it.

# Every LinearMap has an mul!(Y, A, X) method for a matrix X, but it is not
# always a loop over the columns. A CompositeMap (which is what A * B builds)
# materializes an intermediate factor via convert(AbstractArray, ...), and that
# allocates host arrays no matter where the caller's arrays live, so it fails on
# a device backend. As such, only a map that wraps a matrix claims the trait,
# since its mul! is a real GEMM. Every other map falls back to Funicular's
# column loop, which does the same work LinearMaps would have done, but on the
# caller's arrays.
Funicular.panel_capable(::LinearMap) = false
Funicular.panel_capable(A::LinearMaps.WrappedMap) = Funicular.panel_capable(A.lmap)
Funicular.panel_capable(A::Union{LinearMaps.AdjointMap,LinearMaps.TransposeMap}) =
    Funicular.panel_capable(A.lmap)

# Hermitian symmetry is a trait a LinearMap carries and composes, so this reads
# it rather than probing.
Funicular.ishermitian_op(A::LinearMap) = ishermitian(A)

function Funicular.linearmap(G)
    rows, cols = size(G)
    T = eltype(G)
    adj = adjoint(G)
    LinearMap{T}((y, x) -> mul!(y, G, x), (x, y) -> mul!(x, adj, y), Int(rows), Int(cols);
                 ismutating=true, ishermitian=Funicular.ishermitian_op(G))
end

Funicular.linearmap(A::LinearMap) = A

end # module FunicularLinearMapsExt
