module FunicularLinearMapsExt

using Funicular
using LinearAlgebra
using LinearMaps

# A LinearMap already answers size, eltype, adjoint and mul!, so the operator
# contract holds for one without any adapter. It does not answer the optional
# traits, and both of those can be defined for it.

# When a map has no better way, LinearMaps applies it to a matrix by looping over
# the columns, the same loop panelmul! would otherwise run itself. Letting the
# map do it means that a map that does know how to take several columns at once,
# a WrappedMap around a dense matrix for instance, can do so.
Funicular.panel_capable(::LinearMap) = true

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
