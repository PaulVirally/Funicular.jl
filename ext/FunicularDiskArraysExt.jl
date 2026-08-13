module FunicularDiskArraysExt

import DiskArrays
import HDF5

import Funicular: diskarray

using Funicular: DISK_LOCK

# Interop only: this lets generic chunked-array code read a file Funicular wrote.
# Funicular's own hot path goes through the pipeline and never comes here, so a
# block read is allowed to allocate.
struct PanelDiskArray{T} <: DiskArrays.AbstractDiskArray{T,2}
    file::HDF5.File
    dataset::HDF5.Dataset
    size::Tuple{Int,Int}
    chunk::Tuple{Int,Int}
end

function diskarray(path::AbstractString)
    file = HDF5.h5open(path, "r")
    try
        haskey(file, "panels") || throw(ArgumentError("the file at $(repr(path)) has no \"panels\" dataset, so it is not one Funicular wrote"))
        dataset = file["panels"]
        N, k = size(dataset)
        w = HDF5.attrs(file)["panel_width"]
        PanelDiskArray{eltype(dataset)}(file, dataset, (Int(N), Int(k)), (Int(N), clamp(Int(w), 1, Int(k))))
    catch
        close(file)
        rethrow()
    end
end

Base.size(A::PanelDiskArray) = A.size
Base.close(A::PanelDiskArray) = (close(A.dataset); close(A.file); nothing)

DiskArrays.haschunks(::PanelDiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(A::PanelDiskArray) = DiskArrays.GridChunks(A, A.chunk)

function DiskArrays.readblock!(A::PanelDiskArray, out, r::AbstractUnitRange...)
    block = lock(DISK_LOCK) do
        A.dataset[r...]
    end
    copyto!(out, block)
    out
end

function DiskArrays.writeblock!(A::PanelDiskArray, v, r::AbstractUnitRange...)
    lock(DISK_LOCK) do
        A.dataset[r...] = v
    end
    v
end

end # module FunicularDiskArraysExt
