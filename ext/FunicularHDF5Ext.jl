module FunicularHDF5Ext

import HDF5

import Funicular: close_panel_store, open_panel_store, panel_store_layout,
                  read_panel_region!, write_panel_region!

using Funicular: storage_eltype_named

# The layout is one file per PanelMatrix, holding one chunked dataset whose chunk
# dimensions (N, w) are aligned to panels. Reads and writes always cover a whole
# panel rather than part of one, so a chunk is exactly the unit of I/O.
const DATASET = "panels"
const FORMAT = 1

struct H5Store
    file::HDF5.File
    dataset::HDF5.Dataset
end

function open_panel_store(path::AbstractString, mode::AbstractString; N::Integer=0, k::Integer=0, w::Integer=0, stored::DataType=ComplexF64, computed::DataType=ComplexF64)
    mode in ("w", "r", "r+") || throw(ArgumentError("a panel store opens in mode \"w\", \"r\" or \"r+\", got $(repr(mode))"))
    file = HDF5.h5open(path, mode)
    try
        mode == "w" || return H5Store(file, open_existing(file, path))
        dataset = HDF5.create_dataset(file, DATASET, HDF5.datatype(stored),
                                      HDF5.dataspace((Int(N), Int(k)));
                                      chunk=(Int(N), min(Int(w), Int(k))))
        attributes = HDF5.attrs(file)
        attributes["funicular_format"] = FORMAT
        attributes["panel_width"] = Int(w)
        attributes["storage_eltype"] = string(stored)
        attributes["compute_eltype"] = string(computed)
        H5Store(file, dataset)
    catch
        close(file)
        rethrow()
    end
end

function open_existing(file::HDF5.File, path::AbstractString)
    haskey(file, DATASET) || throw(ArgumentError("the file at $(repr(path)) has no $(repr(DATASET)) dataset, so it is not one Funicular wrote"))
    dataset = file[DATASET]
    ndims(dataset) == 2 || throw(ArgumentError("the $(repr(DATASET)) dataset in $(repr(path)) is $(ndims(dataset))-dimensional; a PanelMatrix is stored as one N × k dataset"))
    dataset
end

function panel_store_layout(store::H5Store)
    N, k = size(store.dataset)
    attributes = HDF5.attrs(store.file)
    w = haskey(attributes, "panel_width") ? Int(attributes["panel_width"]) : chunk_width(store, k)
    stored = storage_eltype_named(string(eltype(store.dataset)))
    computed = haskey(attributes, "compute_eltype") ? storage_eltype_named(attributes["compute_eltype"]) : stored
    (Int(N), Int(k), clamp(w, 1, Int(k)), stored, computed)
end

# A file written by something other than Funicular still partitions cleanly if it
# is chunked by columns, so the chunk width is used when the panel_width
# attribute is missing.
function chunk_width(store::H5Store, k::Integer)
    properties = HDF5.get_create_properties(store.dataset)
    try
        properties.layout === :chunked || return Int(k)
        Int(properties.chunk[2])
    finally
        close(properties)
    end
end

read_panel_region!(store::H5Store, dst::AbstractMatrix, cols::AbstractUnitRange) = copyto!(dst, store.dataset, :, cols)

function write_panel_region!(store::H5Store, src::AbstractMatrix, cols::AbstractUnitRange)
    store.dataset[:, cols] = src
    src
end

function close_panel_store(store::H5Store)
    close(store.dataset)
    close(store.file)
    nothing
end

end # module FunicularHDF5Ext
