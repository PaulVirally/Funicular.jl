abstract type Tier end

struct HostTier <: Tier end

struct DiskTier <: Tier end

# A panel whose contents are a function of its index rather than bytes stored
# somewhere. It is colder than the disk tier in the sense that matters to the
# host pool: there is nothing to write when it leaves host memory.
struct GhostTier <: Tier end

struct DeviceTier{B<:DeviceBackend} <: Tier
    backend::B
end

backend(tier::DeviceTier) = tier.backend

Base.show(io::IO, ::HostTier) = print(io, "host")
Base.show(io::IO, ::DiskTier) = print(io, "disk")
Base.show(io::IO, ::GhostTier) = print(io, "ghost")
Base.show(io::IO, tier::DeviceTier) = print(io, "device(", nameof(typeof(tier.backend)), ")")

struct HostBlock
    slab::Int
    offset::Int
    nbytes::Int
end

# The tier handle for a host-resident panel. Holds the panel's own view, the
# slab it aliases (which roots the memory), and the block to give back on
# release.
struct HostStorage{T,O}
    array::Matrix{T}
    slab::HostSlab{O}
    block::HostBlock
end

storage(handle::HostStorage) = handle.array
storage(handle::AbstractArray) = handle

# One file behind one PanelMatrix, holding every panel of it at the columns it
# occupies in the full matrix. `handle` is opened by the HDF5 extension and is
# the only part of a store that src/ does not understand; everything else is
# what the residency layer needs to know without opening the file.
mutable struct PanelStore
    path::String
    handle::Any
    N::Int
    k::Int
    w::Int
    stored::DataType
    computed::DataType
    readonly::Bool
    scratch::Bool
    reads::Int
    writes::Int
end

function Base.show(io::IO, store::PanelStore)
    print(io, "PanelStore(", repr(store.path), ", ", store.N, "×", store.k,
          " ", store.stored, ", w = ", store.w)
    store.readonly && print(io, ", read only")
    store.scratch && print(io, ", scratch")
    print(io, ")")
end

# Where a cold panel lives. The store is shared by every panel of one matrix and
# the column range says which slice of it is this panel's.
struct DiskHome
    store::PanelStore
    cols::UnitRange{Int}
end

# What a ghost matrix is made of: the generator and the seed are shared by every
# panel, and a column's index picks its random stream out of the seed.
struct GhostSource{F}
    generator::F
    seed::UInt64
    N::Int
    k::Int
    w::Int
end

struct GhostHome{F}
    source::GhostSource{F}
    cols::UnitRange{Int}
end

function Base.show(io::IO, source::GhostSource)
    print(io, "GhostSource(", source.generator, ", seed = ", repr(source.seed),
          ", ", source.N, "×", source.k, ", w = ", source.w, ")")
end
