using Funicular
using LinearAlgebra
using Random
using Serialization
using Test

using Funicular: DeviceBackend, DiskTier, HostStorage, Panel, PanelSteps,
                 RowSteps, TaskEvent, TaskQueue, alloc_device, alloc_host_slab,
                 alloc_host_storage!, assert_conformal, choose_panel_width,
                 checkout_device_buffers!, checkin_device_buffers!,
                 convert_copy!, device_bytes_allocated, device_bytes_required,
                 devicetype, disk_reads, disk_writes, d2h!, evict!, free!,
                 gram_accumulate!, gram_accumulate_ka!, h2d!, host_bytes_free,
                 host_bytes_in_use, host_bytes_reserved, human_readable_bytes,
                 ishermitian_op, isresident, ka_backend, make_queue,
                 materialize!, materialize_rows!, nsteps, ondisk, panel_bytes,
                 panel_capable, panelstorage, prefetch!, rdiv_upper!,
                 rdiv_upper_ka!, readpanels, record_event, release!,
                 resolve_panel_width, row_block_height, rowtraversal,
                 slab_matrix, stage_channel, stage_host!, storage_eltype,
                 submit!, sweep!, sync_all, sync_event, sync_queue,
                 supports_eltype, todevice, transfers_are_real, updatepanels,
                 wait_event, workspace_bytes, writeback!, writeback_rows!,
                 writepanels

# Neither GPU package is a dependency of this test environment: Metal installs
# on Apple silicon only, and CUDA drags a driver stack onto machines that have
# no NVIDIA card. Either one would break resolution of the suite elsewhere. Each
# has an environment of its own, and running the suite from it covers that
# backend as well as the reference one:
#
#   julia --project=test/metal -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=test/metal --threads=4 test/runtests.jl
#
#   julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=test/cuda --threads=4 test/runtests.jl
const HAS_METAL = Sys.isapple() && Sys.ARCH === :aarch64 && try
    @eval using Metal
    true
catch
    false
end

const HAS_CUDA = !Sys.isapple() && try
    @eval using CUDA
    true
catch
    false
end

# HDF5 and DiskArrays install anywhere the package does, so the disk tier is
# part of `] test` rather than an environment of its own. The suite still has to
# run without them, on a machine with no GPU and no HDF5.
const HAS_HDF5 = try
    @eval using HDF5
    true
catch
    false
end

const HAS_DISKARRAYS = HAS_HDF5 && try
    @eval using DiskArrays
    true
catch
    false
end

const HAS_LINEARMAPS = try
    @eval using LinearMaps
    true
catch
    false
end

HAS_HDF5 || @info "HDF5 is not in this environment; the disk tier tests are skipped."
HAS_LINEARMAPS || @info "LinearMaps is not in this environment; the adapter tests are skipped."

const BACKENDS = DeviceBackend[CPUBackend()]

if HAS_METAL && Metal.functional()
    push!(BACKENDS, Funicular.metal_backend())
end

if HAS_CUDA && CUDA.functional()
    push!(BACKENDS, Funicular.cuda_backend())
end

length(BACKENDS) == 1 && @info "No GPU backend in this environment; running on the CPU reference backend only. See test/setup.jl for how to add one."

available_backends() = BACKENDS

const CURRENT_BACKEND = Ref{DeviceBackend}(first(BACKENDS))

current_backend() = CURRENT_BACKEND[]

function withbackend(f, backend::DeviceBackend)
    previous = CURRENT_BACKEND[]
    CURRENT_BACKEND[] = backend
    try
        f()
    finally
        CURRENT_BACKEND[] = previous
    end
end

backendlabel(backend::DeviceBackend) = String(nameof(typeof(backend)))

# The jitter knob means the same thing on every backend that has one: seconds of
# random delay before each queued operation. A backend whose queues are real
# streams has no such knob, and does not need one, since its schedule already
# varies on its own.
function jittery_backend()
    backend = current_backend()
    :jitter in fieldnames(typeof(backend)) || return backend
    typeof(backend)(jitter=2e-4)
end

# A panel function that throws reaches the caller bare on a backend that runs it
# on the calling task, and wrapped on one whose queues are chains of tasks.
const QUEUE_FAILURE = Union{TaskFailedException,ErrorException}

# A jittered repetition costs a sleep per queued operation on the CPU backend
# and a command buffer as well on a GPU one, so the schedule is stressed hardest
# on the reference backend.
repetitions() = current_backend() isa CPUBackend ? 50 : 5

# Apple GPUs have no double precision, so a backend that refuses ComplexF64 gets
# the single precision pair instead.
testeltypes() = supports_eltype(current_backend(), ComplexF64) ? (ComplexF64, ComplexF32, Float64) : (ComplexF32, Float32)

defaulteltype() = first(testeltypes())

# Stands in for a backend without double precision, like Metal.
struct NarrowBackend <: DeviceBackend end

Funicular.supports_eltype(::NarrowBackend, ::Type{T}) where {T} = !(T <: Union{Float64,ComplexF64})

function testplan(; backend::DeviceBackend=current_backend(), device_budget=64 * 2^20, host_budget=64 * 2^20, kwargs...)
    ResidencyPlan(; backend=backend, device_budget=device_budget, host_budget=host_budget, kwargs...)
end

# A host budget counted in panels rather than in bytes, so a disk tier test can
# say how much of a matrix is allowed in memory at once. Blocks are aligned and
# never straddle a slab, so the budget has to be a whole number of them.
panelbudget(panels::Integer, N::Integer, w::Integer, ::Type{S}) where {S} = panels * cld(N * w * sizeof(S), 64) * 64

randmatrix(::Type{T}, N, k) where {T<:Complex} = rand(T, N, k) .- T(0.5, 0.5)
randmatrix(::Type{T}, N, k) where {T<:Real} = rand(T, N, k) .- T(0.5)

# Device buffers reach a test as whatever array type the backend uses, and the
# reference to compare them against is always on the host.
host(A) = Array(A)

devicebuffer(::Type{T}, dims::Integer...) where {T} = alloc_device(current_backend(), T, dims)

function zerobuffer(::Type{T}, dims::Integer...) where {T}
    buffer = devicebuffer(T, dims...)
    fill!(buffer, zero(T))
end

narrowed(::Type{T}) where {T<:Complex} = Complex{narrowed(real(T))}
narrowed(::Type{Float64}) = Float32
narrowed(::Type{Float32}) = Float16

# HDF5 has no half precision datatype, so a backend whose narrowed eltype is
# Float16 cannot exercise the narrowed cold tiers.
storable(::Type{T}) where {T} = real(T) !== Float16

blastol(::Type{T}, N, k) where {T} = 100 * eps(real(T)) * sqrt(N * k)

orthogonality_bound(::Type{T}, N, k) where {T} = 100 * N * k * eps(real(T))

include("operators.jl")
