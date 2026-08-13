# When to use this

Funicular is built for one specific situation, and this page is meant to help
you decide whether you are in it. For most tall-skinny work you are not, and
something simpler will serve you better.

## Use it if

  * The matrix is tall and skinny, and the column is the meaningful unit. That
    is, millions of rows and at most a few thousand columns, applied to and
    combined a column at a time, with no single entry ever wanted on its own.
    Randomized range finders, subspace iterations, block Krylov bases and trace
    estimators all produce matrices of this shape.

  * A panel fits on the device but the matrix does not. One column, and in
    practice a block of a few dozen, has room in device memory next to the
    operator and its workspace, while the full basis does not and may not fit in
    host memory either. This is the case Funicular is designed for.

  * Applying the operator costs more than moving a column. Call the time to
    apply your operator to one column `C`, and the time to move that column to
    the device and back `T`. Funicular hides the transfers behind the compute,
    so it helps when `C` is comfortably larger than `T`, and helps more the
    larger the ratio is. The arithmetic is given below.

  * You touch the columns in sweeps. A streaming pipeline can schedule ahead of
    itself when every column is touched in order, over and over. An access
    pattern that jumps around cannot be prefetched and gets nothing from any of
    this.

## Do not use it if

  * Everything already fits in device memory. A device matrix and the vendor's
    BLAS do this well, and Funicular can only add transfers to it: a panel
    matrix keeps its panels in host memory, so a sweep pays an upload and a
    writeback that a resident matrix does not. The pipeline hides all but the
    first upload and the last writeback, and `benchmark/resident.jl` measures
    what is left. In that case, use raw device arrays.

  * A single column does not fit on the device. The operator itself then has to
    be split across devices, and that belongs to a different layer than this
    one. That case is out of scope permanently, though nothing here forecloses
    it: the executor never assumes a device panel is any particular array type,
    only that the vendor API can move it, so a distributed panel type could be
    slotted in later.

  * Your operator is cheap next to the transfers, so you are bandwidth bound.
    Double buffering still overlaps the two directions of the link, but there is
    no compute to hide behind and the run costs what the transfers cost. Measure
    before you commit: `benchmark/overlap.jl` sweeps the ratio and reports what
    the pipeline recovers at each one.

  * You want to index the matrix. There is no `getindex` and there will not be:
    a scalar read of a cold panel is a panel-sized disk read for sixteen bytes,
    and cheap-looking syntax should not hide a cost like that. Use `Matrix(pm)`
    when the whole matrix fits in host memory, `foreachpanel` to get one panel
    at a time on the device, and the panel BLAS functions for everything else.

  * The data is larger than host memory and disk put together. No storage layer
    helps with that, so the problem itself has to be made smaller.

## The arithmetic

In the regime above, the transfers cost essentially nothing. This follows from
one comparison. Moving a column across the host link runs at a few tens of
gigabytes per second when the host memory is page-locked, while device memory
is one to two orders of magnitude faster than that. As such, an operator that
reads and writes its argument even a handful of times inside device memory has
already spent longer than moving that argument would have taken, and one built
on transforms, on products against a factor that stays resident, or on any
iterated kernel spends far more. That is the common case for the operators this
package is for, and it is why the transfers can be made free rather than merely
cheaper.

Write `C` for the time to apply the operator to one panel and `T` for the time
to move that panel up and back. A schedule that uploads panel `j+1` while the
operator works on panel `j`, and writes panel `j-1` back at the same time, costs

```
max(C, T) per panel, instead of C + T
```

so at `C/T = 5` it recovers 83% of the transfer time and at `C/T = 10`, 91%.
What is left over is the first upload and the last writeback, which have nothing
to overlap against. Deeper buffering would hide those too and is not
implemented.

`benchmark/overlap.jl` measures this directly. It calibrates a synthetic panel
function against the measured copy time, sweeps `C/T` through 0.5, 1, 2, 5 and
10, and holds the two-buffer pipeline to `max(C, T) × 1.15`. On the hardware it
has been run on, at ratios of 5 and 10, the pipeline costs the compute plus
about one percent: every transfer but the two at the ends disappears.

## Two ways to lose it

Both are easy to do by accident, and neither one fails a correctness test.

The first is pageable host memory. An asynchronous copy out of memory that is
not page-locked is not asynchronous: the driver stages it through a bounce
buffer of its own and the call blocks. Funicular page-locks the host tier by
construction, and `benchmark/pinned.jl` checks that it still does.

The second is a synchronization inside the loop. One `synchronize()` per panel,
or one hidden inside a stream-ordered allocation, or a vendor library quietly
waiting on a buffer that another stream touched, is enough to serialize the
pipeline while every test still passes. `benchmark/overlap.jl` is the only thing
that notices, and it did catch exactly that during development.

## The unified memory baseline

A GPU can already oversubscribe its memory without any package at all: allocate
the matrix in unified memory and let the driver fault pages in as they are
touched. `benchmark/unified.jl` runs the same work both ways and expects
Funicular to be at least twice as fast, on the grounds that an explicit schedule
that knows what the next panel is should beat demand paging that finds out on a
fault. If it ever stops being twice as fast, the pipeline has a bug.
