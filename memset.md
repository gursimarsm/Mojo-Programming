# Fast memset in Mojo

In this tutorial we will implement a memset version optimized for small sizes
using Mojo's autotuning feature.

The idea behind the implementation is based on Nadav Rotem's work [[1](https://github.com/nadavrot/memset_benchmark)], and is also well-described in [[2](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/4f7c3da72d557ed418828823a8e59942859d677f.pdf)].

We briefly summarize the approach below.

## High-level overview

For the best memset performance we want to use the widest possible register
width for the memory access. For instance, if we want to store 19 bytes, we
want to use vector width 16 and use two overlapping stores. To store 9 bytes,
we would want to use two 8-byte stores.

However, before we get to actually doing stores, we need to perform size
checks to make sure that we're in the right range. I.e. we want to use 8
bytes stores for sizes 8-16, 16 bytes stores for sizes 16-32, etc.

The order in which we do the size checks significantly affects performance
and ideally we would like to run as few checks as possible for the sizes
that occur most often. I.e. if most of the sizes we see are 16-32, then we
want to first check if it's within that range before we check if it's in
8-16 or some other range.

This results in a number of different comparison "trees" that can be used to
perform the size checks, and in this tutorial we use Mojo's autotuning to pick
the most optimal one given the distribution of input data.

## Implementation

We will start as we always start - with imports and type aliases.


```mojo
from Assert import assert_param
from Autotune import autotune_fork, search
from DType import DType
from IO import print_no_newline
from List import VariadicList
from Math import min, max
from Pointer import DTypePointer, Pointer
from Time import now
from Memory import memset as stdlib_memset

alias ValueType = UI8
alias BufferPtrType = DTypePointer[DType.ui8]

alias memset_fn_type = fn(BufferPtrType, ValueType, Int) -> None
```

Now let's add some auxiliary functions. We will use them to benchmark various
memset implementations and visualize results.


```mojo
fn measure_time(
    func: memset_fn_type, size: Int, ITERS: Int, SAMPLES: Int
) -> Int:
    alias alloc_size = 1024 * 1024
    let ptr = BufferPtrType.alloc(alloc_size)

    var best = -1
    for sample in range(SAMPLES):
        let tic = now()
        for iter in range(ITERS):
            # Offset pointer to shake up cache a bit
            let offset_ptr = ptr.offset((iter * 128) & 1024)

            # Just in case compiler will try to outsmart us and avoid repeating
            # memset, change the value we're filling with
            let v = ValueType(iter&255)

            # Actually call the memset function
            func(offset_ptr, v.value, size)

        let toc = now()
        if best < 0 or toc - tic < best:
            best = toc - tic

    ptr.free()
    return best

alias MULT = 2_000

fn visualize_result(size: Int, result: Int):
    print_no_newline("Size: ")
    if size < 10:
        print_no_newline(" ")
    print_no_newline(size, "  |")
    for _ in range(result // MULT):
        print_no_newline("*")
    print()


fn benchmark(func: memset_fn_type, title: StringRef):
    print("\n=====================")
    print(title)
    print("---------------------\n")

    alias benchmark_iterations = 30 * MULT
    alias warmup_samples = 10
    alias benchmark_samples = 1000

    # Warmup
    for size in range(35):
        _ = measure_time(
            func, size, benchmark_iterations, warmup_samples
        )

    # Actual run
    for size in range(35):
        let result = measure_time(
            func, size, benchmark_iterations, benchmark_samples
        )

        visualize_result(size, result)
```

### Reproducing results from the paper

Let's implement a memset version from the paper in Mojo and compare it against
the system memset.



```mojo
@always_inline
fn overlapped_store[
    width: Int
](ptr: BufferPtrType, value: ValueType, count: Int):
    let v = SIMD.splat[DType.ui8, width](value)
    ptr.simd_store[width](v)
    ptr.simd_store[width](count - width, v)


fn memset_manual(ptr: BufferPtrType, value: ValueType, count: Int):
    if count < 32:
        if count < 5:
            if count == 0:
                return
            # 0 < count <= 4
            ptr.store(0, value)
            ptr.store(count - 1, value)
            if count <= 2:
                return
            ptr.store(1, value)
            ptr.store(count - 2, value)
            return

        if count <= 16:
            if count >= 8:
                # 8 <= count < 16
                overlapped_store[8](ptr, value, count)
                return
            # 4 < count < 8
            overlapped_store[4](ptr, value, count)
            return

        # 16 <= count < 32
        overlapped_store[16](ptr, value, count)
    else:
        # 32 < count
        memset_system(ptr, value, count)


fn memset_system(ptr: BufferPtrType, value: ValueType, count: Int):
    stdlib_memset(ptr, value.value, count)

```

Let's benchmark our version of memset vs the standard memset.

> _**Note**: We're optimizing memset for tiniest sizes and benchmarking that properly is tricky. The notebook environment makes it even harder, and while we tried our best to tune the notebook to demonstrate the performance difference, it is hard to guarantee that the results will be stable from run to run._


```mojo
benchmark(memset_manual, "Manual memset")
benchmark(memset_system, "System memset")
```

    
    =====================
    Manual memset
    ---------------------
    
    Size:  0   |******************************************
    Size:  1   |******************************************
    Size:  2   |******************************************
    Size:  3   |******************************************
    Size:  4   |******************************************
    Size:  5   |***************************************************
    Size:  6   |***************************************************
    Size:  7   |***************************************************
    Size:  8   |******************************************
    Size:  9   |******************************************
    Size: 10   |******************************************
    Size: 11   |******************************************
    Size: 12   |******************************************
    Size: 13   |******************************************
    Size: 14   |******************************************
    Size: 15   |******************************************
    Size: 16   |******************************************
    Size: 17   |************************************************************
    Size: 18   |************************************************************
    Size: 19   |************************************************************
    Size: 20   |************************************************************
    Size: 21   |************************************************************
    Size: 22   |************************************************************
    Size: 23   |************************************************************
    Size: 24   |************************************************************
    Size: 25   |************************************************************
    Size: 26   |************************************************************
    Size: 27   |************************************************************
    Size: 28   |************************************************************
    Size: 29   |************************************************************
    Size: 30   |************************************************************
    Size: 31   |************************************************************
    Size: 32   |********************************************************************
    Size: 33   |********************************************************************
    Size: 34   |********************************************************************
    
    =====================
    System memset
    ---------------------
    
    Size:  0   |************************************************************
    Size:  1   |************************************************************
    Size:  2   |************************************************************
    Size:  3   |************************************************************
    Size:  4   |************************************************************
    Size:  5   |************************************************************
    Size:  6   |************************************************************
    Size:  7   |************************************************************
    Size:  8   |************************************************************
    Size:  9   |************************************************************
    Size: 10   |************************************************************
    Size: 11   |************************************************************
    Size: 12   |************************************************************
    Size: 13   |************************************************************
    Size: 14   |************************************************************
    Size: 15   |************************************************************
    Size: 16   |************************************************************
    Size: 17   |************************************************************
    Size: 18   |************************************************************
    Size: 19   |************************************************************
    Size: 20   |************************************************************
    Size: 21   |************************************************************
    Size: 22   |************************************************************
    Size: 23   |************************************************************
    Size: 24   |************************************************************
    Size: 25   |************************************************************
    Size: 26   |************************************************************
    Size: 27   |************************************************************
    Size: 28   |************************************************************
    Size: 29   |************************************************************
    Size: 30   |************************************************************
    Size: 31   |************************************************************
    Size: 32   |***************************************************
    Size: 33   |***************************************************
    Size: 34   |***************************************************


### Tweaking the implementation for different sizes

We can see that it's already much faster for small sizes.
That version was specifically optimized for a certain input size distribution,
e.g. we can see that sizes 8-16 and 0-4 work fastest.

But what if in **our use case** the distribution is different? Let's imagine that
in our case the most common sizes are 16-32 - is this version the most optimal
version we can use then? The answer is obviously "no", and we can easily tweak
the implementation to work better for these sizes - we just need to move the
corresponding check closer to the beginning of the function. E.g. like so:


```mojo
fn memset_manual_2(ptr: BufferPtrType, value: ValueType, count: Int):
    if count < 32:
        if count >= 16:
            # 16 <= count < 32
            overlapped_store[16](ptr, value, count)
            return

        if count < 5:
            if count == 0:
                return
            # 0 < count <= 4
            ptr.store(0, value)
            ptr.store(count - 1, value)
            if count <= 2:
                return
            ptr.store(1, value)
            ptr.store(count - 2, value)
            return

        if count >= 8:
            # 8 <= count < 16
            overlapped_store[8](ptr, value, count)
            return
        # 4 < count < 8
        overlapped_store[4](ptr, value, count)

    else:
        # 32 < count
        memset_system(ptr, value, count)
```

Let's check the performance of this version.


```mojo
benchmark(memset_manual_2, "Manual memset v2")
benchmark(memset_system, "Mojo system memset")
```

    
    =====================
    Manual memset v2
    ---------------------
    
    Size:  0   |************************************************************
    Size:  1   |************************************************************
    Size:  2   |************************************************************
    Size:  3   |***************************************************
    Size:  4   |***************************************************
    Size:  5   |************************************************************
    Size:  6   |************************************************************
    Size:  7   |************************************************************
    Size:  8   |***************************************************
    Size:  9   |***************************************************
    Size: 10   |***************************************************
    Size: 11   |***************************************************
    Size: 12   |***************************************************
    Size: 13   |***************************************************
    Size: 14   |***************************************************
    Size: 15   |***************************************************
    Size: 16   |**********************************
    Size: 17   |**********************************
    Size: 18   |**********************************
    Size: 19   |**********************************
    Size: 20   |**********************************
    Size: 21   |**********************************
    Size: 22   |**********************************
    Size: 23   |**********************************
    Size: 24   |**********************************
    Size: 25   |**********************************
    Size: 26   |**********************************
    Size: 27   |**********************************
    Size: 28   |**********************************
    Size: 29   |**********************************
    Size: 30   |**********************************
    Size: 31   |**********************************
    Size: 32   |********************************************************************
    Size: 33   |********************************************************************
    Size: 34   |********************************************************************
    
    =====================
    Mojo system memset
    ---------------------
    
    Size:  0   |************************************************************
    Size:  1   |************************************************************
    Size:  2   |************************************************************
    Size:  3   |************************************************************
    Size:  4   |************************************************************
    Size:  5   |************************************************************
    Size:  6   |************************************************************
    Size:  7   |************************************************************
    Size:  8   |************************************************************
    Size:  9   |************************************************************
    Size: 10   |************************************************************
    Size: 11   |************************************************************
    Size: 12   |************************************************************
    Size: 13   |************************************************************
    Size: 14   |************************************************************
    Size: 15   |************************************************************
    Size: 16   |************************************************************
    Size: 17   |************************************************************
    Size: 18   |************************************************************
    Size: 19   |************************************************************
    Size: 20   |************************************************************
    Size: 21   |************************************************************
    Size: 22   |************************************************************
    Size: 23   |************************************************************
    Size: 24   |************************************************************
    Size: 25   |************************************************************
    Size: 26   |************************************************************
    Size: 27   |************************************************************
    Size: 28   |************************************************************
    Size: 29   |************************************************************
    Size: 30   |************************************************************
    Size: 31   |************************************************************
    Size: 32   |***************************************************
    Size: 33   |***************************************************
    Size: 34   |***************************************************


The performance is now much better on the 16-32 sizes!

The problem is that we had to manually re-write the code. Wouldn't it be nice
if it was done automatically?

In Mojo this is possible (and quite easy) - we can generate multiple
implementations and let the compiler pick the fastest one for us evaluating
them on sizes we want!

### Mojo implementation

Let's dive into that.

The first thing we need to do is to generate all possible candidates. To do
that we will need to iteratively generate size checks to understand what size
for the overlapping store we can use. Once we localize the size interval, we
just call the overlapping store of the corresponding size.

To express this we will implement an adaptive function `memset_impl_layer` two
parameters designating the current interval of possible size values. When we
generate a new size check, we split that interval into two parts and
recursively call the same functions on those two parts. Once we reach the
minimal intervals, we will call the corresponding overlapped_store function.

This first implementation covers minimal interval cases:


```mojo
@adaptive
@always_inline
fn memset_impl_layer[
    lower: Int, upper: Int
](ptr: BufferPtrType, value: ValueType, count: Int):
    @parameter
    if lower == -100 and upper == 0:
        pass
    elif lower == 0 and upper == 4:
        ptr.store(0, value)
        ptr.store(count - 1, value)
        if count <= 2:
            return
        ptr.store(1, value)
        ptr.store(count - 2, value)
    elif lower == 4 and upper == 8:
        overlapped_store[4](ptr, value, count)
    elif lower == 8 and upper == 16:
        overlapped_store[8](ptr, value, count)
    elif lower == 16 and upper == 32:
        overlapped_store[16](ptr, value, count)
    elif lower == 32 and upper == 100:
        memset_system(ptr, value, count)
    else:
        assert_param[False]()
```

Let's now add an implementation for the other case, where we need to generate a
size check.


```mojo
@adaptive
@always_inline
fn memset_impl_layer[
    lower: Int, upper: Int
](ptr: BufferPtrType, value: ValueType, count: Int):
    alias cur: Int
    autotune_fork[Int, 0, 4, 8, 16, 32 -> cur]()

    assert_param[cur > lower]()
    assert_param[cur < upper]()

    if count > cur:
        memset_impl_layer[max(cur, lower), upper](ptr, value, count)
    else:
        memset_impl_layer[lower, min(cur, upper)](ptr, value, count)
```

Here we use `autotune_fork` to generate all possible at that point checks.

We will discard values beyond the current interval, and for the values within
we will recursively call this function on the interval splits.

This is sufficient to generate multiple correct versions of memset, but to
achieve the best performance we need to take into account one more factor: when
we're dealing with such small sizes, even the code location matters a lot. E.g.
if we swap Then and Else branches and invert the condition, we might get a
different performance of the final function.

To account for that, let's add one more implementation of our function, but now
with branches swapped:


```mojo
@adaptive
@always_inline
fn memset_impl_layer[
    lower: Int, upper: Int
](ptr: BufferPtrType, value: ValueType, count: Int):
    alias cur: Int
    autotune_fork[Int, 0, 4, 8, 16, 32 -> cur]()

    assert_param[cur > lower]()
    assert_param[cur < upper]()

    if count <= cur:
        memset_impl_layer[lower, min(cur, upper)](ptr, value, count)
    else:
        memset_impl_layer[max(cur, lower), upper](ptr, value, count)
```

We defined building blocks for our implementation, now we need to add a top
level entry-point that will kick off the recursion we've just defined.

We will simply call our function with [-100,100] interval - -100 and 100 simply
designate that no checks have been performed yet. This interval will be refined
as we generate more and more check until we have enough to emit actual stores.


```mojo
@adaptive
fn memset_autotune_impl(ptr: BufferPtrType, value: ValueType, count: Int):
    memset_impl_layer[-100, 100](ptr, value, count)
```

Ok, we're done with our memset implementation, now we just need to plug it to
autotuning infrastructure to let the Mojo compiler do the search and pick the
best implementation.

To do that, we need to define an evaluator - this is a function that will take
an array of function pointers to all implementations of our function and will
need to return an index of the best candidate.

There are no limitations in how this function can be implemented - it can
return the first or a random candidate, or it can actually benchmark all of
them and pick the fastest - this is what we're going to do for this example.


```mojo
fn memset_evaluator(funcs: Pointer[memset_fn_type], size: Int) -> Int:
    # This size is picked at random, in real code we could use a real size
    # distribution here.
    let size_to_optimize_for = 17
    print(size_to_optimize_for)

    var best_idx: Int = -1
    var best_time: Int = -1

    alias eval_iterations = MULT
    alias eval_samples = 500

    # Find the function that's the fastest on the size we're optimizing for
    for f_idx in range(size):
        let func = funcs.load(f_idx)
        let cur_time = measure_time(
            func, size_to_optimize_for, eval_iterations, eval_samples
        )
        if best_idx < 0:
            best_idx = f_idx
            best_time = cur_time
        if best_time > cur_time:
            best_idx = f_idx
            best_time = cur_time

    return best_idx
```

The evaluator is ready, the last brush stroke is to add a function that will
call the best candidate.

The search will be performed at compile time, and at runtime we will go
directly to the best implementation.


```mojo
fn memset_autotune(ptr: BufferPtrType, value: ValueType, count: Int):
    # Get the set of all candidates
    alias candidates = memset_autotune_impl.__adaptive_set

    # Use the evaluator to select the best candidate.
    alias best_impl: memset_fn_type
    search[memset_fn_type, VariadicList(candidates), memset_evaluator -> best_impl]()

    # Run the best candidate
    return best_impl(ptr, value, count)
```

We are now ready to benchmark our function, let's see how its performance looks!


```mojo
benchmark(memset_manual, "Mojo manual memset")
benchmark(memset_manual_2, "Mojo manual memset v2")
benchmark(memset_system, "Mojo system memset")
benchmark(memset_autotune, "Mojo autotune memset")

```

    
    =====================
    Mojo manual memset
    ---------------------
    
    Size:  0   |******************************************
    Size:  1   |******************************************
    Size:  2   |******************************************
    Size:  3   |******************************************
    Size:  4   |******************************************
    Size:  5   |***************************************************
    Size:  6   |***************************************************
    Size:  7   |***************************************************
    Size:  8   |******************************************
    Size:  9   |******************************************
    Size: 10   |******************************************
    Size: 11   |******************************************
    Size: 12   |******************************************
    Size: 13   |******************************************
    Size: 14   |******************************************
    Size: 15   |******************************************
    Size: 16   |******************************************
    Size: 17   |************************************************************
    Size: 18   |************************************************************
    Size: 19   |************************************************************
    Size: 20   |************************************************************
    Size: 21   |************************************************************
    Size: 22   |************************************************************
    Size: 23   |************************************************************
    Size: 24   |************************************************************
    Size: 25   |************************************************************
    Size: 26   |************************************************************
    Size: 27   |************************************************************
    Size: 28   |************************************************************
    Size: 29   |************************************************************
    Size: 30   |************************************************************
    Size: 31   |************************************************************
    Size: 32   |********************************************************************
    Size: 33   |********************************************************************
    Size: 34   |********************************************************************
    
    =====================
    Mojo manual memset v2
    ---------------------
    
    Size:  0   |************************************************************
    Size:  1   |************************************************************
    Size:  2   |************************************************************
    Size:  3   |***************************************************
    Size:  4   |***************************************************
    Size:  5   |************************************************************
    Size:  6   |************************************************************
    Size:  7   |************************************************************
    Size:  8   |***************************************************
    Size:  9   |***************************************************
    Size: 10   |***************************************************
    Size: 11   |***************************************************
    Size: 12   |***************************************************
    Size: 13   |***************************************************
    Size: 14   |***************************************************
    Size: 15   |***************************************************
    Size: 16   |**********************************
    Size: 17   |**********************************
    Size: 18   |**********************************
    Size: 19   |**********************************
    Size: 20   |**********************************
    Size: 21   |**********************************
    Size: 22   |**********************************
    Size: 23   |**********************************
    Size: 24   |**********************************
    Size: 25   |**********************************
    Size: 26   |**********************************
    Size: 27   |**********************************
    Size: 28   |**********************************
    Size: 29   |**********************************
    Size: 30   |**********************************
    Size: 31   |**********************************
    Size: 32   |********************************************************************
    Size: 33   |********************************************************************
    Size: 34   |********************************************************************
    
    =====================
    Mojo system memset
    ---------------------
    
    Size:  0   |************************************************************
    Size:  1   |************************************************************
    Size:  2   |************************************************************
    Size:  3   |************************************************************
    Size:  4   |************************************************************
    Size:  5   |************************************************************
    Size:  6   |************************************************************
    Size:  7   |************************************************************
    Size:  8   |************************************************************
    Size:  9   |************************************************************
    Size: 10   |************************************************************
    Size: 11   |************************************************************
    Size: 12   |************************************************************
    Size: 13   |************************************************************
    Size: 14   |************************************************************
    Size: 15   |************************************************************
    Size: 16   |************************************************************
    Size: 17   |************************************************************
    Size: 18   |************************************************************
    Size: 19   |************************************************************
    Size: 20   |************************************************************
    Size: 21   |************************************************************
    Size: 22   |************************************************************
    Size: 23   |************************************************************
    Size: 24   |************************************************************
    Size: 25   |************************************************************
    Size: 26   |************************************************************
    Size: 27   |************************************************************
    Size: 28   |************************************************************
    Size: 29   |************************************************************
    Size: 30   |************************************************************
    Size: 31   |************************************************************
    Size: 32   |***************************************************
    Size: 33   |***************************************************
    Size: 34   |***************************************************
    
    =====================
    Mojo autotune memset
    ---------------------
    
    Size:  0   |************************************************************
    Size:  1   |********************************************************************
    Size:  2   |********************************************************************
    Size:  3   |************************************************************
    Size:  4   |************************************************************
    Size:  5   |***************************************************
    Size:  6   |***************************************************
    Size:  7   |***************************************************
    Size:  8   |***************************************************
    Size:  9   |***************************************************
    Size: 10   |***************************************************
    Size: 11   |***************************************************
    Size: 12   |***************************************************
    Size: 13   |***************************************************
    Size: 14   |***************************************************
    Size: 15   |***************************************************
    Size: 16   |***************************************************
    Size: 17   |**********************************
    Size: 18   |**********************************
    Size: 19   |**********************************
    Size: 20   |**********************************
    Size: 21   |**********************************
    Size: 22   |**********************************
    Size: 23   |**********************************
    Size: 24   |**********************************
    Size: 25   |**********************************
    Size: 26   |**********************************
    Size: 27   |**********************************
    Size: 28   |**********************************
    Size: 29   |**********************************
    Size: 30   |**********************************
    Size: 31   |**********************************
    Size: 32   |**********************************
    Size: 33   |********************************************************************
    Size: 34   |********************************************************************



```mojo

```
