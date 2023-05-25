# Matrix multiplication in Mojo

This notebook describes how to write a matrix multiplication (matmul) algorithm in Mojo. We will start with a pure Python implementation, transition to a naive implementation that is essentially a copy of the Python one, then add types, then continue the optimizations by vectorizing, tiling, and parallelizing the implementation.

First, let's define matrix multiplication. Given two dense matrices $A$ and $B$ of dimensions $M\times K$ and $K\times N$ respectively, we want to compute their dot product $C = A . B$ (also known as matmul). The dot product $C += A . B$ is defined by

$$C_{i,j} += \sum_{k \in [0 \cdots K)} A_{i,k} B_{k,j}$$

> Please take look at our [blog](https://www.modular.com/blog/ais-compute-fragmentation-what-matrix-multiplication-teaches-us) post on matmul and why it is important for ML and DL workloads.

The format of this notebook is to start with an implementation which is identical to that of Python (effectively renaming the file extension), then look at how adding types to the implementation helps performance before extending the implementation by leveraging the vectorization and parallelization capabilities available on modern hardware. Throughout the execution, we report the GFlops achieved.

<div class="alert alert-block alert-info">
<b>Note:</b> Mojo Playground is designed only for testing the Mojo language.
The cloud environment is not always stable and performance varies, so it is not
an appropriate environment for performance benchmarking. However, we believe it
can still demonstrate the magnitude of performance gains provided by Mojo. For
more information about the compute power in the Mojo Playground, see the <a
href="https://docs.modular.com/mojo/faq.html#mojo-playground">Mojo FAQ</a>.
</div>

## Python Implementation

Let's first implement matmul in Python directly from the definition.


```mojo
%%python
def matmul_python(C, A, B):
    for m in range(C.rows):
        for n in range(C.cols):
            for k in range(A.cols):
                C[m, n] += A[m, k] * B[k, n]
```

Let's benchmark our implementation using 128 by 128 square matrices and report the achieved GFLops.


```mojo
%%python
import numpy as np
from timeit import timeit

class Matrix:
    def __init__(self, value, rows, cols):
        self.value = value
        self.rows = rows
        self.cols = cols
        
    def __getitem__(self, idxs):
        return self.value[idxs[0]][idxs[1]]
    
    def __setitem__(self, idxs, value):
        self.value[idxs[0]][idxs[1]] = value

def benchmark_matmul_python(M, N, K):
    A = Matrix(list(np.random.rand(M, K)), M, K)
    B = Matrix(list(np.random.rand(K, N)), K, N)
    C = Matrix(list(np.zeros((M, N))), M, N)
    secs = timeit(lambda: matmul_python(C, A, B), number=2)/2
    gflops = ((2*M*N*K)/secs) / 1e9
    print(gflops, "GFLOP/s")
    return gflops
```


```mojo
python_gflops = benchmark_matmul_python(128, 128, 128).to_f64()
```

    0.0016717199881536883 GFLOP/s


## Importing the Python implementation to Mojo

Using Mojo is as simple as Python. First, let's include that modules from the Mojo stdlib that we are going to use:


```mojo
#|code-fold: true
#|code-summary: "Import utilities and define `Matrix` (click to show/hide)"

from Benchmark import Benchmark
from DType import DType
from Intrinsics import strided_load
from List import VariadicList
from Math import div_ceil, min
from Memory import memset_zero
from Object import object, Attr
from Pointer import DTypePointer
from Random import rand, random_f64
from TargetInfo import dtype_sizeof, dtype_simd_width
```

Then, we can copy and paste our Python code. Mojo is a superset of Python, so the same Python code will run as Mojo code


```mojo
# This exactly the same Python implementation, 
# but is infact Mojo code!
def matmul_untyped(C, A, B):
    for m in range(C.rows):
        for n in range(C.cols):
            for k in range(A.cols):
                C[m, n] += A[m, k] * B[k, n]
```

We can then benchmark the implementation. As before we use a 128 by 128 matrix


```mojo
def matrix_getitem(self, i) -> object:
    return self.value[i]


def matrix_setitem(self, i, value) -> object:
    self.value[i] = value
    return None


def matrix_append(self, value) -> object:
    self.value.append(value)
    return None


def matrix_init(rows: Int, cols: Int) -> object:
    value = object([])
    return object(
        Attr("value", value), Attr("__getitem__", matrix_getitem), Attr("__setitem__", matrix_setitem), 
        Attr("rows", rows), Attr("cols", cols), Attr("append", matrix_append),
    )

def benchmark_matmul_untyped(M: Int, N: Int, K: Int, python_gflops: F64):
    C = matrix_init(M, N)
    A = matrix_init(M, K)
    B = matrix_init(K, N)
    for i in range(M):
        c_row = object([])
        b_row = object([])
        a_row = object([])
        for j in range(N):
            c_row.append(0.0)
            b_row.append(random_f64(-5, 5))
            a_row.append(random_f64(-5, 5))
        C.append(c_row)
        B.append(b_row)
        A.append(a_row)

    @parameter
    fn test_fn():
        try:
            _ = matmul_untyped(C, A, B)
        except:
            pass

    let secs = F64(Benchmark().run[test_fn]()) / 1_000_000_000
    let gflops = ((2*M*N*K)/secs) / 1e9
    let speedup : F64 = gflops / python_gflops
    print(gflops, "GFLOP/s, a", speedup.value, "x speedup over Python")
```


```mojo
benchmark_matmul_untyped(128, 128, 128, python_gflops)
```

    0.029258 GFLOP/s, a 17.501798 x speedup over Python


Note the huge speedup with no effort that we have gotten.

## Adding types to the Python implementation

The above program, while achieving better performance than Python, is still not the best we can get from Mojo. If we tell Mojo the types of the inputs, it can optimize much of the code away and reduce dispatching costs (unlike Python, which only uses types for type checking, Mojo exploits type info for performance optimizations as well).

To do that, let's first define a `Matrix` struct. The `Matrix` struct contains a data pointer along with size fields. While the `Matrix` struct can be parametrized on any data type, here we set the data type to be f32 for conciseness.


```mojo
struct Matrix:
    var data: DTypePointer[DType.f32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.f32].alloc(rows * cols)
        rand(self.data, rows*cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> F32:
        return self.load[1](y, x)

    @always_inline
    fn load[nelts:Int](self, y: Int, x: Int) -> SIMD[DType.f32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: F32):
        return self.store[1](y, x, val)

    @always_inline
    fn store[nelts:Int](self, y: Int, x: Int, val: SIMD[DType.f32, nelts]):
        self.data.simd_store[nelts](y * self.cols + x, val)
```

> Note that we implement `getitem` and `setitem` in terms of `load` and `store`. For the naive implementation of matmul it does not make a difference, but we will utilize this later in a more optimized vectorized version of matmul. We are also defining a `load_tr`, which loads a vector from the columns specified at the offset.

With the above `Matrix` type we can effectively copy and paste the Python implementation and just add type annotations:


```mojo
# Note that C, A, and B have types.
fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]
```

We are going to benchmark the implementations as we improve, so let's write a helper function that will do that for us: 


```mojo
@always_inline
def benchmark[func : fn(Matrix, Matrix, Matrix) -> None]
    (M : Int, N : Int, K : Int, python_gflops: F64):
    var C = Matrix(M, N)
    C.zero()
    var A = Matrix(M, K)
    var B = Matrix(K, N)

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    let secs = F64(Benchmark().run[test_fn]()) / 1_000_000_000
    # Prevent matrices from being destroyed before we finished benchmarking them.
    _ = A.data
    _ = B.data
    _ = C.data

    let gflops = ((2*M*N*K)/secs) / 1e9
    let speedup : F64 = gflops / python_gflops
    print(gflops, "GFLOP/s, a", speedup.value, "x speedup over Python")

```

Benchmarking shows significant speedups. We increase the size of the matrix to 512 by 512, since Mojo is much faster than Python.


```mojo
benchmark[matmul_naive](512, 512, 512, python_gflops)
```

    3.120776 GFLOP/s, a 1866.805513 x speedup over Python


Adding type annotations gives a huge improvement compared to the original untyped version.

## Vectorizing the inner most loop

We can do better than the above implementation by utilizing the vector instructions. Rather than assuming a vector width, we query the simd width of the specified dtype using `dtype_simd_width`. This makes our code portable as we transition to other hardware. Leverage SIMD instructions is as easy as:


```mojo
# Mojo has SIMD vector types, we can vectorize the Matmul code as follows.
alias nelts = dtype_simd_width[DType.f32]() # The SIMD vector width.
fn matmul_vectorized_0(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for nv in range(0, C.cols, nelts):
                C.store[nelts](m,nv, C.load[nelts](m,nv) + A[m,k] * B.load[nelts](k,nv))
        
            # Handle remaining elements with scalars.
            for n in range(nelts*(C.cols//nelts), C.cols):
                C[m,n] += A[m,k] * B[k,n]
```

We can benchmark the above implementation. Note that many compilers can detect naive loops and perform optimizations on them. Mojo, however, allows you to be explicit and precisely control what optimizations are applied.


```mojo
benchmark[matmul_vectorized_0](512, 512, 512, python_gflops)
```

    14.318266 GFLOP/s, a 8564.990560 x speedup over Python


Vectorization is a common optimization, and Mojo provides a higher-order function that performs vectorization for you. The `vectorize` function takes a vector width and a function which is parameteric on the vector width and is going to be evaluated in a vectorized manner.


```mojo
# Simplify the code by using the builtin vectorize function
from Functional import vectorize
fn matmul_vectorized_1(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[nelts, dot](C.cols)
```

There is only a slight difference in terms of performance between the two implementations:


```mojo
benchmark[matmul_vectorized_1](512, 512, 512, python_gflops)
```

    13.978696 GFLOP/s, a 8361.864443 x speedup over Python


## Parallelizing Matmul

To get the best performance from modern processors, one has to utilize the multiple cores they have. With Mojo it can be easily achieved with `parallelize` function.

Let's modify our matmul implementation and make it multi-threaded (for simplicity, we only `parallelize` on the M dimension):


```mojo
# Parallelize the code by using the builtin parallelize function
from Functional import parallelize
fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[nelts, dot](C.cols)
        
    parallelize[calc_row](C.rows)
```

We can benchmark the parallel matmul implementation.


```mojo
benchmark[matmul_parallelized](512, 512, 512, python_gflops)
```

    24.091962 GFLOP/s, a 14411.481509 x speedup over Python


## Tiling Matmul

Tiling is an optimization performed for matmul to increase cache locality. The idea is to keep sub-matrices resident in the cache and increase the reuse. The tile function itself can be written in Mojo as:


```mojo
from Functional import Static2DTileUnitFunc as Tile2DFunc
```


```mojo
# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)
```

The above will perform 2 dimensional tiling over a 2D iteration space defined to be between $([0, end_x], [0, end_y])$. Once we define it above, we can use it within our matmul kernel. For simplicity we choose `4` as the tile height and since we also want to vectorize we use `4 * nelts` as the tile width (since we vectorize on the columns).


```mojo
# Use the above tile function to perform tiled matmul.
fn matmul_tiled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n + x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))
                vectorize[nelts, dot](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows)
```

Again, we can benchmark the tiled parallel matmul implementation:


```mojo
benchmark[matmul_tiled_parallelized](512, 512, 512, python_gflops)
```

    23.391380 GFLOP/s, a 13992.403260 x speedup over Python


One source of overhead in the above implementation is the fact that the we are not unrolling the loops introduced by vectorize of the dot function. We can do that via the `vectorize_unroll` higher-order function in Mojo:


```mojo
# Unroll the vectorized loop by a constant factor.
from Functional import vectorize_unroll
fn matmul_tiled_unrolled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n+x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                vectorize_unroll[nelts, tile_x//nelts, dot](tile_x)

        alias tile_size = 4
        tile[calc_tile, nelts*tile_size, tile_size](A.cols, C.cols)
      
    parallelize[calc_row](C.rows)
```

Again, we can benchmark the new tiled parallel matmul implementation with unrolled and vectorized inner loop:


```mojo
benchmark[matmul_tiled_unrolled_parallelized](512, 512, 512, python_gflops)
```

    24.263229 GFLOP/s, a 14513.931176 x speedup over Python


## Searching for the `tile_factor`


```mojo
from Autotune import autotune, search
from Time import now
from Pointer import Pointer

alias matmul_fn_sig_type = fn(Matrix, Matrix, Matrix) -> None
```

The choice of the tile factor can greatly impact the performace of the full matmul,
but the optimal tile factor is highly hardware-dependent, and is influenced by the
cache configuration and other hard-to-model effects. We want to write portable code
without having to know everything about the hardware, so we can ask Mojo to automatically
select the best tile factor using autotuning.


```mojo
# Autotune the tile size used in the matmul.
@adaptive
fn matmul_autotune_impl(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n+x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        # Instead of hardcoding to tile_size = 4, search for the fastest 
        # tile size by evaluting this function as tile size varies.
        alias tile_size = autotune(1, 2, 4, 8, 16, 32)
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)
      
    parallelize[calc_row](C.rows)
```

This will generate multiple candidates for the matmul function. To teach Mojo how
to find the best tile factor, we provide an evaluator function Mojo can use to
assess each candidate.


```mojo
fn matmul_evaluator(funcs: Pointer[matmul_fn_sig_type], size: Int) -> Int:
    print("matmul_evaluator, number of candidates: ", size)

    let eval_begin: Int = now()

    # This size is picked at random, in real code we could use a real size
    # distribution here.
    let M = 512
    let N = 512
    let K = 512
    print("Optimizing for size:", M, "x", N, "x", K)

    var best_idx: Int = -1
    var best_time: Int = -1

    alias eval_iterations = 10
    alias eval_samples = 10

    var C = Matrix(M, N)
    var A = Matrix(M, K)
    var B = Matrix(K, N)
    let Cptr = Pointer[Matrix].address_of(C).address
    let Aptr = Pointer[Matrix].address_of(A).address
    let Bptr = Pointer[Matrix].address_of(B).address

    # Find the function that's the fastest on the size we're optimizing for
    for f_idx in range(size):
        let func = funcs.load(f_idx)

        @always_inline
        @parameter
        fn wrapper():
            func(C, A, B)
        let cur_time = Benchmark(1, 100_000, 500_000_000, 1000_000_000).run[wrapper]()

        if best_idx < 0:
            best_idx = f_idx
            best_time = cur_time
        if best_time > cur_time:
            best_idx = f_idx
            best_time = cur_time

    let eval_end: Int = now()
    # Prevent matrices from being destroyed before we finished benchmarking them.
    _ = A.data
    _ = B.data
    _ = C.data
    print("Time spent in matmul_evaluator, ms:", (eval_end - eval_begin) // 1000000)
    print("Best candidate idx:", best_idx)
    return best_idx
```

Finally, we need to define an entry function that would simply call the best candidate.


```mojo
fn matmul_autotune(C: Matrix, A: Matrix, B: Matrix):
    alias best_impl: matmul_fn_sig_type
    search[
        matmul_fn_sig_type,
        VariadicList(matmul_autotune_impl.__adaptive_set),
        matmul_evaluator -> best_impl
    ]()
    # Run the best candidate
    return best_impl(C, A, B)
```

Let's benchmark our new implementation:


```mojo
benchmark[matmul_autotune](512, 512, 512, python_gflops)
```

    matmul_evaluator, number of candidates:  6
    Optimizing for size: 512 x 512 x 512
    Time spent in matmul_evaluator, ms: 8650
    Best candidate idx: 2
    23.488668 GFLOP/s, a 14050.599768 x speedup over Python

