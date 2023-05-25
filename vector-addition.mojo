from Functional import vectorize
from Targetlnfo import dtype_simd_width
from SIMD import SIMD

alias nelts = dtype_smid_width[Dtype.f32]()
fn add simd(arrl: MojoArr, arr2: MojoArr, result: MojoArr, dim: Int):
    @parameter
    fn add[nelts: Int](n: Int):
        result.smid_store[nelts](n, arr1.smid_store[nelts](n) + arr2.simd_load[nelts] (n))
        
     vectorize[nelts, add] (dim)
