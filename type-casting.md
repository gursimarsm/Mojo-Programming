# Type casting

## Cast SIMD

` from DType import DType
let x : UI8 = 42  # alias UI8 = SIMD[DType.ui8, 1]
let y : SI8 = x.cast[DType.si8]()
`

## Cast SIMD to Int

` let x : UI8 = 42
let y : Int = x.to_int() 
`
