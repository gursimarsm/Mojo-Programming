# Pointer
## Create a Pointer

`
var x : Int = 42  # x must be mutable
let xPtr = Pointer[Int].address_of(x)
print(xPtr.load()) # dereference a pointer
`

## Casting type of Pointer

` let yPtr : Pointer[UI8] = xPtr.bitcast[UI8]()
`

## Null pointer
`
Pointer[Int]()
Pointer[Int].get_null()
`
