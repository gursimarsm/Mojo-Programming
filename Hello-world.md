# Hello, Mojo🔥

Mojo is designed as a superset of Python, so a lot of language features you are
familiar with and the concepts that you know in Python translate directly to
Mojo. For instance, a "Hello World" program in Mojo looks exactly as it does in
Python:


```mojo
print("Hello Mojo!")
```

And as we'll show later, you can also import existing Python packages and use
them like you're used to.

But Mojo provides a ton of powerful features on top of Python,
so that's what we'll focus on in this notebook. 

To be clear, this guide is not your traditional introduction to a programming
language. This notebook assumes you're already familiar Python and some systems
programming concepts so we can focus on what's special about Mojo.

This runnable notebook is actually based on the [Mojo programming
manual](https://docs.modular.com/mojo/programming-manual.html), but we've
simplified a bunch of the explanation so you can focus on playing with the
code. If you want to learn more about a topic, refer to the complete manual.

Let's get started!

<div class="alert alert-block alert-success">
<b>Share feedback:</b> We really want to hear from you! Please send us bug
reports, suggestions, and questions through our <a
href="https://docs.modular.com/mojo/community.html">Mojo community channels</a>.
</div>

<div class="alert alert-block alert-info">
<b>Note:</b> Mojo Playground is designed only for testing the Mojo language.
The cloud environment is not always stable and performance varies, so it is not
an appropriate environment for performance benchmarking. However, we believe it
can still demonstrate the magnitude of performance gains provided by Mojo, as
shown in the <code>Matmul.ipynb</code> notebook. For
more information about the compute power in the Mojo Playground, see the <a
href="https://docs.modular.com/mojo/faq.html#mojo-playground">Mojo FAQ</a>.
</div>

## Basic systems programming extensions

Python is not designed nor does it excel for systems programming, but Mojo is. This section describes how to perform basic systems programming in Mojo.

### `let` and `var` declarations

Exactly like Python you can assign values to a name and it implicitly creates a
function-scope variable within a function. This provides a very dynamic and
easy way to write code, but it also creates a challenge for two reasons:

1) Systems programmers often want to declare that a value that is immutable.
2) Systems programmers want to get an error if they mistype a variable name in an assignment.

To support this, Mojo supports `let` and `var` declarations, which introduce a
new scoped runtime value: `let` is immutable and `var` is mutable. These
values use lexical scoping and support name shadowing:


```mojo
def your_function(a, b):
    let c = a
    # Uncomment to see an error:
    # c = b  # error: c is immutable

    if c != b:
        let d = b
        print(d)

your_function(2, 3)
```

`let` and `var` declarations also support type specifiers, patterns,
and late initialization:


```mojo
def your_function():
    let x: Int = 42
    let y: F64 = 17.0

    let z: F32
    if x != 0:
        z = 1.0
    else:
        z = foo()
    print(z)

def foo() -> F32:
    return 3.14

your_function()
```

### `struct` types

Modern systems programming have the ability to build high-level and safe
abstractions on top of low-level data layout controls, indirection-free field
access, and other niche tricks. Mojo provides that with the `struct` type.

`struct` types are similar in many ways to classes. However, where classes are
extremely dynamic with dynamic dispatch, monkey-patching (or dynamic method
"swizzling"), and dynamically bound instance properties, `struct`s are static,
bound at compile time, and are inlined into their container instead of being
implicitly indirect and reference counted.

Here’s a simple definition of a `struct`:


```mojo
struct MyPair:
    var first: Int
    var second: Int

    # We use 'fn' instead of 'def' here - we'll explain that soon
    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second

    fn __lt__(self, rhs: MyPair) -> Bool:
        return self.first < rhs.first or
              (self.first == rhs.first and
               self.second < rhs.second)
```

The biggest difference compared to a `class` is that all instance properties in
a `struct` **must** be explicitly declared with a `var` or `let` declaration.
This allows the Mojo compiler to layout and access property values
precisely in memory without indirection or other overhead.

Struct fields are bound statically: they aren't looked up with a dictionary
indirection. As such, you cannot `del` a method or reassign it at runtime. This
enables the Mojo compiler to perform guaranteed static dispatch, use guaranteed
static access to fields, and inline a struct into the stack frame or enclosing
type that uses it without indirection or other overheads.

### Strong type checking

Although you can still use dynamic types just like in Python, Mojo also allows
you to use strong type checking in your program. 

One of the primary ways to employ strong type checking is with Mojo's `struct`
type. A `struct` definition in Mojo defines a compile-time-bound name, and
references to that name in a type context are treated as a strong specification
for the value being defined. For example, consider the following code that uses
the `MyPair` struct shown above:


```mojo
def pairTest() -> Bool:
    let p = MyPair(1, 2)
    # Uncomment to see an error:
    # return p < 4 # gives a compile time error
    return True
```

If you uncomment the first return statement and run it, you’ll get a
compile-time error telling you that `4` cannot be converted to `MyPair`, which
is what the RHS of `__lt__` requires (in the `MyPair` definition).


### Overloaded functions & methods

Also just like Python, you can define functions in Mojo without specifying
argument types and let Mojo infer the data types. But when you want to ensure
type safety, Mojo also offers full support for overloaded functions and
methods.

Essentially, this allows you to define multiple functions with the same name
but with different arguments. This is a common feature seen in many languages
such as C++, Java, and Swift.

Let’s look at an example:


```mojo
struct Complex:
    var re: F32
    var im: F32

    fn __init__(inout self, x: F32):
        """Construct a complex number given a real number."""
        self.re = x
        self.im = 0.0

    fn __init__(inout self, r: F32, i: F32):
        """Construct a complex number given its real and imaginary components."""
        self.re = r
        self.im = i
```

You can implement overloads anywhere you want: for module functions and for
methods in a class or a struct.

Mojo doesn't support overloading solely on result type, and doesn't use result
type or contextual type information for type inference, keeping things simple,
fast, and predictable. Mojo will never produce an "expression too complex"
error, because its type-checker is simple and fast by definition.

### `fn` definitions

The extensions above are the cornerstone that provides low-level programming
and provide abstraction capabilities, but many systems programmers prefer more
control and predictability than what `def` in Mojo provides. To recap, `def` is
defined by necessity to be very dynamic, flexible and generally compatible with
Python: arguments are mutable, local variables are implicitly declared on first
use, and scoping isn’t enforced. This is great for high level programming and
scripting, but is not always great for systems programming. To complement this,
Mojo provides an `fn` declaration which is like a “strict mode” for `def`.

`fn` and `def` are always interchangeable from an interface level: there is
nothing a `def` can provide that a `fn` cannot (or vice versa). The
difference is that a `fn` is more limited and controlled on the _inside_ of
its body (alternatively: pedantic and strict). Specifically, `fn`s have a
number of limitations compared to `def`s:

1. Argument values default to being immutable in the body of the function (like
a `let`), instead of mutable (like a `var`). This catches accidental
mutations, and permits the use of non-copyable types as arguments.

2. Argument values require a type specification (except for `self` in a
method), catching accidental omission of type specifications. Similarly, a
missing return type specifier is interpreted as returning `None` instead of an
unknown return type. Note that both can be explicitly declared to return
`object`, which allows one to opt-in to the behavior of a `def` if desired.

3. Implicit declaration of local variables is disabled, so all locals must be
declared. This catches name typos and dovetails with the scoping provided by
`let` and `var`.

4. Both support raising exceptions, but this must be explicitly declared on a
`fn` with the `raises` function effect, placed after the function argument
list.

### The `__copyinit__` and `__moveinit__` special methods

Mojo supports full "value semantics" as seen in languages like C++ and Swift,
and it makes defining simple aggregates of fields very easy with its `@value`
decorator (described in more detail in the [Programming Manual](https://docs.modular.com/mojo/programming-manual.html)).

For advanced use cases, Mojo allows you to define custom constructors (using
Python's existing `__init__` special method), custom destructors (using the
existing `__del__` special method) and custom copy and move constructors using
the new `__copyinit__` and `__moveinit__` special methods.

These low-level customization hooks can be useful when doing low level systems
programming, e.g. with manual memory management. For example, consider a heap
array type that needs to allocate memory for the data when constructed and
destroy it when the value is destroyed:


```mojo
from Pointer import Pointer
from IO import print_no_newline

struct HeapArray:
    var data: Pointer[Int]
    var size: Int
    var cap: Int

    fn __init__(inout self):
        self.cap = 16
        self.size = 0
        self.data = Pointer[Int].alloc(self.cap)

    fn __init__(inout self, size: Int, val: Int):
        self.cap = size * 2
        self.size = size
        self.data = Pointer[Int].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, val)
     
    fn __del__(owned self):
        self.data.free()

    fn dump(self):
        print_no_newline("[")
        for i in range(self.size):
            if i > 0:
                print_no_newline(", ")
            print_no_newline(self.data.load(i))
        print("]")
```

This array type is implemented using low level functions to show a simple
example of how this works. However, if you go ahead and try this out, you might
be surprised:


```mojo
var a = HeapArray(3, 1)
a.dump()   # Should print [1, 1, 1]
# Uncomment to see an error:
# var b = a  # ERROR: Vector doesn't implement __copyinit__

var b = HeapArray(4, 2)
b.dump()   # Should print [2, 2, 2, 2]
a.dump()   # Should print [1, 1, 1]
```

The compiler isn’t allowing us to make a copy of our array: `HeapArray` contains
an instance of `Pointer` (which is equivalent to a low-level C pointer), and Mojo
can’t know “what the pointer means” or “how to copy it” - this is one reason
why application level programmers should use higher level types like arrays and
slices! More generally, some types (like atomic numbers) cannot be copied or
moved around at all, because their address provides an **identity** just like a
class instance does.

In this case, we do want our array to be copyable around, and to enable this, we
implement the `__copyinit__` special method, which conventionally looks like this:


```mojo
struct HeapArray:
    var data: Pointer[Int]
    var size: Int
    var cap: Int

    fn __init__(inout self):
        self.cap = 16
        self.size = 0
        self.data = Pointer[Int].alloc(self.cap)

    fn __init__(inout self, size: Int, val: Int):
        self.cap = size * 2
        self.size = size
        self.data = Pointer[Int].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, val)

    fn __copyinit__(inout self, other: Self):
        self.cap = other.cap
        self.size = other.size
        self.data = Pointer[Int].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, other.data.load(i))
            
    fn __del__(owned self):
        self.data.free()

    fn dump(self):
        print_no_newline("[")
        for i in range(self.size):
            if i > 0:
                print_no_newline(", ")
            print_no_newline(self.data.load(i))
        print("]")

```

With this implementation, our code above works correctly and the `b = a` copy
produces a logically distinct instance of the array with its own lifetime and
data. Mojo also supports the `__moveinit__` method which allows both Rust-style
moves (which take a value when a lifetime ends) and C++-style moves (where the
contents of a value is removed but the destructor still runs), and allows
defining custom move logic.  Please see the [Value
Lifecycle](https://docs.modular.com/mojo/programming-manual.html#value-lifecycle-birth-life-and-death-of-a-value)
section in the Programming Manual for more information.


```mojo
var a = HeapArray(3, 1)
a.dump()   # Should print [1, 1, 1]
# This is no longer an error:
var b = a

b.dump()   # Should print [1, 1, 1]
a.dump()   # Should print [1, 1, 1]

```


Mojo provides full control over the lifetime of a value, including the ability
to make types copyable, move-only, and not-movable. This is more control than
languages like Swift and Rust, which require values to at least be movable. If
you are curious how `existing` can be passed into the `__copyinit__` method
without itself creating a copy, check out the section on
[Borrowed](#borrowed-argument-convention) argument convention below.

## Python integration

It's easy to use Python modules you know and love in Mojo. You can import
any Python module into your Mojo program and create Python types from Mojo
types.


### Importing Python modules

To import a Python module in Mojo, just call `Python.import_module()` with the
module name:


```mojo
from PythonInterface import Python

# This is equivalent to Python's `import numpy as np`
let np = Python.import_module("numpy")

# Now use numpy as if writing in Python
array = np.array([1, 2, 3])
print(array)
```

Yes, this imports Python NumPy, and you can import _any other Python module_.

Currently, you cannot import individual members (such as a single Python class
or function)—you must import the whole Python module and then access members
through the module name.

There's no need to worry about memory management when using Python in Mojo.
Everything just works because Mojo was designed for Python from the beginning.

### Mojo types in Python

Mojo primitive types implicitly convert into Python objects.
Today we support lists, tuples, integers, floats, booleans, and strings.

For example, given this Python function that prints Python types:


```mojo
%%python
def type_printer(my_list, my_tuple, my_int, my_string, my_float):
    print(type(my_list))
    print(type(my_tuple))
    print(type(my_int))
    print(type(my_string))
    print(type(my_float))
```

You can pass the Python function Mojo types, no problem:


```mojo
type_printer([0, 3], (False, True), 4, "orange", 3.4)
```

Notice that in a Jupyter notebook, the Python function declared above is
automatically available to any Mojo code in following code cells. (In other
situations, you will need to [import the Python
module](https://docs.modular.com/mojo/programming-manual.html#python-integration).)

Mojo doesn't have a standard Dictionary yet, so it is not yet possible
to create a Python dictionary from a Mojo dictionary. You can work with
Python dictionaries in Mojo though!


## Parameterization: compile time meta-programming

Mojo supports a full compile-time metaprogramming functionality built into the
compiler as a separate stage of compilation - after parsing, semantic analysis,
and IR generation, but before lowering to target-specific code. It uses the same
host language for runtime programs as it does for metaprograms, and leverages
MLIR to represent and evaluate these programs in a predictable way.

Let’s take a look at some simple examples.

### Defining parameterized types and functions

Mojo structs and functions may each be parameterized, but an example can help
motivate why we care. Let’s look at a
“[SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)” type,
which represents a low-level vector register in hardware that holds multiple
instances of a scalar data-type. Hardware accelerators these days are getting
exotic datatypes, and it isn’t uncommon to work with CPUs that have 512-bit or
longer SIMD vectors. There is a lot of diversity in hardware (including many
brands like SSE, AVX-512, NEON, SVE, RVV, etc) but many operations are common
and used by numerics and ML kernel developers - this type exposes them to Mojo
programmers.

Here is very simplified and cut down version of the SIMD API from the Mojo
standard library. We use `HeapArray` to store the SIMD data for this example and
implement basic operations on our type using loops - we do that simply to mimic
the desired SIMD type behavior for the sake of demonstration. The real Stdlib
implementation is backed by real SIMD instructions which are accessed through
Mojo's ability to use MLIR directly (see more on that topic in the [Advanced
Mojo Features](#advanced-mojo-features) section).


```mojo
from List import VariadicList

struct MySIMD[size: Int]:
    var value: HeapArray

    # Create a new SIMD from a number of scalars
    fn __init__(inout self, *elems: Int):
        self.value = HeapArray(size, 0)
        let elems_list = VariadicList(elems)
        for i in range(elems_list.__len__()):
            self[i] = elems_list[i]

    fn __copyinit__(inout self, other: MySIMD[size]):
        self.value = other.value

    fn __getitem__(self, i: Int) -> Int:
        return self.value.data.load(i)
    
    fn __setitem__(self, i: Int, val: Int):
        return self.value.data.store(i, val)

    # Fill a SIMD with a duplicated scalar value.
    fn splat(self, x: Int) -> Self:
        for i in range(size):
            self[i] = x
        return self

    # Many standard operators are supported.
    fn __add__(self, rhs: MySIMD[size]) -> MySIMD[size]:
        let result = MySIMD[size]()
        for i in range(size):
            result[i] = self[i] + rhs[i]
        return result
    
    fn __sub__(self, rhs: Self) -> Self:
        let result = MySIMD[size]()
        for i in range(size):
            result[i] = self[i] - rhs[i]
        return result

    fn concat[rhs_size: Int](self, rhs: MySIMD[rhs_size]) -> MySIMD[size + rhs_size]:
        let result = MySIMD[size + rhs_size]()
        for i in range(size):
            result[i] = self[i]
        for j in range(rhs_size):
            result[size + j] = rhs[j]
        return result

    fn dump(self):
        self.value.dump()
```

Parameters in Mojo are declared in square brackets using an extended version of
the [PEP695 syntax](https://peps.python.org/pep-0695/). They are named and have
types like normal values in a Mojo program, but they are evaluated at compile
time instead of runtime by the target program. The runtime program may use the
value of parameters - because the parameters are resolved at compile time
before they are needed by the runtime program - but the compile time parameter
expressions may not use runtime values.

In the example above, there are two declared parameters: the `MySIMD` struct is
parameterized by a `size` parameter, and `concat` method is further
parametrized with an `rhs_size` parameter. Because `MySIMD` is a parameterized
type, the type of a `self` argument carries the parameters - the full type
name is `MySIMD[size]`. While it is always valid to write this out (as shown
in the return type of `_add__`), this can be verbose: we recommend using the
`Self` type (from [PEP673](https://peps.python.org/pep-0673/)) like the
`__sub__` example does.

The actual `SIMD` type provided by Mojo Stdlib is also parametrized on a data
type of the elements.

### Using parameterized types and functions

The `size` specifies the number of elements in a SIMD vector, the example
below shows how our type can be used:


```mojo
# Make a vector of 4 elements.
let a = MySIMD[4](1, 2, 3, 4)

# Make a vector of 4 elements and splat a scalar value into it.
let b = MySIMD[4]().splat(100)

# Add them together and print the result
let c = a + b
c.dump()

# Make a vector of 2 elements.
let d = MySIMD[2](10, 20)

# Make a vector of 2 elements.
let e = MySIMD[2](70, 50)

let f = d.concat[2](e)
f.dump()

# Uncomment to see the error:
# let x = a + e # ERROR: Operation MySIMD[4]+MySIMD[2] is not defined

let y = f + a
y.dump()

```

Note that the `concat` method needs an additional parameter to indicate the
size of the second SIMD vector: that is handled by parameterizing the call to
`concat`. Our toy SIMD type shows the use of a concrete type (`Int`), but the
major power of parameters comes from the ability to define parametric algorithms
and types, e.g. it is quite easy to define parametric algorithms, e.g. ones that
are length- and DType-agnostic:



```mojo
from DType import DType
from Math import sqrt

fn rsqrt[width: Int, dt: DType](x: SIMD[dt, width]) -> SIMD[dt, width]:
    return 1 / sqrt(x)
```

The Mojo compiler is fairly smart about type inference with parameters. Note
that this function is able to call the parametric `sqrt(x)` function without
specifying the parameters, the compiler infers its parameters as if you wrote
`sqrt[width,type](x)` explicitly. Also note that `rsqrt` chose to define
its first parameter named `width` but the SIMD type names it `size` without
challenge.

### Parameter expressions are just Mojo code

All parameters and parameter expressions are typed using the same type system
as the runtime program: `Int` and `DType` are implemented in the Mojo standard
library as structs. Parameters are quite powerful, supporting the use of
expressions with operators, function calls etc at compile time, just like a
runtime program. This enables the use of many ‘dependent type’ features, for
example, you might want to define a helper function to concatenate two SIMD
vectors, like we did in the example above:


```mojo
fn concat[len1: Int, len2: Int](lhs: MySIMD[len1], rhs: MySIMD[len2]) -> MySIMD[len1+len2]:
    let result = MySIMD[len1 + len2]()
    for i in range(len1):
        result[i] = lhs[i]
    for j in range(len2):
        result[len1 + j] = rhs[j]
    return result


let a = MySIMD[2](1, 2)
let x = concat[2,2](a, a)
x.dump()
```

Note how the result length is the sum of the input vector lengths, and you can
express that with a simple + operation. For a more complex example, take a look
at the `SIMD.shuffle` method in the standard library: it takes two input SIMD
values, a vector shuffle mask as a list, and returns a SIMD that matches the
length of the shuffle mask.

### Powerful compile-time programming

While simple expressions are useful, sometimes you want to write imperative
compile-time logic with control flow. For example, the `isclose` function in the
`Math` module uses exact equality for integers but `close` comparison for floating
point. You can even do compile time recursion, e.g. here is an example “tree
reduction” algorithm that sums all elements of a vector recursively into a
scalar:


```mojo
fn slice[new_size: Int, size: Int](x: MySIMD[size], offset: Int) -> MySIMD[new_size]:
    let result = MySIMD[new_size]()
    for i in range(new_size):
        result[i] = x[i + offset]
    return result

fn reduce_add[size: Int](x: MySIMD[size]) -> Int:
    @parameter
    if size == 1:
        return x[0]
    elif size == 2:
        return x[0] + x[1]

    # Extract the top/bottom halves, add them, sum the elements.
    alias half_size = size // 2
    let lhs = slice[half_size, size](x, 0)
    let rhs = slice[half_size, size](x, half_size)
    return reduce_add[half_size](lhs + rhs)
    
let x = MySIMD[4](1, 2, 3, 4)
x.dump()
print("Elements sum:", reduce_add[4](x))
```

This makes use of the `@parameter if` feature, which is an if statement that
runs at compile time. It requires that its condition be a valid parameter
expression, and ensures that only the live branch of the if is compiled into
the program.

### Mojo types are just parameter expressions

While we’ve shown how you can use parameter expressions within types, in both
Python and Mojo, type annotations can themselves be arbitrary expressions.
Types in Mojo have a special metatype type, allowing type-parametric algorithms
and functions to be defined, for example we can extend our `HeapArray` struct to
support arbitrary types of the elements:


```mojo
struct Array[Type: AnyType]:
    var data: Pointer[Type]
    var size: Int
    var cap: Int

    fn __init__(inout self):
        self.cap = 16
        self.size = 0
        self.data = Pointer[Type].alloc(self.cap)

    fn __init__(inout self, size: Int, value: Type):
        self.cap = size * 2
        self.size = size
        self.data = Pointer[Type].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, value)

    fn __copyinit__(inout self, other: Self):
        self.cap = other.cap
        self.size = other.size
        self.data = Pointer[Type].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, other.data.load(i))
    
    fn __getitem__(self, i: Int) -> Type:
        return self.data.load(i)
    
    fn __setitem__(self, i: Int, value: Type):
        return self.data.store(i, value)
            
    fn __del__(owned self):
        self.data.free()

var v = Array[F32](4, 3.14)
print(v[0], v[1], v[2], v[3])
```

Notice that the `type` parameter is being used as the formal type for the
`value` arguments and the return type of the `__getitem__` function. Parameters
allow the `Array` type to provide different APIs based on the different
use-cases. There are many other cases that benefit from more advanced use
cases. For example, the parallel processing library defines the
`parallelForEachN` algorithm, which executes a closure N times in parallel,
feeding in a value from the context. That value can be of any type:


```mojo
fn parallelize[func: fn (Int) -> None](num_work_items: Int):
    # Not actually parallel: see the 'Functional' module for real implementation.
    for i in range(num_work_items):
        func(i)
```

Another example where this is important is with variadic generics, where an
algorithm or data structure may need to be defined over a list of heterogenous
types:


```mojo
#struct Tuple[*ElementTys: AnyType]:
#    var _storage : ElementTys
```

which will allow us to fully define `Tuple` (and related types like
`Function`) in the standard library. This is not implemented yet, but we
expect this to work soon.

### `alias`: named parameter expressions

It is very common to want to *name* compile time values. Whereas `var` defines a
runtime value, and `let` defines a runtime constant, we need a way to define a
compile time temporary value. For this, Mojo uses an `alias` declaration. For
example, the `DType` struct implements a simple enum using aliases for the
enumerators like this (the actual internal implementation details vary a bit):


```mojo
struct dtype:
    alias invalid = 0
    alias bool = 1
    alias si8 = 2
    alias ui8 = 3
    alias si16 = 4
    alias ui16 = 5
    alias f32 = 15
```

This allows clients to use `DType.f32` as a parameter expression (which also
works as a runtime value of course) naturally.

Types are another common use for `alias`: because types are just compile time
expressions, it is very handy to be able to do things like this:


```mojo
alias F16 = SIMD[DType.f16, 1]
alias UI8 = SIMD[DType.ui8, 1]

var x : F16   # F16 works like a "typedef"
```

Like `var` and `let`, aliases obey scope and you can use local aliases
within functions as you’d expect.

### Autotuning and adaptive compilation

Mojo parameter expressions allow you to write portable parametric algorithms
like you can do in other languages, but when writing high performance code you
still have to pick concrete values to use for the parameters. For example when
writing high performance numeric algorithms, you might want to use memory
tiling to accelerate the algorithm, but the dimensions to use depend highly on
the available hardware features, the sizes of the cache, what gets fused into
the kernel, and many other fiddly details.

Even vector length can be difficult to manage, because the vector length of a
typical machine depends on the datatype, and some datatypes like `bfloat16`
don't have full support on all implementations. Mojo helps by providing an
`autotune` function in the standard library. For example, if you want to write
a vector-length-agnostic algorithm to a buffer of data, you might write it like
this:


```mojo
from Autotune import autotune
from Pointer import DTypePointer
from Functional import vectorize

fn buffer_elementwise_add[
    dt: DType
](lhs: DTypePointer[dt], rhs: DTypePointer[dt], result: DTypePointer[dt], N: Int):
    """Perform elementwise addition of N elements in RHS and LHS and store
    the result in RESULT.
    """
    @parameter
    fn add_simd[size: Int](idx: Int):
        let lhs_simd = lhs.simd_load[size](idx)
        let rhs_simd = rhs.simd_load[size](idx)
        result.simd_store[size](idx, lhs_simd + rhs_simd)
    
    # Pick vector length for this dtype and hardware
    alias vector_len = autotune(1, 4, 8, 16, 32)

    # Use it as the vectorization length
    vectorize[vector_len, add_simd](N)
```

We can now call our function as usual:


```mojo
let N = 32
let a = DTypePointer[DType.f32].alloc(N)
let b = DTypePointer[DType.f32].alloc(N)
let res = DTypePointer[DType.f32].alloc(N)
# Initialize arrays with some values
for i in range(N):
    a.store(i, 2.0)
    b.store(i, 40.0)
    res.store(i, -1)
    
buffer_elementwise_add[DType.f32](a, b, res, N)
print(a.load(10), b.load(10), res.load(10))
```

When compiling instantiations of this code Mojo forks compilation of this
algorithm and decides which value to use by measuring what works best in
practice for the target hardware. It evaluates the different values of the
`vector_len` expression and picks the fastest one according to a user-defined
performance evaluator. Because it measures and evaluates each option
individually, it might pick a different vector length for `F32` than for
`SI8`, for example. This simple feature is pretty powerful - going beyond
simple integer constants - because functions and types are also parameter
expressions.

Autotuning is an inherently exponential technique that benefits from internal
implementation details of the Mojo compiler stack (particularly MLIR, integrated
caching, and distribution of compilation). This is also a power-user feature and
needs continued development and iteration over time.

In the example above we didn't define the performance evaluator function, and
the compiler just picked one of the available implementations. However, we dive
deep into how to do that in other notebooks: we recommend checking out [Matrix
Multiplication](https://docs.modular.com/mojo/notebooks/Matmul.html) and [Fast
Memset in Mojo](https://docs.modular.com/mojo/notebooks/Memset.html).

## Argument passing control and memory ownership

In both Python and Mojo, much of the language revolves around function calls: a
lot of the (apparently) built-in functionality is implemented in the standard
library with “dunder” methods. Mojo takes this a step further than Python, by
putting the most basic things (like integers and the `object` type itself) into
the standard library.

### Why argument conventions are important

In Python all fundamental values are references to objects - a Python programmer
typically thinks about the programming model as everything being reference
semantic. However, at the CPython or machine level, we can see that the
references themselves are actually passed _by-copy_, by copying a pointer and
adjusting reference counts.

Mojo on the other hand provides full control over value copies, aliasing of
references, and mutations.

### By-reference arguments

Let’s start with the simple case: passing mutable references to values vs
passing immutable references. As we already know, arguments that are passed to
`fn`’s are immutable by default:


```mojo
struct MyInt:
    var value: Int
    fn __init__(inout self, v: Int):
        self.value = v
    fn __copyinit__(inout self, other: MyInt):
        self.value = other.value
        
        
    # self and rhs are both immutable in __add__.
    fn __add__(self, rhs: MyInt) -> MyInt:
        return MyInt(self.value + rhs.value)
        

    # ... but this cannot work for __iadd__
    # Uncomment to see the error:
    #fn __iadd__(self, rhs: Int):
    #    self = self + rhs  # ERROR: cannot assign to self!

```

The problem here is that `__iadd__` needs to mutate the internal state of the
integer. The solution in Mojo is to declare that the argument is passed “inout“
 by using the `inout` marker on the argument name (`self` in this
case):


```mojo
struct MyInt:
    var value: Int
    fn __init__(inout self, v: Int):
        self.value = v

    fn __copyinit__(inout self, other: MyInt):
        self.value = other.value
        
    # self and rhs are both immutable in __add__.
    fn __add__(self, rhs: MyInt) -> MyInt:
        return MyInt(self.value + rhs.value)
        

    # ... now this works:
    fn __iadd__(inout self, rhs: Int):
        self = self + rhs  # OK
```

Because this argument is passed by-reference, the `self` argument is mutable in
the callee, and any changes are visible in the caller - even if the caller has
a non-trivial computation to access it, like an array subscript:



```mojo
var x = 42
x += 1
print(x)    # prints 43 of course

var a = Array[Int](16, 0)
a[4] = 7
a[4] += 1
print(a[4])  # Prints 8

let y = x
# Uncomment to see the error:
# y += 1       # ERROR: Cannot mutate 'let' value
```

Mojo implements the in-place mutation of the `Array` element by
emitting a call to `__getitem__` into a temporary buffer, followed by a store
with `__setitem__` after the call. Mutation of the `let` value fails because it
isn’t possible to form a mutable reference to an immutable value. Similarly,
the compiler rejects attempts to use a subscript with a by-ref argument if it
implements `__getitem__` but not `__setitem__`.

There is nothing special about `self` in Mojo, and you can have multiple
different by-ref arguments. For example, you can define and use a swap function
like this:



```mojo
fn swap(inout lhs: Int, inout rhs: Int):
    let tmp = lhs
    lhs = rhs
    rhs = tmp

var x = 42
var y = 12
print(x, y)  # Prints 42, 12
swap(x, y)
print(x, y)  # Prints 12, 42
```

<a id='borrowed-argument-convention'></a>

### “Borrowed” argument convention

Now that we know how by-reference argument passing works, you may wonder how
by-value argument passing works and how that interacts with the `__copyinit__`
method which implements copy constructors. In Mojo, the default convention for
passing arguments to functions is to pass with the “borrowed” argument
convention. You can spell this out explicitly if you’d like:



```mojo
# A type that is so expensive to copy around we don't even have a
# __copyinit__ method.
struct SomethingBig:
    var id_number: Int
    var huge: Array[Int]
    fn __init__(inout self, id: Int):
        self.huge = Array[Int](1000, 0)
        self.id_number = id

    # self is passed by-reference for mutation as described above.
    fn set_id(inout self, number: Int):
        self.id_number = number

    # Arguments like self are passed as borrowed by default.
    fn print_id(self):  # Same as: fn print_id(borrowed self):
        print(self.id_number)

fn use_something_big(borrowed a: SomethingBig, b: SomethingBig):
    """'a' and 'b' are passed the same, because 'borrowed' is the default."""
    a.print_id()
    b.print_id()

let a = SomethingBig(10)
let b = SomethingBig(20)
use_something_big(a, b)
```

This default applies to all arguments uniformly, including the `self` argument
of methods. The borrowed convention passes an _immutable reference_ to the
value from the caller’s context, instead of copying the value. This is much
more efficient when passing large values, or when passing expensive values like
a reference counted pointer (which is the default for Python/Mojo classes),
because the copy constructor and destructor don’t have to be invoked when
passing the argument. Here is a more elaborate example building on the code
above:



```mojo
fn try_something_big():
    # Big thing sits on the stack: after we construct it it cannot be
    # moved or copied.
    let big = SomethingBig(30)
    # We still want to do useful things with it though!
    big.print_id()
    # Do other things with it.
    use_something_big(big, big)

try_something_big()
```

Because the default argument convention is borrowed, we get very simple and
logical code which does the right thing by default: for example, we don’t want
to copy or move all of `SomethingBig` just to invoke the `print_id` method,
or when calling `use_something_big`.

The borrowed convention is similar and has precedent to other languages. For
example, the borrowed argument convention is similar in some ways to passing an
argument by `const&` in C++. This avoids a copy of the value, and disables
mutability in the callee. The borrowed convention differs from `const&` in
C++ in two important ways though:

1. The Mojo compiler implements a borrow checker (similar to Rust) that
prevents code from dynamically forming mutable references to a value when there
are immutable references outstanding, and prevents having multiple mutable
references to the same value. You are allowed to have multiple borrows (as the
call to `use_something_big` does above) but cannot pass something by mutable
reference and borrow at the same time. (TODO: Not currently enabled).

2. Small values like `Int`, `Float`, and `SIMD` are passed directly in
machine registers instead of through an extra indirection (this is because they
are declared with the `@register_passable` decorator, see below). This is a
[significant performance
enhancement](https://www.forrestthewoods.com/blog/should-small-rust-structs-be-passed-by-copy-or-by-borrow/)
when compared to languages like C++ and Rust, and moves this optimization from
every call site to being declarative on a type.

Rust is another important language and the Mojo and Rust borrow checkers
enforce the same exclusivity invariants. The major difference between Rust and
Mojo is that no sigil is required on the caller side to pass by borrow, Mojo is
more efficient when passing small values, and Rust defaults to moving values by
default instead of passing them around by borrow. These policy and syntax
decisions allows Mojo to provide an arguably easier to use programming model.


### “Owned” argument convention

The final argument convention that Mojo supports is the `owned` argument
convention. This convention is used for functions that want to take exclusive
ownership over a value, and it is often used with the postfix `^` operator.

For example, consider working with a move-only type like a unique pointer:


```mojo
# This is not really a unique pointer, we just model its behavior here:
struct UniquePointer:
    var ptr: Int
    
    fn __init__(inout self, ptr: Int):
        self.ptr = ptr
    
    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = existing.ptr
        
    fn __del__(owned self):
        self.ptr = 0
```

If we try copying it, we would correctly get an error:


```mojo
let p = UniquePointer(100)
# Uncomment to see the error:
# let q = p # ERROR: value of type 'UniquePointer' cannot be copied into its destination
```

While the borrow convention makes it easy to work with the unique pointer
without ceremony, at some point you may want to transfer ownership to some other
function. This is what the `^` operator does.

For movable types, the `^` operator ends the lifetime of a value binding and
transfers the value to something else (in this case, the `take_ptr` function).
To support this, you can define functions as taking owned arguments, e.g. you
define `take_ptr` like so:


```mojo
fn use_ptr(borrowed p: UniquePointer):
    print("use_ptr")
    print(p.ptr)

fn take_ptr(owned p: UniquePointer):
    print("take_ptr")
    print(p.ptr)
    
fn work_with_unique_ptrs():
    let p = UniquePointer(100)
    use_ptr(p)    # Perfectly fine to pass to borrowing function.
    use_ptr(p)
    take_ptr(p^)  # Pass ownership of the `p` value to another function.

    # Uncomment to see an error:
    # use_ptr(p) # ERROR: p is no longer valid here!

work_with_unique_ptrs()
```

Because it is declared `owned`, the `take_ptr` function knows it has unique
access to the value.  This is very important for things like unique pointers,
can be useful to avoid copies, and is a generalization for other cases as well.

For example, you will notably see the `owned` convention on destructors and on
consuming move initializers, e.g., our `HeapArray` used that in its
`__del__` method - this is because you need to own a value to destroy it or to
steal its parts!

This is because you need to own a value to destroy it or to steal its parts!

### `@register_passable` struct decorator

As described above, the default fundamental model for working with values is
that they live in memory so they have identity, which means they are passed
indirectly to and from functions (equivalently, they are passed ‘by reference’
at the machine level). This is great for types that cannot be moved, and is a
good safe default for large objects or things with expensive copy operations.
However, it is really inefficient for tiny things like a single integer or
floating point number!

To solve this, Mojo allows structs to opt-in to being passed in a register
instead of passing through memory with the `@register_passable` decorator.
You’ll see this decorator on types like `Int` in the standard library:



```mojo
@register_passable("trivial")
struct MyInt:
   var value: Int

   fn __init__(value: Int) -> Self:
       return Self {value: value}

let x = MyInt(10)
```

The basic `@register_passable` decorator does not change the fundamental
behavior of a type: it still needs to have a `__copyinit__` method to be
copyable, may still have `__init__` and `__del__` methods, etc. The major
effect of this decorator is on internal implementation details:
`@register_passable` types are typically passed in machine registers (subject
to the details of the underlying architecture of course).

There are only a few observable effects of this decorator to the typical Mojo
programmer:

1. `@register_passable` types are not being able to hold instances of types
that are not themselves `@register_passable`.

2. instances of `@register_passable` types do not have predictable identity,
and so the ‘self’ pointer is not stable/predictable (e.g. in hash tables).

3. `@register_passable` arguments and result are exposed to C and C++ directly,
instead of being passed by-pointer.

4. The `__init__` and `__copyinit__` methods of this type are implicitly
static (like `__new__` in Python) and return its result by-value instead of
taking `inout self`.

We expect that this decorator will be used pervasively on core standard library
types, but is safe to ignore for general application level code.

The `MyInt` example above actually uses the `"trivial"` variant of this
decorator.  It changes the passing convention as described above but also
disallows copy and move constructors and destructors (synthesizing them all
trivially).

<a id='advanced-mojo-features'></a>

## Advanced Mojo features

This section describes power-user features that are important for building the
bottom-est level of the standard library. This level of the stack is inhabited
by narrow features that require experience with compiler internals to
understand and utilize effectively.

### `@always_inline` decorator

For implementing high-performant kernels it's often important to control
optimizations that compiler applies to the code. It's important to be able to
both enable optimizations we need and disable optimizations we do not want.
Traditional compilers usually rely on various heuristics to decide whether to
apply a given optimization or not (e.g. whether to inline a call or not, or
whether to unroll a loop or not). While this usually gives a decent baseline,
it's often unpredictable. That's why Mojo introduces special decorators that
provide full control over compiler optimizations.

The first decorator we'll demonstrate is `@always_inline`. It is used on a
function and instructs compiler to always inline this function when it's called.


```mojo
@always_inline
fn foo(x: Int, y: Int) -> Int:
    return x + y

fn bar(z: Int):
    let r = foo(z, z) # This call will be inlined
```

In future we will also introduce an opposite decorator, which would prevent
compiler from inlining a function, and similar decorators to control other
optimizations, such as loop unrolling.

### Direct access to MLIR

Mojo is not just built ontop of MLIR, it also provides a way to access it. This
allows integration with any hardware targets and also lets us make sure that the
code Mojo compiler produces is **exactly** what we want. This is extremely
important when we want to utilize hardware-specific features directly and not
rely on a compiler for that.

This feature is used, for instance, to back our `SIMD` type implementation. If
you'd like to learn more about it, you can take a look at the [Low-Level
IR](https://docs.modular.com/mojo/notebooks/BoolMLIR.html) notebook that gives a
taste of it.
