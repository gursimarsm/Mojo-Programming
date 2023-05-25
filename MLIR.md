# Low-level IR in Mojo

Mojo is a high-level programming language with an extensive set of modern features. But Mojo also provides you, the programmer, access to all of the low-level primitives that you need to write powerful -- yet zero-cost -- abstractions.

These primitives are implemented in [MLIR](https://mlir.llvm.org), an extensible intermediate representation (IR) format for compiler design. Many different programming languages and compilers translate their source programs into MLIR, and because Mojo provides direct access to MLIR features, this means Mojo programs can enjoy the benefits of each of these tools.

Going one step further, Mojo's unique combination of zero-cost abstractions with MLIR interoperability means that Mojo programs can take full advantage of *anything* that interfaces with MLIR. While this isn't something normal Mojo programmers may ever need to do, it's an extremely powerful capability when extending a system to interface with a new datatype, or an esoteric new accelerator feature.

To illustrate these ideas, we'll implement a boolean type in Mojo below, which we'll call `OurBool`. We'll make extensive use of MLIR, so let's begin with a short primer.


## What is MLIR?

MLIR is an intermediate representation of a program, not unlike an assembly language, in which a sequential set of instructions operate on in-memory values.

More importantly, MLIR is modular and extensible. MLIR is composed of an ever-growing number of "dialects." Each dialect defines operations and optimizations: for example, the ['math' dialect](https://mlir.llvm.org/docs/Dialects/MathOps/) provides mathematical operations such as sine and cosine, the ['amdgpu' dialect](https://mlir.llvm.org/docs/Dialects/AMDGPU/) provides operations specific to AMD processors, and so on.

Each of MLIR's dialects can interoperate with the others. This is why MLIR is said to unlock heterogeneous compute: as newer, faster processors and architectures are developed, new MLIR dialects are implemented to generate optimal code for those environments. Any new MLIR dialect can be translated seamlessly into other dialects, so as more get added, all existing MLIR becomes more powerful.

This means that our own custom types, such as the `OurBool` type we'll create below, can be used to provide programmers with a high-level, Python-like interface. But "under the covers," Mojo and MLIR will optimize our convenient, high-level types for each new processor that appears in the future.

There's much more to write about why MLIR is such a revolutionary technology, but let's get back to Mojo and defining the `OurBool` type. There will be opportunities to learn more about MLIR along the way.

## Defining the `OurBool` type

We can use Mojo's `struct` keyword to define a new type `OurBool`:


```mojo
struct OurBool:
    var value: __mlir_type.i1
```

A boolean can represent 0 or 1, "true" or "false." To store this information, `OurBool` has a single member, called `value`. Its type is represented *directly in MLIR*, using the MLIR builtin type [`i1`](https://mlir.llvm.org/docs/Dialects/Builtin/#integertype). In fact, you can use any MLIR type in Mojo, by prefixing the type name with `__mlir_type`.

As we'll see below, representing our boolean value with `i1` will allow us to utilize all of the MLIR operations and optimizations that interface with the `i1` type -- and there are many of them!

Having defined `OurBool`, we can now declare a variable of this type:


```mojo
let a: OurBool
```

## Leveraging MLIR

Naturally, we might next try to create an instance of `OurBool`. Attempting to do so at this point, however, results in an error:

```mojo
let a = OurBool() # error: 'OurBool' does not implement an '__init__' method
```

As in Python, `__init__` is a [special method](https://docs.python.org/3/reference/datamodel.html#specialnames) that can be defined to customize the behavior of a type. We can implement an `__init__` method that takes no arguments, and returns an `OurBool` with a "false" value.


```mojo
struct OurBool:
    var value: __mlir_type.i1

    fn __init__(inout self):
        self.value = __mlir_op.`index.bool.constant`[
            value : __mlir_attr.`false`,
        ]()
```

To initialize the underlying `i1` value, we use an MLIR operation from its ['index' dialect](https://mlir.llvm.org/docs/Dialects/IndexOps/), called [`index.bool.constant`](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexboolconstant-mlirindexboolconstantop).

MLIR's 'index' dialect provides us with operations for manipulating builtin MLIR types, such as the `i1` we use to store the value of `OurBool`. The `index.bool.constant` operation takes a `true` or `false` compile-time constant as input, and produces a runtime output of type `i1` with the given value.

So, as shown above, in addition to any MLIR type, Mojo also provides direct access to any MLIR operation via the `__mlir_op` prefix, and to any attribute via the `__mlir_attr` prefix. MLIR attributes are used to represent compile-time constants.

As you can see above, the syntax for interacting with MLIR isn't always pretty: MLIR attributes are passed in between square brackets `[...]`, and the operation is executed via a parentheses suffix `(...)`, which can take runtime argument values. However, most Mojo programmers will not need to access MLIR directly, and for the few that do, this "ugly" syntax gives them superpowers: they can define high-level types that are easy to use, but that internally plug into MLIR and its powerful system of dialects.

We think this is very exciting, but let's bring things back down to earth: having defined an `__init__` method, we can now create an instance of our `OurBool` type:


```mojo
let b = OurBool()
```

## Value semantics in Mojo

We can now instantiate `OurBool`, but using it is another story:

```mojo
let a = OurBool()
let b = a # error: 'OurBool' does not implement the '__copyinit__' method
```

Mojo uses "value semantics" by default, meaning that it expects to create a copy of `a` when assigning to `b`. However, Mojo doesn't make any assumptions about *how* to copy `OurBool`, or its underlying `i1` value. The error indicates that we should implement a `__copyinit__` method, which would implement the copying logic.

In our case, however, `OurBool` is a very simple type, with only one "trivially copyable" member. We can use a decorator to tell the Mojo compiler that, saving us the trouble of defining our own `__copyinit__` boilerplate. Trivially copyable types must implement an `__init__` method that returns an instance of themselves, so we must also rewrite our initializer slightly.


```mojo
@register_passable("trivial")
struct OurBool:
    var value: __mlir_type.i1

    fn __init__() -> Self:
        return Self {
            value: __mlir_op.`index.bool.constant`[
                value : __mlir_attr.`false`,
            ]()
        }
```

We can now copy `OurBool` as we please:


```mojo
let c = OurBool()
let d = c
```

## Compile-time constants

It's not very useful to have a boolean type that can only represent "false." Let's define compile-time constants that represent true and false `OurBool` values.

First, let's define another `__init__` constructor for `OurBool` that takes its `i1` value as an argument:


```mojo
@register_passable("trivial")
struct OurBool:
    var value: __mlir_type.i1

    # ...

    fn __init__(value: __mlir_type.i1) -> Self:
        return Self {value: value}
```

This allows us to define compile-time constant `OurBool` values, using the `alias` keyword. First, let's define `OurTrue`:


```mojo
alias OurTrue = OurBool(__mlir_attr.`true`)
```

Here we're passing in an MLIR compile-time constant value of `true`, which has the `i1` type that our new `__init__` constructor expects. We can use a slightly different syntax for `OurFalse`:


```mojo
alias OurFalse: OurBool = __mlir_attr.`false`
```

`OurFalse` is declared to be of type `OurBool`, and then assigned an `i1` type -- in this case, the `OurBool` constructor we added is called implicitly.

With true and false constants, we can also simplify our original `__init__` constructor for `OurBool`. Instead of constructing an MLIR value, we can simply return our `OurFalse` constant:


```mojo
alias OurTrue = OurBool(__mlir_attr.`true`)
alias OurFalse: OurBool = __mlir_attr.`false`


@register_passable("trivial")
struct OurBool:
    var value: __mlir_type.i1

    # We can simplify our no-argument constructor:
    fn __init__() -> Self:
        return OurFalse

    fn __init__(value: __mlir_type.i1) -> Self:
        return Self {value: value}
```

Note also that we can define `OurTrue` before we define `OurBool`. The Mojo compiler is smart enough to figure this out.

With these constants, we can now define variables with both true and false values of `OurBool`:


```mojo
let e = OurTrue
let f = OurFalse
```

## Implementing `__bool__`

Of course, the reason booleans are ubiquitous in programming is because they can be used for program control flow. However, if we attempt to use `OurBool` in this way, we get an error:

```mojo
let a = OurTrue
if a: print("It's true!") # error: 'OurBool' does not implement the '__bool__' method
```

When Mojo attempts to execute our program, it needs to be able to determine whether to print "It's true!" or not. It doesn't yet know that `OurBool` represents a boolean value -- Mojo just sees a struct that is 1 bit in size. However, Mojo also provides interfaces that convey boolean qualities, which are the same as those used by Mojo's standard library types, like `Bool`. In practice, this means Mojo gives you full control: any type that's packaged with the language's standard library is one for which you could define your own version.

In the case of our error message, Mojo is telling us that implementing a `__bool__` method on `OurBool` would signify that it has boolean qualities.

Thankfully, `__bool__` is simple to implement: Mojo's standard library and builtin types are all implemented on top of MLIR, and so the builtin `Bool` type also defines a constructor that takes an `i1`, just like `OurBool`:


```mojo
alias OurTrue = OurBool(__mlir_attr.`true`)
alias OurFalse: OurBool = __mlir_attr.`false`


@register_passable("trivial")
struct OurBool:
    var value: __mlir_type.i1

    # ...

    fn __init__(value: __mlir_type.i1) -> Self:
        return Self {value: value}

    # Our new method converts `OurBool` to `Bool`:
    fn __bool__(self) -> Bool:
        return Bool(self.value)
```

Now we can use `OurBool` anywhere we can use the builtin `Bool` type:


```mojo
let g = OurTrue
if g: print("It's true!")
```

    It's true!


## Avoiding type conversion with `__mlir_i1__`

Our `OurBool` type is looking great, and by providing a conversion to `Bool`, it can be used anywhere the builtin `Bool` type can. But in the last section we promised you "full control," the ability to define your own version of any type built into Mojo or its standard library. Surely `Bool` doesn't implement `__bool__` to convert itself into `Bool`?

Indeed it doesn't: when Mojo evaluates a conditional expression, it actually attempts to convert it to an MLIR `i1` value, by searching for the special interface method `__mlir_i1__`. (The automatic conversion to `Bool` occurs because `Bool` is known to implement the `__mlir_i1__` method.)

Again, Mojo is designed to be extensible and modular. By implementing all the special methods `Bool` does, we can create a type that can replace it entirely. Let's do so by implementing `__mlir_i1__` on `OurBool`:


```mojo
alias OurTrue = OurBool(__mlir_attr.`true`)
alias OurFalse: OurBool = __mlir_attr.`false`


@register_passable("trivial")
struct OurBool:
    var value: __mlir_type.i1

    fn __init__(value: __mlir_type.i1) -> Self:
        return Self {value: value}

    # ...

    # Our new method converts `OurBool` to `i1`:
    fn __mlir_i1__(self) -> __mlir_type.i1:
        return self.value
```

We can still use `OurBool` in conditionals just as we did before:


```mojo
let h = OurTrue
if h: print("No more Bool conversion!")
```

    No more Bool conversion!


But this time, no conversion to `Bool` occurs. You can try adding `print` statements to the `__bool__` and `__mlir_i1__` methods, or even removing the `__bool__` method entirely, to see for yourself.

## Adding functionality with MLIR

There are many more ways we can improve `OurBool`. Many of those involve implementing special methods, some of which you may recognize from Python, and some which are specific to Mojo. For example, we can implement inversion of a `OurBool` value by adding a `__invert__` method. We can also add an `__eq__` method, which allows two `OurBool` to be compared with the `==` operator.

What sets Mojo apart is the fact that we can implement each of these using MLIR. To implement `__eq__`, for example, we use the [`index.casts`](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexcasts-mlirindexcastsop) operation to cast our `i1` values to the MLIR index dialect's `index` type, and then the [`index.cmp`](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexcmp-mlirindexcmpop) operation to compare them for equality. And with the `__eq__` method implemented, we can then implement `__invert__` in terms of `__eq__`:


```mojo
alias OurTrue = OurBool(__mlir_attr.`true`)
alias OurFalse: OurBool = __mlir_attr.`false`


@register_passable("trivial")
struct OurBool:
    var value: __mlir_type.i1

    fn __init__(value: __mlir_type.i1) -> Self:
        return Self {value: value}

    # ...

    fn __mlir_i1__(self) -> __mlir_type.i1:
        return self.value

    fn __eq__(self, rhs: OurBool) -> Self:
        let lhsIndex = __mlir_op.`index.casts`[_type : __mlir_type.index](
            self.value
        )
        let rhsIndex = __mlir_op.`index.casts`[_type : __mlir_type.index](
            rhs.value
        )
        return Self(
            __mlir_op.`index.cmp`[
                pred : __mlir_attr.`#index<cmp_predicate eq>`
            ](lhsIndex, rhsIndex)
        )

    fn __invert__(self) -> Self:
        return OurFalse if self == OurTrue else OurTrue
```

This allows us to use the `~` operator with `OurBool`:


```mojo
let i = OurFalse
if ~i: print("It's false!")
```

    It's false!


This extensible design is what allows even "built in" Mojo types like `Bool`, `Int`, and even `Tuple` (!!) to be implemented in the Mojo standard library in terms of MLIR, rather than hard-coded into the Mojo language. This also means that there's almost nothing that those types can achieve that user-defined types cannot.

By extension, this means that the incredible performance that Mojo unlocks for machine learning workflows isn't due to some magic being performed behind a curtain -- you can define your own high-level types that, in their implementation, use low-level MLIR to achieve unprecedented speed and control.

## The promise of modularity

As we've seen, Mojo's integration with MLIR allows Mojo programmers to implement zero-cost abstractons on par with Mojo's own builtin and standard library types.

MLIR is open-source and extensible: new dialects are being added all the time, and those dialects then become available to use in Mojo. All the while, Mojo code gets more powerful and more optimized for new hardware -- with no additional work necessary by Mojo programmers.

What this means is that your own custom types, whether those be `OurBool` or `OurTensor`, can be used to provide programmers with an easy-to-use and unchanging interface. But behind the scenes, MLIR will optimize those convenient, high-level types for the computing environments of tomorrow.

In other words: Mojo isn't magic, it's modular.
