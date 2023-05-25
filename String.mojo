from String import String
let s : String = "hello world"
print(s[0])
print(s[:5])
print(s[::2])
print(String("hello world")[0])

# The following code doesn’t work because `StringRef` cannot be subscripted!
let s = "hello world" # Here s is a `StringRef`
print(s[:5])
print("hello world"[0])
