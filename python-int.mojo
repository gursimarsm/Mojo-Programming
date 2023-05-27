from PythonObject import PythonObject
alias int = PythonObject

let x : int = 2
print(x ** 100) # `int` does not overflow: 1267650600228229401496703205376
print(2 ** 100) # `Int` overflows: 0
