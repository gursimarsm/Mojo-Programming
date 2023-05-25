# Interact with Python
## Execute Python code directly in the playground

```
%%python
print("The number is", 36)
```

## Invoke Python interpreter from Mojo

``` 
let x: Int = 36

from PythonInterface import Python
let py = Python.import_module("builtins")

py.print("The number is", x)
```
