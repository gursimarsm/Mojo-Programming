
from PythonInterface import Python

fn create_matrix(rows : Int, cols : Int) raises -> PythonObject:
    # This is equivalent to Python's `import numpy as np`
    let np = Python.import_module("numpy")

    # Create matrix with numpy:
    let matrix = np.arange(1, rows*cols+1).reshape(rows, cols)

    return matrix

# Create matrix 9 rows and 5 cols:
print(create_matrix(9, 5))
