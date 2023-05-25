# Creating a plot with matplotlib

def make_plot(m: Matrix):

  plt = Python.import_module("matplotlib.pyplot")

  fig = plt.figure(1, [10, 10 * yn // xn], 64)
  ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], False, 1)
  plt.imshow(image)
  plt.show()

make_plot(compute_mandelbrot())
