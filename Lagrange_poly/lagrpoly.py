import numpy as np
import matplotlib.pyplot as pl

x = np.array((0.5, 1.2, 3.1), dtype=float)
y = np.array((2.5, 0.5, 7.3), dtype=float)

t = np.linspace(-1., 4., 100)

a = (y[2] - (x[2]*(y[1] - y[0]) + x[1]*y[0] - x[0]*y[1])/(x[1] - x[0])) /    \
    (x[2]*(x[2] - x[0] - x[1]) + x[0]*x[1])

b = (y[1] - y[0])/(x[1] - x[0]) - a*(x[0] + x[1])

c = (x[1]*y[0] - x[0]*y[1])/(x[1] - x[0]) + a*x[0]*x[1]

v = a*t**2 + b*t + c

pl.plot(t, v, 'b')
pl.plot(x, y, 'or')

pl.title('Проведение параболы через 3 точки')
pl.xlabel('x')
pl.ylabel('y')

pl.show()




