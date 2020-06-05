from mpl_toolkits.mplot3d import Axes3D
from pylab import *

x = array((-1, 0, 1, 2), dtype=float)
y = array((-1, 0, 1, 2), dtype=float)
xp, yp = meshgrid(x, y)
xp = xp.flatten()
yp = yp.flatten()
xi = linspace(-1, 2, 20)
yi = linspace(-1, 2, 20)
xpi, ypi = meshgrid(xi, yi)
xpi = xpi.flatten()
ypi = ypi.flatten()

tsp = loadtxt('tsp.txt')
tspi = loadtxt('tspi.txt')
z = tsp.reshape(4,4)
zi = tspi.reshape((20,20))

dx = 0.05 * np.ones_like(tsp)
dy = dx.copy()
dz = 0.01*ones_like(tsp)
zer = zeros_like(tsp)


fig = figure()
ax = fig.gca(projection='3d')
ax.bar3d(xp, yp, tsp, dx, dy, dz, color='b', zsort='average')

ax.plot_wireframe(xpi, ypi, zi, rstride=1, cstride=1)


show()








 
