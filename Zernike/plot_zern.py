from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import zern

pi = np.pi

ph, rh = np.meshgrid(np.linspace(-pi,pi,101), np.linspace(0,1,101))
X, Y = rh*np.cos(ph), rh*np.sin(ph)



Z = zern.zernike(5, 1, rh, ph)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
 
Z = zern.zernike(5, 3, rh, ph)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
 
Z = zern.zernike(5, 5, rh, ph)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
 
plt.show()
