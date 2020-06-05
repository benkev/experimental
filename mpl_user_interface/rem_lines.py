import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

x = np.linspace(-2,2,21)
y1 = np.exp(x)
y2 = np.sin(2*x)

ax.plot(x,y1, label='exp')

pl = ax.plot(x,y2, label='sin'); ax.grid(1)




fig.show()
#fig.canvas.show()


pl = pl[0]


#
# To remove the last 2D line:
#
#pl.remove()
#fig.canvas.show()
