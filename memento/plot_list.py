import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)

pl = plt.plot([1,2,3], [3,2,1])
pl += plt.plot([1,2,3], [5,1,7])
pl += plt.plot([3,4,5], [1,6,4])
pl += plt.plot([3,4,5], [0,3,4])

plt.show()

p0 = pl[0]
p0.set_visible(False); fig.canvas.show()
p0.set_visible(True); fig.canvas.show()
p0.set_lw(5); fig.canvas.show()

p1 = pl[1]
p1.remove(); fig.canvas.show()
p1.set_visible(True); fig.canvas.show()


fig.canvas.show()


ax.add_line(p1)
plt.show()
