import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# create a rectangle
rect = patches.Rectangle((0.25,0.55), 0.4, 0.3, fill=False, ec='k')

# calculate the x and y points 
y = np.linspace(0,1,11)
x = np.linspace(0,1,11)

# create a list of possible coordinates
g = np.meshgrid(x, y)
coords = np.array(zip(*(c.flat for c in g)))

# create the list of valid coordinates (from untransformed)
rectpoints = np.vstack([p for p in coords \
                        if rect.contains_point(tuple(p), radius=0)])
outpoints = np.vstack([p for p in coords \
                       if not rect.contains_point(p, radius=0)])

# just to see if this works
fig = plt.figure()
ax = fig.add_subplot(111)
ax.add_artist(rect)
ax.plot(rectpoints[:,0], rectpoints[:,1], 'r.')
ax.plot(outpoints[:,0], outpoints[:,1], 'k.')
plt.show()


#
# Why this does not work????????????????????
#
fig1 = plt.figure()
ax1 = fig.add_subplot(111)
r1 = ax1.add_patch(patches.Rectangle((0.25,0.55), 0.4, 0.3, fill=False, ec='k'))

rp1 = np.vstack([p for p in coords if r1.contains_point(p, radius=0)])
op1 = np.vstack([p for p in coords \
                       if not r1.contains_point(p, radius=0)])

          
