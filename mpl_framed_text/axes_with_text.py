from pylab import *

f = figure()
ax = f.add_subplot(111)
ax.plot(randn(10), lw=8., color='b')
ax.grid(1)

tx = ax.text(2.3, 0.2, 'fontsize=12\nHow many lines?\nThree lines.', color='k',\
             bbox=dict(facecolor='w', edgecolor='k', linewidth=0.7, \
                       boxstyle='round,pad=0.3'))


f.show()










