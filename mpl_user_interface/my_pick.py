from pylab import *
import numpy as np
import matplotlib.pyplot as plt

global xfig, yfig, xyfig

fig = figure()
ax = fig.add_subplot(111)

plis = plot(rand(10), 'b.'); grid(1)

fig.canvas.show()


def on_draw(event):
    """ Get precise line coordinates """
    global xfig, yfig, xyfig
    ax = event.canvas.figure.get_axes()[0]   # Get 0th from list of axes
    line = ax.lines[0]                       # Get 0th from list of lines
    xydat = line.get_xydata()
    xyfig = ax.transData.transform(xydat)
    xfig = xyfig[:,0]
    yfig = xyfig[:,1]

    
def on_click(event):
    # In figure coordinates
    x = event.x
    y = event.y
    d2 = (xfig-x)**2 + (yfig-y)**2
    print x, y
    print np.where(d2 < 100)[0]
    print 'xfig = ', xfig
    print 'yfig = ', yfig
    #print yfig-y
    print 'd = ', sqrt(d2)


ce = fig.canvas.mpl_connect('button_press_event', on_click)
cd = fig.canvas.mpl_connect('draw_event', on_draw)

show()


# fig.canvas.mpl_disconnect(ce)
# fig.canvas.mpl_disconnect(cd)

