import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
                                              NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches

import sys
import Tkinter as tk

wd = tk.Tk()

wd.title("Selection Histogram")

fig = Figure(figsize=(14,3), dpi=100)
ax = fig.add_subplot(111)

canvas = FigureCanvasTkAgg(fig, wd)
canvas.show()

toolbar = NavigationToolbar2TkAgg(canvas, wd)
toolbar.update()
canvas._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
#canvas._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

tkfig = canvas.get_tk_widget()
tkfig.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

x = np.load('x.npy')
y = np.load('y.npy')

nx = len(x)

#
# Plot histogram
#
xaxis = np.arange(nx)
ax.bar(x-0.45, y, 0.9, color='tan', lw=0.5)
ax.set_xlim(-1,128);
ax.set_ylim(-0.5, y.max()+1.0);
ax.set_xticks(x)
ax.grid(True)
ax.xaxis.grid(False)     # Remove vertical grid lines
fig.tight_layout(pad=2.)

ax.set_title('Antenna Occurrences in Selected Closures')
ax.set_ylabel('Antenna Occurrences')
ax.set_xticks(x)
#ax.set_xticklabels(x)
canvas.show()










