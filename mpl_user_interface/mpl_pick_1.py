from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import Tkinter as tk
import time

global tw

tw = None

class CreateTips:
    """
    fig: matplotlib figure
    text: tooltip text
    """
    
    def __init__(self, fig):
        # The Tk widget used to implement FigureCanvasTkAgg
        self.tkfig = fig.canvas.get_tk_widget() # Tk widget figure is based on
        self.cidp = fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.tw = None


    def showtip(self, text):
        # Each second mouse key klick destroys the tooltip 
        if self.tw:
            self.tw.destroy()
            self.tw = None
            return

        # Get Tk absolute screen pixel coordinates: (0,0) is top left corner
        # from the Tk widget used to implement FigureCanvasTkAgg
        x, y = self.tkfig.winfo_pointerxy() # Tk (x,y) in pixels
        x += 25
        y += 20
        # creates a toplevel window
        self.tw = tk.Toplevel()
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=text, justify='left',
                         background="#ffffff", relief='solid', borderwidth=1,
                         wraplength = 180)
        label.pack(ipadx=1)



    def enable(self):
        if self.cidp:
            return
        self.cidp = fig.canvas.mpl_connect('pick_event', self.on_pick)

        
    def disable(self):
        if self.tw:
            self.tw.destroy()
            self.tw = None
        fig.canvas.mpl_disconnect(self.cidp)
        self.cidp = None
        

    def on_pick(self, event):
        line = event.artist
        xdata, ydata = line.get_data()
        ind = event.ind
        self.showtip('x[%d]=%g, y[%d]=%g' % (ind, xdata[ind], ind, ydata[ind]))




fig = figure()
ax = subplot(111)
ax.plot(rand(100), 'o', picker=5)  # 5 points tolerance
grid(1)

ttip = CreateTips(fig)
ttip.enable()

show()



