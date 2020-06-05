from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import Tkinter as tk
import time

global tw

tw = None

class CreateTip:
    """
    create a tooltip for a given widget
    """
    def __init__(self, text='widget info'):
        self.waittime = 300     #milliseconds
        self.wraplength = 180   #pixels
        self.text = text
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()

    def unschedule(self):
        id = self.id
        self.id = None
 
    def showtip(self):
        # Tk absolute screen pixel coordinates: (0,0) is top left corner
        x, y = root.winfo_pointerxy()
        x += 25
        y += 20
        # creates a toplevel window
        self.tw = tk.Toplevel()
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffff", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()


def showtip(fig, text):
    global tw
    # Each second mouse key klick destroys the tooltip 
    if tw:
        tw.destroy()
        tw = None
        return
    
    # Get Tk absolute screen pixel coordinates: (0,0) is top left corner
    # from the Tk widget used to implement FigureCanvasTkAgg
    tkfig = fig.canvas.get_tk_widget()   # the Tk widget figure is based on
    x, y = tkfig.winfo_pointerxy()       # Tk (x,y) in pixels
    x += 25
    y += 20
    # creates a toplevel window
    tw = tk.Toplevel()
    # Leaves only the label and removes the app window
    tw.wm_overrideredirect(True)
    tw.wm_geometry("+%d+%d" % (x, y))
    label = tk.Label(tw, text=text, justify='left',
                   background="#ffffff", relief='solid', borderwidth=1,
                   wraplength = 180)
    label.pack(ipadx=1)



def hidetip(tw):
    if tw:
        tw.destroy()


def on_pick(event):
    line = event.artist
    xdata, ydata = line.get_data()
    ind = event.ind
    print 'on pick line:', np.array([xdata[ind], ydata[ind]]).T
    #print 'Tk screen pixel coordinates: ', root.winfo_pointerxy()
    
    #tip = CreateTip('x[%d]=%g, y[%d]=%g' % (ind, xdata[ind], ind, ydata[ind]))
    #tip.showtip()
    #time.sleep(0.5)
    #tip.hidetip()
    
    showtip(fig, 'x[%d]=%g, y[%d]=%g' % (ind, xdata[ind], ind, ydata[ind]))


def on_motion(event):
    pass
def on_axes_enter(event):
    print 'Axes entered: ', event.inaxes 
def on_axes_leave(event):
    print 'Axes left: ', event.inaxes 



fig = figure()
ax = subplot(111)
ax.plot(rand(100), 'o', picker=5)  # 5 points tolerance
grid(1)

show()




cidp = fig.canvas.mpl_connect('pick_event', on_pick)
#cidm = fig.canvas.mpl_connect('motion_notify_event', on_motion)
#cidae = fig.canvas.mpl_connect('axes_enter_event', on_axes_enter)
#cidal = fig.canvas.mpl_connect('axes_leave_event', on_axes_leave)



