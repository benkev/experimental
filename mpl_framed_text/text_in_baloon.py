from pylab import *

class Baloon:
    
    def __init__(self, fig, ax):
        
        self.canvas = fig.canvas
        self.fig = fig
        self.ax = ax
        self.patch = None         # Instead of Tk widgets, use patches


    def showtip(self, text):
        # Each second mouse key klick destroys the tooltip 
        if self.patch:
            self.patch.remove()
            self.patch = None
            return

        # Get the canvas pixel coordinates: (0,0) is lower left corner
        x, y = self.x, self.y
        ## x += 10
        ## y += 10
        nl = text.count('\n') + 1  # How many lines in the baloon text
        
        # Creates a patch object
        self.patch = patches.Rectangle((x,y), wid, hei, ec='k', fill=True, \
                                       fc='#f0f0f0', \
                                       ls='solid', lw=0.7)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=text, justify='left',
                         background="#ffffff", relief='solid', borderwidth=1,
                         wraplength = 300)
        label.pack(ipadx=1)



    def enable(self):
        if self.cidp:
            return
        self.cidp = self.canvas.mpl_connect('pick_event', self.on_pick)

        
    def disable(self):
        if self.patch:
            self.patch.remove()
            self.patch = None
        self.canvas.mpl_disconnect(self.cidp)
        self.cidp = None
        

    def on_pick(self, event):

        # Store the canvas pixel coordinates of the mouse pointer
        self.x = event.mouseevent.x
        self.y = event.mouseevent.y

        ind = event.ind
        
        print ind, x, y
        
        obj = event.artist
        self.obj = obj




        
f = figure()
ax = f.add_subplot(111)
ax.plot(randn(10), lw=8., color='b')
ax.grid(1)

b = Baloon(fig, ax)

p = patches.Rectangle((5,0.5), 2., 0.25, ec='k', fill=True, fc='#ffffff', \
                      ls='solid', lw=0.7, zorder=10)

ax.add_patch(p)

f.show()

