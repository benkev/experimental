###########################################################################
###########################  CreateTips ###################################
###########################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else: # Python3
    import tkinter as tk



class CreateTips:
    """
    Imitate ToolTips or Baloons with text over a matplotlib figure with a plot.
    The tip appears near the mouse pointer after the first click on a
    graph point, and disappears after the second click.
    
    canvas: matplotlib figure  canvas
    """
    
    def __init__(self, panel):
        
        self.momPanel = panel              # The (plot) panel that called tips 
        self.canvas = panel.canvas

        # The Tk widget used to implement FigureCanvasTkAgg
        self.tkfig = self.canvas.get_tk_widget() # Tk widget figure is based on
        self.tw = None
        self.cidp = None


    def showtip(self, text):
        # Each second mouse key klick destroys the tooltip 
        if self.tw:
            self.tw.destroy()
            self.tw = None
            return

        # Get Tk absolute screen pixel coordinates: (0,0) is top left corner
        # from the Tk widget used to implement FigureCanvasTkAgg
        x, y = self.tkfig.winfo_pointerxy() # Tk (x,y) in pixels
        x += 15
        y += 10
        # creates a toplevel window
        self.tw = tk.Toplevel()
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
        if self.tw:
            self.tw.destroy()
            self.tw = None
        self.canvas.mpl_disconnect(self.cidp)
        self.cidp = None
        

    def on_pick(self, event):

        pname = self.momPanel.panelName
        
        obj = event.artist
        self.obj = event.artist

        if pname in ('Vis_Panel', 'Clopha_Panel', 'Cloamp_Panel'):
            
            ind = event.ind
            if len(ind) > 1:
                self.showtip('Ambiguous')
            else:
                ind = ind[0]
                self.momPanel.green_picked_and_draw_closure(ind)

        elif pname == 'TileRxPanel':

            ind = event.ind
            ind = ind[0]
            objname = obj.get_label()
            if objname == 'Rx':
                self.showtip('Rx: %d' % \
                             (ind+1))
            elif objname == 'Tile':  
                self.showtip('Ant: %d\nTile: %s' % \
                             (ind, str(1+ind//8)+str(1+ind%8)))

        elif pname == 'Selection_Histogram':

            self.momPanel.highlightBarSelected(obj)






 
