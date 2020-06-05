"""
Since we are creating a bitcoin trading application, it only makes sense that
we're going to have to incorporate some price data. Not only do we want to just
plot the prices, but many people will want to see prices in the form of OHLC
candlesticks, and then others will also want to see various indicators like
EMA/SMA crossovers and things like RSI or MACD.

To do this, we first need to know how to actually embed a Matplotlib graph into
a Tkinter application. Here's how!

First, we're going to be using Matplotlib.

Now we're going to need to add the following imports to our Tkinter application:
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
     NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import Tkinter as tk
import ttk            # In Python 2.7 ttk is its own package

#
# Now let's go ahead and add a new page. This will be our "graph" page. 
#
class PageThree(tk.Frame):
    """
    Typical stuff here, with a navigation back to head back to main page.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        #
        # Here's where our embedding code begins.
        # First we are defining our figure then adding a subplot.
        # From there, we plot as usual some x coordinates and some y.
        #
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])
        #
        # Next, we add the canvas, which is what we intend
        # to render the graph to.
        #
        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #
        # Finally, we add the toolbar, which is the traditional
        # matplotlib tool bar. 
        #
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        #
        # From there, we then pack all of this to our tkinter window. 
        #
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


