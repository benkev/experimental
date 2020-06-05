import Tkinter as tk
#from Tkinter import Frame
from HoverInfo import HoverInfo

class MyApp(tk.Frame):
    def __init__(self, parent=None):
        tk.Frame.__init__(self, parent)
        self.grid()
        self.lbl = tk.Label(self, text='testing')
        self.lbl.grid()
 
        self.hover = HoverInfo(self, 'while hovering press return \n ' \
                             'for an exciting msg', self.HelloWorld)
 
    def HelloWorld(self):
        print 'Hello World'
 
app = MyApp()
app.master.title('test')
