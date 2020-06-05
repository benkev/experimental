from Tkinter import *


class RectTracker:
	
	def __init__(self, canvas):
		self.canvas = canvas
		self.item = None
		
	def draw(self, start, end, **opts):
		"""Draw the rectangle"""
		return self.canvas.create_rectangle(*(list(start)+list(end)), **opts)
		
	def autodraw(self, **opts):
		"""Setup automatic drawing; supports command option"""
		self.start = None
		self.canvas.bind("<Button-1>", self.__update, '+')
		self.canvas.bind("<B1-Motion>", self.__update, '+')
		self.canvas.bind("<ButtonRelease-1>", self.__stop, '+')
		
		self._command = opts.pop('command', lambda *args: None)
		self.rectopts = opts
		
	def __update(self, event):
		if not self.start:
			self.start = [event.x, event.y]
			return
		
		if self.item is not None:
			self.canvas.delete(self.item)
		self.item = self.draw(self.start, (event.x, event.y), **self.rectopts)
		self._command(self.start, (event.x, event.y))
		
	def __stop(self, event):
		self.start = None
		self.canvas.delete(self.item)
		self.item = None

#===================================================================


		
def onDrag(start, end):                  # Callback for rect.autodraw()
    pass
    #global x,y
    #items = rect.hit_test(start, end)
    #for x in rect.items:
    #    if x not in items:
    #        canv.itemconfig(x, fill='grey')
    #    else:
    #        canv.itemconfig(x, fill='blue')

#---------------------------------------------------------------------


root = Tk()
canv = Canvas(root, width=500, height=500)
#rect = canv.create_rectangle(50, 50, 250, 150, fill='red')
#
# .pack():
# fill=NONE or X or Y or BOTH - fill widget if widget grows
# expand=bool - expand widget if parent size grows
#
canv.pack(fill=BOTH, expand=YES)

rect = RectTracker(canv)

rect.autodraw(fill="", width=1, command=onDrag)
















