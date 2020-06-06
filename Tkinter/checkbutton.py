from Tkinter import *

master = Tk()

var = IntVar()

MODES = [
    ("M", "1"),
    ("G", "L"),
    ("T", "RGB"),
    ("C", "CMYK"),
]

#
# There MUST be separate variables for each Checkbutton
# Otherwise, the result will be absurd. Try to run this script.
#
v = StringVar()
v.set("L") # initialize

for text, mode in MODES:
    b = Checkbutton(master, text=text, variable=v)
    b.pack(anchor=W)

