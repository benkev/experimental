from Tkinter import *

root = Tk()

def hello():
    print "hello!"

# create a toplevel menu

## hContMenuBar = Frame(root)
## menubar = Menu(hContMenuBar)
## hContMenuBar.pack()

menubar = Menu(root)

# create a pulldown menu, and add it to the menu bar
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=hello)
filemenu.add_separator() # -----------------------
# Submenu
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Cut", command=hello)
editmenu.add_command(label="Copy", command=hello)
editmenu.add_command(label="Paste", command=hello)
filemenu.add_cascade(label="Edit", menu=editmenu)
filemenu.add_separator() # -----------------------

filemenu.add_command(label="Save", command=hello)
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

# Submenu
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=hello)
filemenu.add_cascade(label="Help", menu=helpmenu)

# display the menu
root.config(menu=menubar)

