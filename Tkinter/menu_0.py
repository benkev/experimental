from Tkinter import *

root = Tk()

def hello():
    print "hello!"

# create a toplevel menu
menubar = Menu(root)

menubar.add_command(label="Button", command=hello)

root.config(menu=menubar)


