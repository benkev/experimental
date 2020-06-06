from Tkinter import *

def fun(x):
    print 'Option ', x

def fun2(x):
    print 'Option ', x

master = Tk()

variable = StringVar(master)
variable.set("one") # default value

w = OptionMenu(master, variable, "one", "two", "three", command=fun)
w.pack(side='left')

var2 = StringVar(master)
var2.set("Monday") # default value

w2 = OptionMenu(master, var2, "Monday", "Tuesday", "Wednesday", command=fun2)
w2.pack(side='left')

