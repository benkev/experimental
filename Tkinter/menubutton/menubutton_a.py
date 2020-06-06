import Tkinter as tk

root = tk.Tk()

mb = tk.Menubutton(root, text='condiments', relief='raised')

mb.menu = tk.Menu(mb, tearoff=0)
mb['menu'] = mb.menu

mayoVar  = tk.IntVar()
ketchVar = tk.IntVar()

mb.menu.add_checkbutton(label='mayo',    variable=mayoVar)
mb.menu.add_checkbutton(label='ketchup', variable=ketchVar)

mb.pack()

