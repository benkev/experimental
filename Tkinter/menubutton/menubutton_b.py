import Tkinter as tk

root = tk.Tk()

mb = tk.Menubutton(root, text='condiments', relief='raised')

picks = tk.Menu(mb, tearoff=False)               
mb.config(menu=picks)

mayoVar  = tk.IntVar()
ketchVar = tk.IntVar()
mayoVar1  = tk.IntVar()
ketchVar1 = tk.IntVar()

picks.add_checkbutton(label='mayo',    variable=mayoVar)
picks.add_checkbutton(label='ketchup', variable=ketchVar)

picks1 = tk.Menu(picks, tearoff=False)
picks.add_cascade(label="Help", menu=picks1)
picks1.add_checkbutton(label='mayo',    variable=mayoVar1)
picks1.add_checkbutton(label='ketchup', variable=ketchVar1)

picks2 = tk.Menu(picks1, tearoff=False)
picks1.add_cascade(label="Interfere", menu=picks2)
picks2.add_command(label='flagg')
picks2.add_command(label='bunner')

picks3 = tk.Menu(picks2, tearoff=False)
picks2.add_cascade(label="Drop", menu=picks3)
picks3.add_command(label='crieg')
picks3.add_command(label='asdfvc')

picks2.add_command(label='troub')
picks2.add_command(label='sackar')


mb.pack()
 
