import Tkinter as tk

def select():
    print 'Stop'
    mb.config(bg=bgcolor, text='Select Points')
    selOn = False
    unselOn = False

def selecting():
    print 'Selecting'
    mb.config(bg='green', text='Selecting')
    selOn = True

def unselecting():
    print 'UnSelecting'
    mb.config(bg='khaki', text='UnSelecting')
    unselOn = True

def undo():
    print 'Undo'

def redo():
    print 'Redo'


selOn = False
unselOn = False

root = tk.Tk()

mb = tk.Menubutton(root, text='Select Points')
picks   = tk.Menu(mb, tearoff=False)               
mb.config(menu=picks)           

picks.add_command(label='Select',  command=selecting)
picks.add_command(label='UnSelect',  command=unselecting)
picks.add_command(label='<- Undo', command=undo)
picks.add_command(label='-> Redo', command=redo)
picks.add_command(label='Stop',  command=select)

mb.pack()
mb.config(bd=1, relief='raised')
bgcolor = mb.cget('background')

picks.entryconfig(2, foreground='#8f8f8f')
picks.entryconfig(3, foreground='#8f8f8f')



