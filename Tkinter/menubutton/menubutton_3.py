from Tkinter import *

root    = Tk()

mbutton = Menubutton(root, text='Food')     # the pull-down stands alone
picks   = Menu(mbutton, tearoff=False)               
mbutton.config(menu=picks)           

picks.add_command(label='Bread',  command=root.quit)
picks.add_command(label='Butter',  command=root.quit)
picks.add_command(label='Juice', command=root.quit)
mbutton.pack(side='left')
mbutton.config(bg='tan', bd=1, relief=RAISED)


#lb = Label(root, text='/', fg='brown', font=("Helvetica", 20))
lb = Label(root, text='/', fg='brown', font=(None, 20))
lb.pack(side='left')


mb2 = Menubutton(root, text='Utencils')     # the pull-down stands alone
p2   = Menu(mb2, tearoff=False)               
mb2.config(menu=p2)           

p2.add_command(label='Spoon',  command=root.quit)
p2.add_command(label='Fork',  command=root.quit)
p2.add_command(label='Knife', command=root.quit)
mb2.pack(side='left')
mb2.config(bg='orange', bd=1, relief=RAISED)

#
# Get index of menu item 'Fork'
#

ixFork = p2.index('Fork')

# Definition:  p2.delete(self, index1, index2=None)
# Docstring:   Delete menu items between INDEX1 and INDEX2 (included).


#
# Remove 'Fork' item from menu p2
#
p2.delete(ixFork)

# Definition:  p2.delete(self, index1, index2=None)
# Docstring:   Delete menu items between INDEX1 and INDEX2 (included).

#
# Insert new items into p2 menu at and after ixFork
#
p2.insert_command(ixFork,   label='Plate')
p2.insert_command(ixFork+1, label='Cup')
    
# Definition:  p2.insert_command(self, index, cnf={}, **kw)
# Docstring:   Add command menu item at INDEX.
