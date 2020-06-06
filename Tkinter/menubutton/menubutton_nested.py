from Tkinter import *

root    = Tk()

mbutton = Menubutton(root, text='Food')     # the pull-down stands alone
picks   = Menu(mbutton, tearoff=False)               
mbutton.config(menu=picks)           

picks.add_command(label='A')#,  command=root.quit)
picks.add_command(label='B')#,  command=root.quit)
picks.add_command(label='C')#, command=root.quit)
mbutton.pack()
mbutton.config(bg='tan', bd=1, relief=RAISED)

 
