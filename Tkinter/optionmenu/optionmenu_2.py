from Tkinter import *

master = Tk()

varMenuItem = StringVar(master)


def fun(val):
    print val
    
def fun1(a, b, c):
    print 'a=%s, b=%s, c=%s' % (a, b, c)
    print 'varMenuItem = ', varMenuItem.get()



def change_opts():
    m = w['menu']
    m.delete(0, 'end')

    mi = ['One', 'Two', 'Three', 'Monday', 'Tuesday', \
          'September', 'June', 'July']
    
    varMenuItem.set('One')
    
    for s in mi:
        m.add_command(label=s, command=lambda v=s: varMenuItem.set(v))
        
    varMenuItem.trace('w', fun1)
    


varMenuItem.set("Y Divided by None") # default value

mi = ["Y Divided by None", "Y Divided by Model", \
              "Y Divided by Corrected"]

w = OptionMenu(master, varMenuItem, *mi, command=fun)
w.pack(side='left')

b = Button(master, text='Change', command=change_opts)
b.pack(side='left')


