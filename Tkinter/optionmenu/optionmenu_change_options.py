import Tkinter as tk

root = tk.Tk()
choices = ('network one', 'network two', 'network three')
var = tk.StringVar(root)
var.set('network one')

def refresh():
    # Reset var and delete all old options
    var.set('one')
    network_select['menu'].delete(0, 'end')

    # Insert list of new options (tk._setit hooks them up to var)
    new_choices = ('one', 'two', 'three')
    for choice in new_choices:
        network_select['menu'].add_command(label=choice, \
                                           command=tk._setit(var, choice))

network_select = tk.OptionMenu(root, var, *choices)
network_select.pack(side='left')

# I made this quick refresh button to demonstrate
refreshButton = tk.Button(root, text='Refresh', command=refresh)
 
refreshButton.pack(side='left')
