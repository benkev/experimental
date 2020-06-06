from Tkinter import *
import Tkinter as tk


def on_select(event):
    wd = event.widget
    index = int(wd.curselection()[0])
    value = wd.get(index)

    #print 'on_select: index = ', index, ', value = ', value

    txt.delete("1.0", tk.END)
    txt.insert("1.0", value)
    


master = Tk()

listbox = Listbox(master)
listbox.pack()

listbox.config(borderwidth=2, selectmode=tk.SINGLE)

listbox.bind('<<ListboxSelect>>', on_select)

listbox.insert(END, "a list entry")

for item in ["one", "two", "three", "four"]:
    listbox.insert(END, item)

txt = tk.Text(master, height=1, width=23, borderwidth=2)
txt.pack()

listbox.selection_set(0)
print 'listbox.selection_includes(0) = ', \
      listbox.selection_includes(0)
listbox.event_generate('<<ListboxSelect>>')






