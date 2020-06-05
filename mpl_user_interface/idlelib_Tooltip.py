from idlelib.ToolTip import *

def main():
    root = Tk()
    b = Button(root, text="Hello", command=root.destroy)
    b.pack()
    root.update()
    tip = ListboxToolTip(b, ["Hello", "world"])
    root.mainloop()

if __name__ == '__main__':
    main()

