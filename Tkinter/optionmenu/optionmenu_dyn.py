import Tkinter as tk

class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.om_variable = tk.StringVar(self)

        b1 = tk.Button(self, text="Colors", width=8, command=self.use_colors)
        b2 = tk.Button(self, text="Sizes", width=8, command=self.use_sizes)

        self.om = tk.OptionMenu(self, self.om_variable, ())
        self.om.configure(width=20)
        self.use_colors()

        b1.pack(side="left")
        b2.pack(side="left")
        self.om.pack(side="left", fill="x", expand=True)

        self.om_variable.trace('w', self.om_callback)


    def reset_option_menu(self, options, index=None):
        '''reset the values in the option menu

        if index is given, set the value of the menu to
        the option at the given index
        '''
        menu = self.om["menu"]
        menu.delete(0, "end")
        for string in options:
            menu.add_command(label=string, 
                             command=lambda value=string:
                                  self.om_variable.set(value))
        if index is not None:
            self.om_variable.set(options[index])

    def use_colors(self):
        '''Switch the option menu to display colors'''
        self.reset_option_menu(["red","orange","green","blue"], 0)

    def use_sizes(self):
        '''Switch the option menu to display sizes'''
        self.reset_option_menu(["x-small", "small", "medium", "large"], 0)

    def om_callback(self, *args):
        v = self.om_variable.get()
        print v

if __name__ == "__main__":
    app = SampleApp() 
