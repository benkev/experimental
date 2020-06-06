import Tkinter as tk
from functools import partial

def pr(x):
    print 'Choice: ', x

root = tk.Tk()

mb = tk.Menubutton(root, text='Measurement Set', relief='raised')

picks = tk.Menu(mb, tearoff=False)               
mb.config(menu=picks)

picks.add_command(label='mayo',    command=partial(pr, 'mayo'))
picks.add_command(label='ketchup', command=partial(pr, 'ketchup'))

picks1 = tk.Menu(picks, tearoff=False)
picks.add_cascade(label="Recent MS", menu=picks1)

f = open('commands.txt')
lines = []
lnum = 0
while True:
    line = f.readline()
    line = line.strip()   # Remove '\n', too
    if line == '': break  # ======================================== >>
    lines.append(str(lnum+1)+' '+line)
    lnum = lnum + 1

f.close()

for i in xrange(lnum):
    line = lines[i]
    print 'line = ', line
    picks1.add_command(label=line, command=partial(pr, i+1))


mb.pack()
 
