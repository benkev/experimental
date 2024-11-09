#!/usr/bin/env python
#
# ls_unique_paths.py
#
# Read a copy of the system PATH variable, remove from it all the duplicate
# paths leaving only the leftmost ones, and print it in the screen,
# one path on a separate line.
#
# Example: 
#
# python ls_unique_paths.py
#

import os

user = os.getenv('USER')
path1 = os.getenv('PATH')    # A single string ':'-separated
path1 = path1.split(':')      # List of all the paths


#
# Find unique paths from path1 and move only them in the path2 list
#
path2 = []
ndir = len(path1)

idir = 0
while (True):
    try:
        d1 = path1.pop(0)
    except IndexError:
        break              # Finish if we ran out of directories in path1

    notFound = True
    for d2 in path2:
        if d1 == d2:
            notFound = False
            break           # Duplicate dir found
    if notFound:
        path2.append(d1)


for p in path2:
    print(p)
