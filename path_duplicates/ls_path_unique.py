#!/usr/bin/env python
#
# ls_path_unique.py
#
# Read a copy of the system PATH variable, remove from it all the duplicate
# paths leaving only the leftmost ones, and print the result
#
# Example: 
#
# python ls_path_unique.py
#

import os
#import numpy as np

user = os.getenv('USER')
path1 = os.getenv('PATH')    # A single string ':'-separated
path1 = path1.split(':')      # List of all the paths


#
# Find unique paths from path1 and move only them in path2
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
