#!/usr/bin/env python
#
# ls_path_unique.py [environment_variable]
#
# Print $PATH (or any environment variable with colon-separated content),
# one path on a separate line, leaving only unique paths.
#
# Reads a copy of the system PATH variable, removes from it all the duplicate
# paths leaving only the leftmost ones, and prints the result to stdout.
#
# Examples: 
#
# python ls_path_unique.py
#
# python ls_paths.py PYTHONPATH
#

import os, sys

if len(sys.argv) > 1:
	PATH = sys.argv[1]
else:
	PATH = 'PATH'          # Default

if os.getenv(PATH) is None:
    print(PATH, " is empy.")
    sys.exit(1)
    
path1 = os.getenv(PATH)    # A single string ':'-separated
path1 = path1.split(':')      # List of all the paths


#
# Find unique paths from path1 and move only them in path2
#
path2 = []
ndir = len(path1)

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
