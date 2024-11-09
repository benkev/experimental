#
# remove_duplicate_paths.py
#
# Read a copy of the system PATH variable, remove from it all the duplicate
# paths leaving only the leftmost ones, and save the result
# in the file remove_from_path.txt .
#
# Example in ~/.bashrc : 
#
# python remove_duplicate_paths.py
# export PATH="$(cat ~/remove_duplicate_paths.txt)"
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

#
# Recreate the PATH with ':' delimiters and save it 
# in ~/remove_duplicate_paths.txt
#
newpath = ':'.join(path2)

f = open('/home/' + user + '/remove_duplicate_paths.txt', 'w')
f.write(newpath)
f.close()



