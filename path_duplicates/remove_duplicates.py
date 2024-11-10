#
# remove_duplicates.py
#
# Read copy of a system environment variable containing colon-separated
# paths, remove from it all the duplicate paths leaving only the leftmost ones,
# and place the result on the standard output. 
# A suggested name for the file to save it:
#    no_duplicates_<env variable name>.txt .
#
# Examples in ~/.bashrc : 
#
# export PATH="$(python ~/bin/remove_duplicates.py)"
#
# python ~/bin/remove_duplicates.py PATH > no_duplicates_PATH.txt
# export PATH="$(cat ~/no_duplicates_PATH.txt)"
#
# python ~/bin/remove_duplicates.py PYTHONPATH > no_duplicates_PYTHONPATH.txt
# export PYTHONPATH="$(cat ~/no_duplicates_PYTHONPATH.txt)"
#

import os, sys

if len(sys.argv) > 1:
	PATH = sys.argv[1]
else:
	PATH = 'PATH'          # Default

user = os.getenv('USER')
path1 = os.getenv(PATH)    # A single string ':'-separated
path1 = path1.split(':')      # List of all the paths
path1 = [p for p in path1 if p != '']    # Remove empty strings, if any 


#
# Find unique paths from path1 and move only them to path2
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
# Recreate the PATH with ':' delimiters
#
newpath = ':'.join(path2)

# f = open('/home/' + user + '/no_duplicates_' + PATH + '.txt', 'w')
# f.write(newpath)
# f.close()

print(newpath)




