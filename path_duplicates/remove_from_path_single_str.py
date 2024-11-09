#
# remove_from_path.py
#
# Read a copy of the system PATH variable, remove from it all the paths 
# containing the string given as the first argument, and save the result
# in the file remove_from_path.txt .
#
# Example in ~/.bashrc : 
#
# 
#

import os, sys

if len(sys.argv) > 1:
    rmstr = sys.argv[1]
else:
    print 'No pattern to look for in PATH provided. Exiting.'
    sys.exit(1)


user = os.getenv('USER')    # A single string ':'-separated
path = os.getenv('PATH')    # A single string ':'-separated
path = path.split(':')      # List of all the paths

#
# Delete ALL paths to CASA
#
path = [p for p in path if not rmstr in p]

#
# Recreate the PATH with ':' delimiters
#
newpath = ':'.join(path)

print 'pwd: ', os.getcwd()

f = open('/home/' + user + '/remove_from_path.txt', 'w')
f.write(newpath)
f.close()


