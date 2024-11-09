#
# remove_from_path.py
#
# Read a copy of the system PATH variable, remove from it all the paths 
# containing the strings given as the arguments, and save the result
# in the file remove_from_path.txt .
#
# Example in ~/.bashrc : 
#
# python remove_from_path.py cuda casa-release
# export PATH="$(cat ~/remove_from_path.txt)"
# export PATH=$PATH:/usr/local/cuda/bin
# export PATH=$PATH:/home/benkev/bin/casa-release-4.6.0-el6/bin
#


import os, sys

if len(sys.argv) > 1:
    rmstr = sys.argv[1:]
else:
    #print 'No pattern to look for in PATH provided. Exiting.'
    print('No pattern to look for in PATH provided. Exiting.')
    sys.exit(1)


user = os.getenv('USER')    # A single string ':'-separated
path = os.getenv('PATH')    # A single string ':'-separated
path = path.split(':')      # List of all the paths

#
# Delete ALL paths containing strings from rmstr
#
for rs in rmstr:
    path = [dr for dr in path if not rs in dr]

#
# Recreate the PATH with ':' delimiters
#
newpath = ':'.join(path)

f = open('/home/' + user + '/remove_from_path.txt', 'w')
f.write(newpath)
f.close()


