#!/usr/bin/env python
#
# ls_path.py [environment_variable]
#
# Print $PATH (or any environment variable with colon-separated content),
# one path on a separate line to stdout.
# 
#
# Example: 
#
# python ls_paths.py
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
path = path1.split(':')      # List of all the paths


for p in path:
    print(p)
