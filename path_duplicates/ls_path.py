#!/usr/bin/env python
#
# ls_path.py
#
# Print $PATH in the screen,
# one path on a separate line.
#
# Example: 
#
# python ls_paths.py
#

import os

user = os.getenv('USER')
path1 = os.getenv('PATH')    # A single string ':'-separated
path = path1.split(':')      # List of all the paths


for p in path:
    print(p)
