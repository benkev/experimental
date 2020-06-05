import distutils
from distutils.core import setup, Extension
import os

import numpy as np
numpydir = os.path.dirname(np.__file__) + '/core/include/numpy/'



module_myarr = Extension('myarr', \
  sources = ['myarrmodule.c'],    \
  include_dirs=[numpydir])


setup(name='MyArr Module', # Distribution name; will be like "myarr-0.11.tar.gz
      version='0.00',
      author='Leonid Benkevitch',
      author_email='benkev@haystack.mit.edu',
      description = 'This is a myarr package',
      url='haystack.mit.edu',
      #py_modules = ['myarr'],
      ext_modules = [module_myarr]
)
