import distutils
from   distutils.core import setup, Extension
import os
import numpy as np

numpydir = os.path.dirname(np.__file__) + '/core/include/numpy/'



module_ctest = Extension('_C_arraytest', \
  sources = ['C_arraytest.c'],    \
  include_dirs=[numpydir])

# Distribution name; will be like "mwa-0.11.tar.gz
setup(name='_C_arraytest Module', 
      version='0.00',
      author='Leonid Benkevitch',
      author_email='benkev@haystack.mit.edu',
      description = 'This is a mwa package',
      url='http://www.haystack.mit.edu/',
      ext_modules = [module_ctest]
)
