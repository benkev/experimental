import distutils
from distutils.core import setup, Extension
import os


module_myarr = Extension('myarr_core', \
  sources = ['myarr_coremodule.c'])


setup(name='myarr', # Distribution name; will be like "myarr-0.11.tar.gz
      version='0.00',
      author='Leonid Benkevitch',
      author_email='benkev@haystack.mit.edu',
      description = 'This is a myarr package',
      url='haystack.mit.edu',
      
      py_modules = ['myarr'],
      ext_modules = [module_myarr]
)
