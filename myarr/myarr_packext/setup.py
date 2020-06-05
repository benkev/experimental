import distutils
from distutils.core import setup, Extension
import os


#module_myarr = Extension('myarr.core', \
#  sources = ['core/myarrmodule.c'])


setup(name='myarr',
      version='0.00',
      author='Leonid Benkevitch',
      author_email='benkev@haystack.mit.edu',
      description = 'This is a myarr package',
      url='haystack.mit.edu',
      
      #package_dir = {'myarr.core': 'core'},
      packages = ['myarr']  #, 'myarr.core'],
#      packages = ['myarr']
      #py_modules = ['myarr'],
      #ext_modules = [module_myarr]
)
