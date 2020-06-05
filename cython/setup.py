# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("xmpl", ["xmpl.pyx"])]

setup(
  name = 'Test app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
