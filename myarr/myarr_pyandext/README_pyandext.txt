python setup.py build
su
<passwd>
python setup.py install

will create and install two modules:

1. myarr_core module from a single c file myarrmodule.c.
2. myarr module from a python program myarr.py, which uses myarr_core.


