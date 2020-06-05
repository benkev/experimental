python setup.py build_ext --inplace

To install locally in ~/lib64/python:

cp myarr.so ~/lib64/python/


Otherwise, if you need to install the module in the system, use the following:
su
<passwd>
python setup.py install

It will create and install a myarr module from a single c file myarrmodule.c.

