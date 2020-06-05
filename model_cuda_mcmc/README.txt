This is an attepmt to use separately compiled model.cu instead of model.cuh
included into gpu_mcmc.cu.

So far unsuccessful.

2019 Feb 07.



============================================

To start using CUDA MCMC software, first enter this directory and issue the
command

$ make

It will create the directories ~/lib64/python, and ~/bin,  and setup the
environment for them by adding to the end of the ~/.bashrc file the commands

export PATH=$PATH:~/bin
export PYTHONPATH=$PYTHONPATH:~/lib64/python

After make finishes, please issue the command

$ source ~/.bashrc

Or simply open a new console window.


