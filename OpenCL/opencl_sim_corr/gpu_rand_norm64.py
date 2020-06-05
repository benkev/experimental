#
# gpu_rand_unif64.py
#
# Generate normally distributted pseudo-random numbers on GPU using pyopencl
#

import numpy as np
import pyopencl as cl
from pyrandl import PyRandl
import time

s_time = time.time()

Nwitem = 256
Nwgroup = 72
nrnd = np.uint32(10000)

Nproc = Nwitem*Nwgroup
nrnd_total = Nproc*nrnd

rndn = np.zeros(nrnd_total, dtype=np.float64)

seed = 90752

#
# Initialize the 4x64bit states for the Nproc generators using Python class
# PyRandl from module pyrandl and save them in rndst[Nproc,4], dtype=np.uint64
# array.
#
prng = PyRandl(seed, Nproc)
rndst = prng.get_rndst()

mf = cl.mem_flags

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

#
# Create input (generator states, rndst) and output (random numbers, rndn)
# buffersin the GPU memory. The mf.COPY_HOST_PTR flag forces copying from
# the host buffer, rndst, to the device buffer (referred as buf_rndst)
# in the GPU memory.
#
buf_rndst = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rndst)
buf_rndn =  cl.Buffer(ctx, mf.WRITE_ONLY, rndn.nbytes)

#
# Read the kernel code from file
#
with open ("ker_rand_norm64.cl") as fh: ker = fh.read()

prg = cl.Program(ctx, ker).build(options=['-I .'])

prg.genrand(queue, (Nproc,), (Nwitem,), buf_rndst, nrnd, buf_rndn)

cl.enqueue_copy(queue, rndn, buf_rndn)

e_time = int(time.time() - s_time)   # Elapsed time, seconds
tmin = e_time//60
tsec = e_time%60
print 'Elapsed time %d min %d sec.' % (tmin, tsec)


queue.flush()
queue.finish()
buf_rndst.release()
buf_rndn.release()
