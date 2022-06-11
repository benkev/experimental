#
# ocl_calc_vvcorr.py
#
# Calculate correlation of very long random sequences and make the
# van Vleck correction of the correlation. The computation of ~10^10
# sequences take ~3 minutes due to the GPU compurations using pyopencl.
#
# Example:
#
# $ ./ssc  1e6	160	 64	 4	2.0 1.75
#
# -- use 160 work groups times 64 work items per block times 1^6 samples in each
#	 thread, which makes 10,240,000,000 samples on total. Number of bits for
#	 quantization is 4, standard deviations of the signals are 2.0 and 1.75.
#
# Or, in CUDA terminology:
# -- use 160 blocks times 64 threads per block times 1^6 samples in each
#	 thread, which makes 10,240,000,000 samples on total.The rest is as above.


import numpy as np
import pyopencl as cl
import pyopencl.tools
import pyopencl.array
from pyrandl import PyRandl
import time, sys

s_time = time.time()

# Nwitem = 256
# Nwgroup = 72
# nrand = np.uint32(10000)

#
# Input from the command line
#
if len(sys.argv) != 7:
	print()
	print('Usage:\n');		   
	print('ocl_calc_vvcorr nrand nwgroup nwitem nbit sig1 sig2\n')
	print('	  nrand: number of random samples in one sequence (thread)')
	print('	  nwgroup: number of work groups (blocks of threads)')
	print('	  nwitem: number of work items (threads) per one work group ' \
		  '(block)')
	print('	  nbit: ADC precision in number of bits')
	print('	  sig1, sig2: standard deviations of the correlated analog ' \
		  'signals\n')
	raise SystemExit

nrand = int(float(sys.argv[1]))
Nwgroup = int(sys.argv[2])
Nwitem = int(sys.argv[3])
nbit = int(sys.argv[4])
sx = float(sys.argv[5])
sy = float(sys.argv[6])

nseq = Nwitem*Nwgroup
ntotal = nseq*nrand

print('nrand=%d, Nwgroup=%d, Nwitem=%d, nbit=%d, nseq=%d, ntotal=%d' % \
		(nrand, Nwgroup, Nwitem, nbit, nseq, ntotal))
print('sx=%f, sy=%f' % (sx, sy))

#
# Define structures
#
calc_corr_input_ = \
	np.dtype([('seed', np.uint64), ('nrand', np.int64), \
			  ('nseq', np.int32), ('nbit', np.int32), \
			  ('sx', np.float64), ('sy', np.float64), \
			  ('a', np.float64), ('b', np.float64), ], align=True)

calc_corr_result_ = \
	np.dtype([('r', np.float64), ('qr', np.float64), \
			  ('acx', np.float64), ('acy', np.float64)], align=True)



npoint = 101  # Points in interpolated correlation table
npos = npoint/2
nsnr = 8


# Theoretical correlations thrho[8] 
thrho = np.array((0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999), \
				 dtype=np.float64)
# Square root of the signal-to-noise ratio
snr = np.zeros(nsnr, dtype=np.float64)
for i in range(nsnr):
	snr[i] = np.sqrt(thrho[i]/(1. - thrho[i]))

# rndn = np.zeros(nrnd_total, dtype=np.float64)

kap = np.zeros(npoint, dtype=np.float64)
rho = np.zeros(npoint, dtype=np.float64)
vcor = np.zeros(nsnr, dtype=np.float64)
cor = np.zeros(nsnr, dtype=np.float64)
qcov = np.zeros(nsnr, dtype=np.float64)
sqcor = np.zeros(nsnr, dtype=np.float64)
dq = np.zeros(nsnr, dtype=np.float64)
dv = np.zeros(nsnr, dtype=np.float64)


seed = 90752


#
# Initialize the 4x64bit states for the nseq generators using Python class
# PyRandl from module pyrandl and save them in rndst[nseq,4], dtype=np.uint64
# array.
#
prng = PyRandl(seed, nseq)
rndst = prng.get_rndst()   # rndst is array rndst[nseq,4]

#
# Initialize OpenCL and prepare data arrays before kernel execution
#
mf = cl.mem_flags

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

#
# Create input (generator states buf_rndst, parameters buf_ci) and output
# (correlations and autocorrelations bus_cr) buffers in the GPU memory.
# The mf.COPY_HOST_PTR flag forces copying from the host buffer, rndst, to the
# device buffer (referred as buf_rndst) in the GPU memory.
#
buf_rndst = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rndst)

#
# As a remark on implementation, the match_dtype_to_c_struct() routine runs a
# small kernel on the given device to ensure that numpy and C offsets and sizes
# match. 
#
calc_corr_input, ci_cdecl = pyopencl.tools.match_dtype_to_c_struct( \
		ctx.devices[0], 'calc_corr_input', calc_corr_input_)
calc_corr_input = \
		cl.tools.get_or_register_dtype("calc_corr_input", calc_corr_input)
ci_h = np.empty(1, dtype=calc_corr_input)

calc_corr_result, cr_cdecl = pyopencl.tools.match_dtype_to_c_struct( \
		ctx.devices[0], 'calc_corr_result', calc_corr_result_)
calc_corr_result = \
		cl.tools.get_or_register_dtype("calc_corr_result", calc_corr_result)
cr_h = np.empty(nseq, dtype=calc_corr_result)

cr = cl.array.empty(queue, nseq, dtype=calc_corr_result)


ci_h[0]['seed'] = np.uint64(seed)
ci_h[0]['nrand'] = np.int64(nrand)
ci_h[0]['nseq'] = np.int32(nseq)
ci_h[0]['nbit'] = np.int32(nbit)
ci_h[0]['sx'] = np.float64(sx)
ci_h[0]['sy'] = np.float64(sy)
ci_h[0]['a'] = np.float64(0)
ci_h[0]['b'] = np.float64(0)

# raise SystemExit

#
# Read the kernel code from file and concatenate (prepend) the kernel code with
# the structure declarations 
#
with open ("ker_calc_vvcorr_xoshiro256starstar.cl") as fh: ker = fh.read()
ker = ci_cdecl + cr_cdecl + ker

prg = cl.Program(ctx, ker).build()   #(options=['-I .'])

for isn in range(nsnr):

	ci_h[0]['a'] = snr[isn] / np.sqrt(1. + pow(snr[isn],2))
	ci_h[0]['b'] = np.sqrt(1. - pow(ci_h[0]['a'],2))

	print('Py: a=%g, b=%g' % (ci_h[0]['a'], ci_h[0]['b']))
	
	ci = cl.array.to_device(queue, ci_h)

	prg.calc_corr(queue, (nseq,), (Nwitem,), buf_rndst, ci.data, cr.data)

	#cl.enqueue_copy(queue, cr, cr.data)

	cr.get(ary=cr_h)              # Download from GPU to cr_h

	r = qr = acx = acy = 0;
	for iseq in range(nseq):
		r	+= cr_h[iseq]['r'];
		qr	+= cr_h[iseq]['qr'];
		acx += cr_h[iseq]['acx'];
		acy += cr_h[iseq]['acy'];

	cor[isn] = r/np.sqrt(acx * acy)
	# sqcor[isn] = slp * qr / ntotal
	qcov[isn] = qr/ntotal

	#
	# Linear interpolation of rho = f(kappa)
	#
	#vcor[isn] = lininterp_vv(npoint, kap, rho, qcov[isn]);


# print()
# print('rndst:')
# for i in range(nseq): print("%20d, %20d, %20d, %20d" % tuple(rndst[i,:]))
# print()

print()
print()
print("thrho = ")
for i in range(nsnr): print("%.8f " % thrho[i])
print()
print("snr =   ")
for i in range(nsnr): print("%.8f " % snr[i])
print()
print("qcov =  ")
for i in range(nsnr): print("%.8f " % qcov[i])
print()
print("cor =   ")
for i in range(nsnr): print("%.8f " % cor[i])
print()
# print("vcor =  " )
# for i in range(nsnr): print("%.8f " % vcor[i])
# print()
# print("sqcor = " )
# for i in range(nsnr): print("%.8f " % sqcor[i])
# print()
# print("dq =    " )
# for i in range(nsnr): print("%10.5f " % dq[i])
# print()
# print("dv =    ") 
# for i in range(nsnr): print("%10.5f " % dv[i])
# print()

	

e_time = int(time.time() - s_time)	 # Elapsed time, seconds
tmin = e_time//60
tsec = e_time%60
print()
print('Elapsed time %d min %d sec.' % (tmin, tsec))


queue.flush()
queue.finish()
buf_rndst.release()
