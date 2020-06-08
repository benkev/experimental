from pylab import *
from obsdatafit import *

p, XYext = read_quasikerr(
    '''/home/benkev/SgrA/Quasi_Kerr_Img/quasi-Kerr_images/'''
    '''ImageLibrary_Ref0/CoarsenedLibrary/pmap__000_000_013_2''')

N = 100
fN = float(N)
l = arange(N, dtype=float)
z1 = zeros((N,N), dtype=complex)
z = zeros((N,N), dtype=complex)

for i in xrange(N):
    for j in xrange(N):
        z1[i,j] = sum(p[i,:]*exp(-2.*pi*1j*l*j/fN))

for j in xrange(N):
    for i in xrange(N):
        z[i,j] = sum(z1[:,j]*exp(-2.*pi*1j*l*i/fN))

