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








## l = arange(8, dtype=float)
## x = array([1., 0., -1., 5., 7., 4., 0., -2.])

## zr = zeros(8)
## zi = zeros(8)

## for i in xrange(8):
##     zr[i] =  sum(x*cos(2*pi*l*i/8.))
##     zi[i] = -sum(x*sin(2*pi*l*i/8.))

## z1 = zr + 1j*zi


