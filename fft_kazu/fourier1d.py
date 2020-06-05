import numpy as np
import numpy.fft as fft
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

#
# This allows using LaTeX in the plot annotations
#
mpl.rcParams['mathtext.fontset'] = 'stix' # Allow math symbols b/w $ and $
mpl.rcParams['text.usetex'] = True # False

pi = np.pi
sqrt = np.sqrt


N = 4096
aN = float(N)
delx = 20./aN
#x = np.arange(-10, 10, delx)
x = np.linspace(-10, 10, N)
delx = x[1] - x[0]
#t = np.linspace(-pi, pi, N)
#win = np.cos(t)

#q = np.zeros((N,))   # Brightness
#b = np.zeros((N,))   # Brightness
#b[256] = 1.
#b[N/4] = 100.
#b[N/2] = 100.

#s = 2.0; mu = 0.
#s = 0.5; mu = 0.
#s = 1.; mu = 0.
#s = 0.1; mu = 2.
s = 0.1; mu = 0.
#s = 0.05; mu = 3.
#s = 0.001; mu = 0.
b = np.exp(-((x-mu)/s)**2)/(s*sqrt(2*pi)) # * win
#b = np.sin(10*x) + np.sin(2*x) + np.sin(100*x)

#
# Shift the origin from the center at N//2 (whether N is even or odd)
# to the beginning at 0
#
bsh = fft.ifftshift(b)  # Shift the origin from the c

# db = 1. 
# tri = 0.
# for ix in range(N):
# 	if -0.5 <= x[ix] and x[ix] <= 0:
# 		tri = tri + db
# 		b[ix] = tri
# 	if 0. < x[ix] and x[ix] <= 0.5:
# 		tri = tri - db
# 		b[ix] = tri

#
# Fourier transform
# The image vsh has 0 frequency at 0. Negative frequencies start an N//2
#
vsh = fft.fft(bsh)

#
# Shift the Fourier image from the beginning at 0 to the center at N//2
#
v = fft.fftshift(vsh)

v_orig = np.copy(v)

# v1 = fft.fft(b)   # Visibility
# v = fft.fftshift(v1)
absv = abs(v)
dx = x[1] - x[0]
freq = fft.fftfreq(N, d=dx) # Frequency ruler 
f = fft.fftshift(freq)      # Shift the beginning at 0 to the center at N//2
df = f[1] - f[0]


fig1 = plt.figure(figsize=(14,6))
ax1 = fig1.add_subplot(1, 2, 1)
ax1.plot(x, b); ax1.grid(1)
ax1.set_title('Brightness')  #: Gaussian $\mu=%g, \sigma=%g$' % (mu, s), \
#			  fontsize=16)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('B', fontsize=12)

#
# Equivalent to shift of the original to the right by delx/2
# so that its maximum be at N/2
#
v = v_orig*np.exp(-2*pi*1j*delx/2*f) 

#fig2 = plt.figure()
ax2 = fig1.add_subplot(1, 2, 2)
ax2.plot(f, v.real)
ax2.plot(f, v.imag)
ax2.plot(f, abs(v), ls='--', lw=2); ax2.grid(1)
ax2.set_title('Visibility', fontsize=16)
ax2.set_xlabel('f', fontsize=12)
ax2.set_ylabel('V', fontsize=12)

plt.figure()
plt.axhline(0, ls="-", color="gray")
plt.axvline(0, ls="-", color="gray")
plt.plot(f, np.angle(v, deg=1), color="Orange", ls="--", lw=2)
plt.grid(1)
plt.yticks(-180+60*np.arange(7))
plt.xlim(-10,10)
plt.ylim(-190,190)
plt.ylabel("Phase (deg)")


# b1 = fft.ifft(v1)
# plt.figure();
# plt.plot(x, b1.real, 'b', x, b1.imag, 'g')
# plt.plot(x[::2], abs(b1[::2]), 'r.'); plt.grid(1)
# plt.title('Inverse FFT of V')
# plt.xlabel('X')
# plt.ylabel('B')

plt.show()

print
print 'del_x = %g, x in [%g..%g]' % (dx, x[0], x[-1])
print 'del_f = %g, f in [%g..%g]' % (df, f[0], f[-1])
