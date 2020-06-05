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



N = 256

sx = .5  # Gaussian STD_x
sy = .5  # Gaussian STD_y
mux = 2. # -0.2 #-0.02 #01
muy = 1. # 0.3  #05

stri = N//32

# x = np.linspace(-5, 5, N)
# step = x[1] - x[0]
step = 20./N
x = np.arange(-10, 10, step)
freq = fft.fftfreq(N, d=step)
f = fft.ifftshift(freq)
U = np.tile(f, (N, 1)) # Repeat f[N] in N lines
V = U.transpose()

X, Y = np.meshgrid(x, x)

pi = np.pi
sqrt = np.sqrt

b = (1./(2.*pi*sx*sy))*np.exp(-0.5*(((X-mux)/sx)**2 + ((Y-muy)/sy)**2))

bsh = fft.ifftshift(b)
vsh = fft.fft2(bsh)   # Visibility
v = fft.fftshift(vsh)
absv = abs(v)

#
# To denoise the margins, set too small
# visibility Re and Im components to exact 0.
#
v[abs(v) < 1e-10] = 0
# v.real[abs(v.real) < 1e-10] = 0.
# v.imag[abs(v.imag) < 1e-10] = 0.

fig1 = plt.figure(figsize=(21,6))
ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
ax1.plot_wireframe(U, V, v.real, rstride=stri, cstride=stri)
ax1.set_title('Visibility.Real', fontsize=16)
ax1.set_ylabel('V', fontsize=12)
ax1.set_xlabel('U', fontsize=12)

ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
ax2.plot_wireframe(U, V, v.imag, rstride=stri, cstride=stri)
ax2.set_title('Visibility.Imag', fontsize=16)
ax2.set_ylabel('V', fontsize=12)
ax2.set_xlabel('U', fontsize=12)

ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
ax3.plot_wireframe(U, V, np.angle(v, deg=1), rstride=stri, cstride=stri)
ax3.set_title('Visibility Phase (deg)', fontsize=16)
#ax3.set_title('Visibility.Abs', fontsize=16)
ax3.set_ylabel('V', fontsize=12)
ax3.set_xlabel('U', fontsize=12)

# fig2 = plt.figure()
# ax4 = fig2.add_subplot(1, 1, 1, projection='3d')
# ax4.plot_wireframe(X, Y, b, rstride=stri, cstride=stri)
# #ax4.plot_wireframe(X, Y, absv)
# #ax4.contour(X, Y, absv, 20); ax4.grid(1)
# ax4.set_title('Brightness', fontsize=16)
# ax4.set_ylabel('X', fontsize=12)
# ax4.set_xlabel('Y', fontsize=12)

plt.figure(figsize=(21,6))

plt.subplot(131)
plt.imshow(v.real, interpolation='none', origin='lower', \
		   extent=[f[0], f[-1], f[0], f[-1]])
plt.title('Visibility Real'); plt.xlabel('U');  plt.ylabel('V')
plt.colorbar()

plt.subplot(132)
plt.imshow(v.imag, interpolation='none', origin='lower', \
		   extent=[f[0], f[-1], f[0], f[-1]])
plt.title('Visibility Imag'); plt.xlabel('U');  plt.ylabel('V')
plt.colorbar()

plt.subplot(133)
plt.imshow(np.angle(v,1), interpolation='none', origin='lower', \
		   extent=[f[0], f[-1], f[0], f[-1]])
plt.title('Phase (deg)'); plt.xlabel('U');  plt.ylabel('V')
plt.colorbar()

#
# Analytical visibility va(U,V)
#
va = np.exp(-2.*pi*((sx*U)**2 + (sy*V)**2) - 2j*pi*(mux*U + muy*V))


plt.figure(figsize=(21,6))

plt.subplot(131)
plt.imshow(va.real, interpolation='none', origin='lower', \
		   extent=[f[0], f[-1], f[0], f[-1]])
plt.title('Analytical Visibility Real'); plt.xlabel('U');  plt.ylabel('V')
plt.colorbar()

plt.subplot(132)
plt.imshow(va.imag, interpolation='none', origin='lower', \
		   extent=[f[0], f[-1], f[0], f[-1]])
plt.title('Analytical Visibility Imag'); plt.xlabel('U');  plt.ylabel('V')
plt.colorbar()

plt.subplot(133)
plt.imshow(np.angle(va,1), interpolation='none', origin='lower', \
		   extent=[f[0], f[-1], f[0], f[-1]])
plt.title('Analytical Phase (deg)'); plt.xlabel('U');  plt.ylabel('V')
plt.colorbar()



plt.show()








