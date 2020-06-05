from pylab import *

x = np.arange(-10,10,0.001)
I = np.zeros(len(x))
I[np.where(np.abs(x)<1e-3)] = 1

sigma = 0.1
I = np.exp(-x*x/2/sigma**2)
I /= I.sum()

plt.figure()
plt.axhline(0, ls="-", color="gray")
plt.axvline(0, ls="-", color="gray")
plt.plot(x, I)
plt.xlim(-0.5,0.5)
plt.ylim(-0.0005,0.0045)
plt.xlabel("Spatial Extent")
plt.ylabel("Real Amplitude")
plt.grid(1)

Ishifted = np.fft.fftshift(I)
V = np.fft.ifftshift(np.fft.fft(Ishifted))

u = np.fft.fftshift(np.fft.fftfreq(len(x), d=0.001))

plt.figure()
plt.axhline(0, ls="-", color="gray")
plt.axvline(0, ls="-", color="gray")
plt.plot(u,np.abs(V), color="red", label="Absolute")
plt.plot(u,np.real(V), color="blue", ls="--", lw=2, label="Real Part")
plt.plot(u,np.imag(V), color="green", ls="--", lw=2, label="Imag Part")
plt.legend()
plt.xlim(-10,10)
plt.xlabel("Spatial Frequency")
plt.ylabel("Fourier Magunitude, Real part, Imag Part")
plt.grid(1)


plt.figure()
plt.axhline(0, ls="-", color="gray")
plt.axvline(0, ls="-", color="gray")
plt.plot(u,np.angle(V, deg=True), color="Orange", ls="--", lw=2, \
	label="Fourier Phase (deg)")
plt.yticks(-180+60*arange(7))
plt.legend()
plt.xlim(-10,10)
plt.ylim(-190,190)
plt.ylabel("Phase (deg)")
plt.grid(1)

plt.show()

