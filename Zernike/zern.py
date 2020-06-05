import numpy as np
from numpy import pi, cos, arccos, sin, sqrt
from math import factorial as fac

def zernike_rad(n, m, rho):
	"""
	Make radial Zernike polynomial on coordinate grid **rho**.
	
	@param [in] n Radial order
	@param [in] m Angular frequency
	@param [in] rho Radial coordinate grid
	@return Radial polynomial with identical shape as **rho**
	"""

	rhoArr = isinstance(rho, (list, tuple, np.ndarray))

	if (n-m)%2 == 1:
		if rhoArr:
			return np.zeros_like(rho)
		else:
			return 0.

	wf = np.zeros_like(rho) if rhoArr else 0.
        
	for k in range((n-m)//2+1):
		wf += (-1 if k%2 else 1) * fac(n-k) //                    \
			  (fac(k) * fac((n+m)/2 - k) * fac((n-m)/2 - k)) *   \
			  rho**(n - 2.0*k)

	return wf



def zernike(n, m, rho, phi):
	"""
	Calculate Zernike mode (m,n) on grid **rho** and **phi**.
	**rho** and **phi** should be radial and azimuthal coordinate grids 
    of identical shape, respectively.

	@param [in] n Radial order
	@param [in] m Angular frequency
    @param [in] rho Radial coordinate grid
    @param [in] phi Azimuthal coordinate grid
    @return Zernike mode (m,n) with identical shape as rho, phi
	"""

	if m >= 0:
		zer = zernike_rad( m, n, rho) * np.cos(m*phi)
	else:
		zer = zernike_rad(-m, n, rho) * np.sin(m*phi)

	return zer
	


def noll_to_zern(j):
	"""
	Convert linear Noll index to tuple of Zernike indices.
	j is the linear Noll coordinate, n is the radial Zernike index 
        and m is the azimuthal Zernike index.

	@param [in] j Zernike mode Noll index
	@return (n, m) tuple of Zernike indices

	@see <https://oeis.org/A176988>.
	"""
	if (j == 0):
		raise ValueError("Noll indices start at 1, 0 is invalid.")

	n = 0
	j1 = j-1
	while (j1 > n):
		n += 1
		j1 -= n

	m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
	return (n, m)



def zernikel(j, rho, phi, norm=True):
	n, m = noll_to_zern(j)
	return zernike(m, n, rho, phi, norm)



def zern_normalisation(nmodes=30):
	"""
	Calculate normalisation vector.
	This function calculates a **nmodes** element vector with 
        normalisation constants for Zernike modes that have not 
        already been normalised.

	@param [in] nmodes Size of normalisation vector.
	"""

	nolls = (noll_to_zern(j+1) for j in xrange(nmodes))
	norms = [(2*(n+1)/(1+(m==0)))**0.5 for n, m  in nolls]
	return np.asanyarray(norms)



def zernike_rad_dct(m, n, rho):
	"""
	Make radial Zernike polynomial on coordinate grid **rho**
	using DCT (discrete cosine transform,
	Janssen, Dirksen, 2007,
	Computing Zernike polynomials of arbitrary degree using discrete Fourier
	transform,
	Journal of European Optical Society.

	
	@param [in] m Angular frequency
	@param [in] n Radial order
	@param [in] rho Radial coordinate grid
	@return Radial polynomial with identical shape as **rho**
	"""

	rhoArr = isinstance(rho, (list, tuple, np.ndarray))

	if (n-m)%2 == 1:
		if rhoArr:
			return np.zeros_like(rho)
		else:
			return 0.

	wf = np.zeros_like(rho) if rhoArr else 0.
        
	Npt = 20*n   # Number of points to evaluate the integral

	for kpt in range(Npt):
		x = rho*cos(2*pi*kpt/Npt)
		Un = sin((n+1)*arccos(x))/sqrt(1 - x**2)
		wf += Un*cos(2*pi*m*kpt/Npt)

	return wf/Npt
		




	







