#
# Print integer coefficients (those before r^(n-2k) )
# of the radial part of Zernike polynomials
#

import numpy as np
from math import factorial as fac

r = 0.5

for n in range(8):            # Radial order
    for m in range(1):        # Angular frequency
        if (n + m)%2 == 0:
            wf = 0.0
            for k in range((n-m)//2+1):
                sg = 
                numer = fac(n-k)
                denom = (fac(k) * fac((n+m)/2.0 - k) * fac( (n-m)/2.0 - k))
                coef =  * numer // denom *      \
                      
                print(n, m, 'nume = ', 'coef = ', coef)
                wf = wf + coef*r**(n-2.0*k)
        #print(n, m, 'wf = ', wf)
    print()        
