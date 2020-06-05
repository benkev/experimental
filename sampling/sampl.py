#
# Experimenting with 8-bit (1+cosine)/2 downsampled to 5 bits, and then down to
# variable number of bits, 'downsample' 
#

from pylab import *
import os, sys

N = 1024

th = linspace(0., pi, N)
cs8 = zeros(N, dtype=uint8)   # "UNSIGNED" 1-byte integers 
cs32 = zeros(N, dtype=uint32) # "UNSIGNED" 4-byte integers 
cs5 =  zeros(N, dtype=uint8)
cs5r = zeros(N, dtype=uint8)
cs = 0.5*(1. + cos(th))
cs32[:] = (2**32-1)*cs         
#cs8 = (~(2**24-1)) & cs32        # ~(2**24-1) = ~'#00FFFFFF' = '#FF000000'

## figure();
## plot(th, cs32, label='32bit cos'); grid(1)
## legend()

cs8[:] = cs32 >> 24

#
# Cutting 8bit downto 5bit
#
cs5[:] = (cs8 >> 3) << 3

#
# Rounding 8bit downto 5bit
#
idxge = where(cs8 >= (2**8-4))   # cos8 >= 256-4, ie > 252
cs5r[idxge] = 2**8-1
idxlt = where(cs8 < (2**8-4))   # cos8 < 256-4, ie <= 252
cs5r[idxlt] = ((cs8[idxlt] + 4) >> 3) << 3

figure();
plot(th, cs8, label='8-bit cos'); grid(1)
plot(th, cs5, label='5-bit cos trunk')
plot(th, cs5r, label='5-bit cos round')
legend()



#
# Cutting and rounding 8bit downto 'downsample' bit
#
downsample = 2  # number of significant bits left after downsampling

nb = 8 - downsample  # bits to cut/round

addn = 2**(nb-1)    # number to add for rounding: smthg like 0000100 or so
csnb =  zeros(N, dtype=uint8)
csnbr = zeros(N, dtype=uint8)

csnb[:] = (cs8 >> nb) << nb    # Cutting 8bit downto nb bit

# Cutting and rounding 8bit downto nb bit
idxge = where(cs8 >= (2**8-addn))   # cos8 >= 256-4, ie > 252
csnbr[idxge] = 2**8-1
idxlt = where(cs8 < (2**8-addn))   # cos8 < 256-4, ie <= 252
csnbr[idxlt] = ((cs8[idxlt] + addn) >> nb) << nb

figure();
plot(th, cs8, 'b', label='8-bit cos'); grid(1)
plot(th, csnb, 'g', label='%d-bit cos trunk' % (8-nb))
plot(th, csnbr, 'r', label='%d-bit cos round' % (8-nb))
legend()



show()

