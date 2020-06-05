#
# 1 pole digital low pass filter on short integer (int16) variables
#
from pylab import *
import sys

N = 100
u = 256
m = 5   #number of digits

x = u*int16(rand(N) > 0.5)
x[20:] = u 
t = arange(N)
y = zeros_like(x)

M = 2**m
anum = int16(M-1)
aden = int16(M)
bnum = int16(1)
aden = int16(M-1)

shif = int16(m)


print t[0], x[0], y[0]
for i in xrange(N-1):
    i1 = i + 1
    y[i1] = (anum*y[i] + bnum*x[i])>>shif
    print t[i1], x[i1], y[i1]

sm = str(m)
figure()
plot(t, x, 'b'); grid(1)
plot(t, y, 'r')
plot(t, y, 'r.')
legend(('x', 'y'), loc='best')
title(r'Int16-based Digital LP Filter Response $y_t$ to Input $x_t$', fontsize=16)
tx = r'$a = (2^'+sm+r'-1)/2^'+sm+r'$', (m, m)
figtext(0.4, 0.4, r'$y_{t+1}=a y_t + b y_t, \,\, a+b=1$', fontsize=20)
figtext(0.4, 0.3, r'$a = (2^'+sm+r'-1)/2^'+sm+r', \,\,b = 1/2^'+sm+r'$', fontsize=20)
xlabel('time')
ylabel('x, y')
show()











