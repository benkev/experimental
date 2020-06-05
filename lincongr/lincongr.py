#
# Linear Congruent Method for random number generation
#
# r_i = (a*r_i-1 + c) mod M
#
# The period of a general LCG is at most m, and for some choices of
# a much less than that. Provided that c is nonzero, the LCG will
# have a full period for all seed values if and only if:
#
#   1. c and M are relatively prime,
#   2. a - 1 is divisible by all prime factors of M,
#   3. a - 1 is a multiple of 4 if M is a multiple of 4.
#

from pylab import *

n = 10000
M = 1024
c = 1
a = 5

r = empty(n, dtype=int)
r[0] = 3
for i in xrange(1,n):
    r[i] = (a*r[i-1] + c) % M

print r
x = where(r == 1)[0]
print diff(x)
