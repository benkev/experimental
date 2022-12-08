import sys, os
import numpy as np
from math import perm, comb, pow
import matplotlib.pyplot as pl
#from scipy.special import erf

# for i in range(1,13):
#     for j in range(i+1):
#         print("%4d " % comb(i,j), end="")
#     print()

if len(sys.argv) < 3:
    # print("Must be 2 parameters: number n and probability p.")
    print("Must be a parameter: number n.")
    raise SystemExit

n = int(sys.argv[1])
# p = float(sys.argv[2])

# if (n <= 0) or (p < 0.0) or (p > 1.0):
#     print("Wrong input: n = %d, p = %f" % (n, p))
#     raise SystemExit

# q = 1 - p

pmf = np.zeros(n+1, dtype=float)

# for k in range(n+1):
#     pmf[k] = comb(n,k)*pow(p,k)*pow(q,n-k)
#     print("%4d " % comb(n,k), end="")
# print()

# for k in range(n+1):
#     print("%6f " % pmf[k], end="")
# print()

# pl.figure()
# pl.plot(pmf, "o")
# pl.grid(1)

# for k in range(n+1):
#     pl.plot(pmf[k], "o")

#prob = np.linspace(0.1, 0.9, 9)
prob = np.array([1, 20, 40, 60, 80], dtype=float)/float(n)

#
# Binomial distribution
#
pl.figure()
pl.grid(1)
pl.title("Binomial distributions")
for p in prob:
    q = 1 - p
    for k in range(n+1):
        pmf[k] = comb(n,k)*pow(p,k)*pow(q,n-k)
    print("p = %g, mu = %g, sig = %g" % (p, n*p, n*p*q))
    #pl.plot(pmf, "o")
    pl.plot(pmf)


#
# Poisson distribution
#
#mu = np.linspace(0.1, 0.9, 9)
mus = np.array([1, 20, 40, 60, 80, 100], dtype=float)

pl.figure()
pl.grid(1)
pl.title("Poisson distributions")
for mu in mus:
    for k in range(n+1):
        pmf[k] = pow(mu,k)*np.exp(-mu)/perm(k)
    print("mu = sig = %g" % mu)
    pl.plot(pmf)


pl.show()

    




