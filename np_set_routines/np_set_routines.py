import numpy as np

a = np.array((1,5,8,-1,0,4,1,-1,-2,7,8,9,8,7,9,2,-6))
b = np.array((4,3,8,1,9,7,1,-1,-3,7,6,9,1,8,3,2,9))

print 'a = ', a
print 'b = ', b
print

ua, ia, iai, iac = np.unique(a, return_index=True, return_inverse=True, \
                             return_counts=True)

ub, ib, ibi, ibc = np.unique(b, return_index=True, return_inverse=True, \
                             return_counts=True)

print 'Unique a = ', ua
print 'Unique indices of a = ', ia
print 'Indices to reconstruct a from ua = ', iai
print 'Numbers of occurrences of ua in a = ', iac
print

print 'Unique b = ', ub
print 'Unique indices of b = ', ib
print 'Indices to reconstruct b from ub = ', ibi
print 'Numbers of occurrences of ub in b = ', ibc
print

ab = np.intersect1d(a, b)

print 'Intersection: Values that are in both a and b:', ab
print

aub = np.union1d(a, b)

print 'Union:   Values that are in either of a and b:', aub
print

amb = np.setdiff1d(a, b)
bma = np.setdiff1d(b, a)

print 'Difference a-b:   Values in a that are not in b:', amb
print
print 'Difference b-a:   Values in b that are not in a:', bma
print




