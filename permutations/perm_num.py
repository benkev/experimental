import numpy as np
import sys, copy
import factoradic as fr





def to_perm(perm_number):
	"""
	Convert a permutation number into the corresponding permutation.
	The argument, an integer number perm_number, is first represented in 
	the factorial number system, also called factoradic, or as 
	a sequence of "factorial digits" in fdigits. In turn, fdigits is
	transformed into an array of the permutation indices, which is returned.
	"""
	
	fdigits = fr.to_factoradic(perm_number)
	fdigits.reverse()	   # Make it start from most significant digit
	perm_len = len(fdigits) # Perm length equals that of factorial number
	perm0 = list(np.arange(perm_len, dtype=int)) # Trivial perm 0,1,2...

	perm = []
	for ix in xrange(perm_len):
		# print 'ix = %d, fdigits[ix] = %d' % (ix, fdigits[ix]) 
		digit = fdigits[ix]	  # Get the multiplier at (perm_len-1-ix)!
		perm_elem = perm0.pop(digit)  # Extract element from digith pos.
		perm.append(perm_elem)		  # And put it to the perm

	aperm = np.array(perm)

	return aperm



def to_perm_number(permutation):
	"""
	For a given permutation find its ordinal number.
	The argument can be any Python sequence incliding strings.
	It is first represented as a list of the permutation ondices where the
	smallest element os zero. Then the factorial number of perm is found.
	as a sequence of "factorial digits" in fdigits. The factorial number
	is converted into the permutation ordinal number, which is returned.
	"""

	# If the argument is string, explode into chars list 
	perm = list(permutation)  
	aperm = np.array(perm)
	if len(aperm) <> len(np.unique(aperm)):
		print 'ERROR: All elements in the permutation must be unique.'
		return None
		#raise SystemExit

	if isinstance(permutation, str):
		min_aperm = ord(min(aperm))
		aperm = np.array(map(ord, perm))
		perm = list(aperm - min_aperm) # Render chars as indices starting at 0
	else:
		min_aperm = min(aperm)
		perm = list(aperm - min_aperm) # Render numbers as indices starting at 0


	perm0 = copy.copy(perm)
	perm0.sort()			# Trivial perm is just perm in ascending order
	perm_len = len(perm0)
	fdigits = np.zeros(perm_len, dtype=int)	 # Factorial number digits

	# print 'perm =  ', perm
	# print 'perm0 = ', perm0

	for ix in xrange(perm_len):
		perm_elem = perm.pop(0)	 # Extract perm element from its beginning
		elem_idx0 = perm0.index(perm_elem) # Where is the element in perm0?
		perm0.pop(elem_idx0)	   # Exclude the element dtom trivial perm
		fdigits[ix] = elem_idx0
		# print 'perm_elem = ', perm_elem, ', fdigits = ', fdigits,
		# print ', perm = ', perm,
		# print ', perm0 = ', perm0

	fdigits = fdigits[::-1]	 # Reverse: make it start from least signif. digit
	perm_number = fr.from_factoradic(fdigits)

	return perm_number 


def perm_to_str(perm, str0):
	"""
	Represent a permutation perm as permutation of charcters. The trivial
	character string permutation is passed in str0.
	Returns permutation in the string form.
	"""
	
	alp0 = np.array(list(str0)) # Explode string into array of chars
	
	if len(alp0) <> len(perm):
		print 'ERROR: Number count in array perm must equal char count in ' \
		    'string str0.'
		return None
		
	perm_s = ''.join(alp0[perm])

	return perm_s

	


if __name__ == "__main__":

	pnum = 511
	perm = to_perm(pnum)
	print 'perm#: ', pnum, ', perm = ', perm
	
	pnum = to_perm_number(perm)
	print 'perm = ', perm, ', to_perm_number(perm) = ', pnum

	print
	
	perm = (5, 7, 1, 2, 6, 4, 0, 3)
	pnum = to_perm_number(perm)
	print 'perm = ', perm, ', to_perm_number(perm) = ', pnum
	
	perm = to_perm(pnum)
	print 'perm#: ', pnum, ', perm = ', perm

	print

	perm = (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
	pnum = to_perm_number(perm)
	print 'perm = ', perm, ', to_perm_number(perm) = ', pnum

	perm = to_perm(pnum)
	print 'perm#: ', pnum, ', perm = ', perm

	print

	perm_s = 'BCADFE'
	pnum = to_perm_number(perm_s)
	print 'perm_s = ', perm_s, ', to_perm_number(perm) = ', pnum
	
	perm = to_perm(pnum)
	str0 = 'ABCDEF'
	# alp0 = np.array(list(perm0))
	# perm_s = ''.join(alp0[perm])
	perm_s = perm_to_str(perm, str0)
	print 'perm#: ', pnum, ', perm = ', perm
	print 'perm_to_str(perm, ''ABCDEF'') = ', perm_s
	

