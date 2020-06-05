#
# count_change.py
#
# Find how many variants are there to change a sum of money with the
# coins of several denominationa listed in noms.
#
# noms must be a list of coin denominations in descending order. Example:
#      noms = [10, 5, 3, 2, 1]
#
# The count of variants is returned.
#

from pylab import *
import copy

def countChange(money, noms):

    if len(noms) > 1:
        maxnom = noms.pop(0)   # Cut the head of the nom list, the biggest coin
        ncoins = money//maxnom
        nways = 0

        for i in xrange(ncoins,-1,-1):      # Check all changes: ncoins..0
            leftover = money - i*maxnom
            nways += countChange(leftover, noms)   # Recursion!!!

        noms.insert(0, maxnom) # Stitch back the head to return the same noms
        
        return nways

    else: # The very last denomination left. Can we pick the sum of this coin?
        
        if money % noms[0] == 0: # The sum can be made of entire # of coins
            return 1
        else:  # The sum CANNOT be made of entire # of noms[0] coins
            return 0


if __name__ == '__main__':
    
    noms = [10, 5, 3, 2, 1]
    #noms = [10, 3, 1]
    money = 3

    count = countChange(money, noms)

    print "The sum %d" % money,
    print " can be picked using coins with the denominations ", noms,
    print "in %d ways."% count

    
    
