#
# list_change_nabor.py
#
# Find all the possible combinatuions of coins of the denominations from
# noms list to make a sum given in money
#                 AND
# Find how many variants are there to change a sum of money with the
# coins of several denominationa listed in noms.
#
# noms must be a list of coin denominations in descending order. Example:
#      noms = [10, 5, 3, 2, 1]
#
# nabor must be an empty list. During the descent down a particular branch
#   of the tree, nabor is appended with the number of coins of a particular
#   denomination. At a tree leaf (a terminal branch) nabor contains exactly
#   the same number of multipliers as the number of denominations in noms_const.
#
# comb_list: on input must be an empty list, on output it will be a list of
#   all the possible combinations of coins from noms making the sum equal
#   to the number in money.
#
#
#


from pylab import *
import copy

def countChange(money, noms, nabor, comb_list):

    if len(noms) > 1:
        
        maxnom = noms.pop(0)   # Cut the head of the nom list, the biggest coin
        
        ncoins = money//maxnom
        nways = 0

        for i in xrange(ncoins,-1,-1):      # Check all changes: ncoins..0
            nabor.append(i)
            leftover = money - i*maxnom
            nways += countChange(leftover, noms, nabor, comb_list) # Recursion
            nabor.pop()

        noms.insert(0, maxnom) # Stitch back the head to return the same noms
        
        return nways

    else:
        if money % noms[0] == 0: # The sum can be made of entire # of coins
            ncoins = money//noms[0]

            nabor.append(ncoins)

            # print 'nabor = ', nabor
            # print 'comb_list = ', comb_list

            comb_list.append(copy.copy(nabor))

            nabor.pop()
            return 1
        
        else:          # The sum CANNOT be made of entire # of noms[0] coins
            return 0



if __name__ == '__main__':
    
    noms = [10, 5, 3, 2, 1]
    #noms = [50, 20, 10, 5, 3, 2, 1]

    money = 23
    #money = 100
    nabor = []        # Must be empty list
    comb_list = []    # Must be empty list
    count = countChange(money, noms, nabor, comb_list)

    combs = array(comb_list, dtype=int)

    print 'All the possible combinations of coins with denominations noms ='
    print '', noms
    print 'are stored in the combs[%d,%d] array of ints: ' % (count, len(noms))
    print combs
    
    print "The sum %d" % money,
    print " can be picked using coins with the denominations ", noms,
    print "in %d ways."% count

