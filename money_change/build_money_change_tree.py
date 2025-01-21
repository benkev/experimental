help_txt = '''
tree = []
build_money_change_tree(tree, money, noms)

Build a tree of all the possible exchanges a sum of money with coins
of the denominations given in the noms list. The noms list must be
sorted in descending order. The tree must be an empty list.

Examples:

  Exchange 13 cent for 10, 5, 3, 2, 1 cent coins:
    tree = []
    build_money_change_tree(tree, 13, [10, 5, 3, 2, 1])

  Exchange $1 for 1, 2, 5, 20, and 50 cent coins:
    tree = []
    build_money_change_tree(tree, 100, [50, 20, 10, 5, 2, 1])

'''

from pylab import *
import copy

ways = []

def build_money_change_tree(money, noms):

    if len(noms) > 1:
        maxnom = noms.pop(0)
        niter = money//maxnom
        nways = 0

        print('With denomination %d, niter=%d:' % (maxnom, niter))

        for i in range(niter,0,-1):      # Check all changes: niter..1
            leftover = money - i*maxnom
            noms1 = copy.copy(noms)
            nways += build_money_change_tree(leftover, noms1)

            print('maxnom=%d, leftover=%d, nways=%d' % \
                  (maxnom, leftover, nways), ', noms=', noms)

        return nways

    else:
        return 1


if __name__ == '__main__':
    
    noms = [10, 5, 3, 2, 1]
    money = 23

    count = build_money_change_tree(money, noms)

    print("count = ", count)



'''
With denomination 10, niter=2:
With denomination 5, niter=0:
maxnom=10, leftover=3, nways=0 , noms= [5, 3, 2, 1]
With denomination 5, niter=2:
With denomination 3, niter=1:
With denomination 2, niter=0:
maxnom=3, leftover=0, nways=0 , noms= [2, 1]
maxnom=5, leftover=3, nways=0 , noms= [3, 2, 1]
With denomination 3, niter=2:
With denomination 2, niter=1:
maxnom=2, leftover=0, nways=1 , noms= [1]
maxnom=3, leftover=2, nways=1 , noms= [2, 1]
With denomination 2, niter=2:
maxnom=2, leftover=1, nways=1 , noms= [1]
maxnom=2, leftover=3, nways=2 , noms= [1]
maxnom=3, leftover=5, nways=3 , noms= [2, 1]
maxnom=5, leftover=8, nways=3 , noms= [3, 2, 1]
maxnom=10, leftover=13, nways=3 , noms= [5, 3, 2, 1]
count =  3
'''





