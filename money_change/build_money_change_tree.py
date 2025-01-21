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


def build_money_change_tree(tree, money, noms):

    if while noms:
        maxnom = noms.pop(0)
        niter = money//maxnom

        print('With denomination %d, niter=%d:' % (maxnom, niter))

        for i in range(niter,0,-1):      # Check all changes: niter..1
            leftover = money - i*maxnom
            noms1 = copy.copy(noms)
            build_money_change_tree(leftover, noms1)


        return # nways

    else:
        return # 1


if __name__ == '__main__':
    
    noms = [10, 5, 3, 2, 1]
    money = 13

    tree = []
    
    build_money_change_tree(tree, money, noms)

    print("count = ", count)




