help_txt = '''

tree = build_money_change_tree(money, coins)

Build a tree of all the possible exchanges a sum of money with coins
of the denominations given in the coins list. The coins list must be
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


def build_money_change_tree(money, coins):
    print("1. coins = ", coins)
#   if coins:
    coin = coins.pop(0)   # Extract the biggest coin from the list
    niter = money//coin
    print("2. coin = ", coin)

    if coins is []:
        tree = [coin for n in range(niter)]
        print("3. tree = ", tree)
        return tree

    tree = []
    for i in range(niter,0,-1):      # Check all changes: niter .. 1
        leftover = money - i*coin
        coins1 = copy.copy(coins)
        subtree = build_money_change_tree(leftover, coins1)
        print("subtree = ", subtree)
        change = [coin for n in range(i)]
        change.extend(subtree)
        tree.append(change)
        print("4. tree = ", tree)

    return tree
#   else:
#       return []



if __name__ == '__main__':
    
    coins = [2, 1]
    money = 5

    tree = build_money_change_tree(money, coins)








