help_txt = '''

tree = build_money_change_tree(money, coins)

Build a tree of all the possible exchanges a sum of money with coins
of the denominations given in the coins list. The coins list must be
sorted in descending order. The tree must be an empty list.

Examples:

  Exchange 13 cent for 10, 5, 3, 2, 1 cent coins:
    build_money_change_tree(13, [10, 5, 3, 2, 1])

  Exchange $1 for 1, 2, 5, 20, and 50 cent coins:
    build_money_change_tree(100, [50, 20, 10, 5, 2, 1])

'''

# from pylab import *
import copy


def build_money_change_tree(money, coins):
    
#    print("1. money = ", money, ", coins = ", coins)

    tree = []

    while True:
        coin = coins.pop(0)   # Extract the biggest coin from the list
        niter = money//coin
#        print("2. coin = ", coin, ", niter = ", niter)

        if len(coins) == 0:
            tree.append([coin for n in range(niter)])
            return tree

        for i in range(niter,0,-1):      # Check all changes: niter .. 1
            leftover = money - i*coin
            change = [coin for n in range(i)]

#            print("4. leftover = ", leftover, ", change = ", change)

            if leftover == coins[-1]:   # No need to dive into recursion
                change.append(leftover)
            elif leftover != 0:
                coins1 = copy.copy(coins)

                subtree = build_money_change_tree(leftover, coins1)

#                print("subtree = ", subtree)
                
                change.extend(subtree)

            tree.append(change)

#            print("5. tree = ", tree)

    return tree


#
# exchanges_from_tree(tree, xcgs):
#      Create a 1D-list xchg with all exchanges from tree.
#

def exchanges_from_tree(tree, xcgs):
    
    n_elem = len(tree)
    for i in range(n_elem):
        elem = tree[i]
        if isinstance(elem, list):
            exchanges_from_tree(elem, xcgs)
            xcgs.extend(xcg)
        else:
            xcgs.append(elem)

    return





if __name__ == '__main__':

    # money = 5
    # coins = [2, 1]

    # tree = build_money_change_tree(money, coins)

    # print("tree = ", tree)

    # print("\n==========================================\n")

    # money = 6
    # coins = [2, 1]

    # tree = build_money_change_tree(money, coins)

    # print("tree = ", tree)

    # print("\n==========================================\n")

    # money = 6
    # coins = [3, 2, 1]

    # tree = build_money_change_tree(money, coins)

    # print("tree = ", tree)

    # print("\n==========================================\n")

    money = 20
    coins = [10, 5, 3, 2, 1]
    # money = 100
    # coins = [50, 20, 10, 5, 2, 1]

    coins1 = copy.copy(coins)  # Backup  
    
    tree = build_money_change_tree(money, coins)
    
#    print("==========================================\n")

    print("money = ", money, ", coins = ", coins1)
    print()
    print("tree = ")
    for i in range(len(tree)):
        print("  ", tree[i])
        print()


    xcgs = []





