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
import sys, copy
import numpy as np


def build_money_change_tree(money, coins):

    if len(coins) == 0:
        print("Error: the coins list is empty.")
        return []

    
#    print("1. money = ", money, ", coins = ", coins)

    tree = []

    while True:
        coin = coins.pop(0)   # Extract the biggest coin from the list
        niter = money//coin
#        print("2. coin = ", coin, ", niter = ", niter, ", coins = ", coins)

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
# exchanges_from_tree(list_of_trees):
#
#      Returns a 1D-list with all exchanges from the list_of_trees.
#
def exchanges_from_trees(list_of_trees):

    # exchanges:  # 1D list of the exchanges extracted from the tree
    # xchg:       # A single accumulated exchange as a tree branch 
    # i_depth:    # Depth of recursion (for debugging only)

#
# Recursive traversal of the exchange tree
#
    def get_exchange(tree):
        
        nonlocal exchanges, xchg, i_depth

        i_depth += 1  # Depth of recursion (for debugging only)

        i_xchg = len(xchg) # The location in thee branch to cut the tail from
        
        n_elem = len(tree)
        
#        print()
#        print(i_depth,": Entry: n_elem = ", n_elem, ", tree = ", tree,
#              ", xchg = ",xchg)
#        print(i_depth,": 1. exchanges = ", exchanges)

        ints_only = True

        for ie in range(n_elem):
            
#            print()
#            print(i_depth,": for ie; ie = ", ie)

            elem = tree[ie]

            if isinstance(elem, int):
                xchg.append(elem)
            else:
#                print(i_depth,": LIST: elem = ", elem, ", xchg = ", xchg)
                ints_only = False

                get_exchange(elem)

            
#            print(i_depth,": 2. exchanges = ", exchanges)


        if ints_only: # Another branch ready - add it to exchanges
            # Add its copy to the result, keeping xchg in exchanges safe
            exchanges.append(copy.copy(xchg))

#            print(i_depth,": 3. exchanges = ", exchanges)
            
        del xchg[i_xchg:]   # Return to the previous branch state

#        i_depth -= 1        # Depth of recursion (for debugging only)

        return

            
    exchanges = []  # 1D list of the exchanges extracted from the tree
    xchg = []       # A single accumulated exchange as a tree branch 
    # i_xchg = 0      # The location in thee branch to delete the tail from
    i_depth = 0     # Depth of recursion

    
    n_trees = len(list_of_trees)
    for i_tree in range(n_trees):
        tree = list_of_trees[i_tree]
        xchg = []  # A single exchange
#        print("\ni_tree = ", i_tree)
        get_exchange(tree)

    return exchanges





if __name__ == '__main__':

    # money = 5
    # coins = [2, 1]

    # list_of_trees = build_money_change_tree(money, coins)

    # print("list_of_trees = ", list_of_trees)

    # print("\n==========================================\n")

    # money = 6
    # coins = [2, 1]

    # list_of_trees = build_money_change_tree(money, coins)

    # print("list_of_trees = ", list_of_trees)

    # print("\n==========================================\n")

    # money = 6
    # coins = [3, 2, 1]
    # coins1 = copy.copy(coins)  # Backup  

    # list_of_trees = build_money_change_tree(money, coins)

    # print("list_of_trees = ", list_of_trees)

    # print("\n==========================================\n")

    money = 10
    coins = [5, 3]
    # money = 10
    # coins = [5, 2, 1]
    # coins = [5, 3, 2, 1]

    # money = 20
    # coins = [10, 5, 3, 2, 1]

    # money = 100
    # coins = [50, 20, 10, 5, 2, 1]

    coins1 = copy.copy(coins)  # Backup  
    
    list_of_trees = build_money_change_tree(money, coins)
    
#    print("==========================================\n")

    print("money = ", money, ", coins = ", coins1)
    print()
    print("list_of_trees = ")
    for i in range(len(list_of_trees)):
        print("  ", list_of_trees[i])
        print()

        
    exchanges = exchanges_from_trees(list_of_trees)

    n_exchanges = len(exchanges)
    
    print()
    print("exchanges = \n")
    for i in range(len(exchanges)):
        print(sum(exchanges[i]), exchanges[i])

    print()
    print("Total of exchanges: ", n_exchanges)



