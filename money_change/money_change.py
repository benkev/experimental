help_txt = '''

list_of_trees = money_change(money, coins)

Build a tree of all the possible changes a sum of money with coins
of the denominations given in the coins list. The coins list must be
sorted in descending order. The tree must be an empty list.

Examples:

  Change 13 cent for 10, 5, 3, 2, 1 cent coins:
    build_money_change_trees(13, [10, 5, 3, 2, 1])

  Change $1 for 1, 2, 5, 20, and 50 cent coins:
    build_money_change_trees(100, [50, 20, 10, 5, 2, 1])

'''

# from pylab import *
import sys, copy
import numpy as np


def build_money_change_trees(money, coins):

    if len(coins) == 0:
        print("Error: the coins list is empty.")
        return []

    list_of_trees = []

    while True:
        coin = coins.pop(0)   # Extract the biggest coin from the list
        niter = money//coin

        if len(coins) == 0:
            list_of_trees.append([coin for n in range(niter)])
            return list_of_trees

        for i in range(niter,0,-1):      # Check all changes: niter .. 1
            leftover = money - i*coin
            change = [coin for n in range(i)]

            if leftover == coins[-1]:   # No need to dive into recursion
                change.append(leftover)
            elif leftover != 0:
                coins1 = copy.copy(coins)
                subtree = build_money_change_trees(leftover, coins1)
                change.extend(subtree)

            list_of_trees.append(change)

    return list_of_trees


#
# changes_from_trees(list_of_trees):
#
#   Parsing the list of change trees.
#   Returns a 1D-list with all changes from the list_of_trees.
#
def changes_from_trees(list_of_trees):

    # change_list:    # 1D list of the changes extracted from the tree
    # single_change:  # A single accumulated change as a tree branch 
    # i_depth:        # Depth of recursion (for debugging only)

#
# Recursive traversal of a single change tree
#
    def get_change(tree):
        
        nonlocal change_list, single_change, i_depth

        #i_depth += 1  # Depth of recursion (for debugging only)

        i_chg = len(single_change) # Loc in the branch to cut the tail from
        n_elem = len(tree)
        ints_only = True

        for ie in range(n_elem):
            
            elem = tree[ie]

            if isinstance(elem, int):
                single_change.append(elem)
            else:
                ints_only = False
                get_change(elem)

        if ints_only: # Another branch ready - add it to change_list
            # Add its copy to the result,
            # keeping single_change in change_list safe
            change_list.append(copy.copy(single_change))

        del single_change[i_chg:]   # Return to the previous branch state

        # i_depth -= 1        # Depth of recursion (for debugging only)

        return

            
    change_list = []  # 1D list of the changes extracted from the tree
    single_change = []       # A single accumulated change as a tree branch 
    i_depth = 0     # Depth of recursion

    
    n_trees = len(list_of_trees)
    for i_tree in range(n_trees):
        tree = list_of_trees[i_tree]
        single_change = []  # A single change
        get_change(tree)

    return change_list





if __name__ == '__main__':

    # money = 5
    # coins = [2, 1]

    # list_of_trees = build_money_change_trees(money, coins)

    # print("list_of_trees = ", list_of_trees)

    # print("\n==========================================\n")

    # money = 6
    # coins = [2, 1]

    # list_of_trees = build_money_change_trees(money, coins)

    # print("list_of_trees = ", list_of_trees)

    # print("\n==========================================\n")

    # money = 6
    # coins = [3, 2, 1]
    # coins1 = copy.copy(coins)  # Backup  

    # list_of_trees = build_money_change_trees(money, coins)

    # print("list_of_trees = ", list_of_trees)

    # print("\n==========================================\n")

    #money = 10
    # coins = [5, 3]
    # money = 10
    #coins = [5, 2, 1]
    # coins = [5, 3, 2, 1]

    # money = 20
    # coins = [10, 5, 3, 2, 1]

    money = 100
    coins = [50, 20, 10, 5, 2, 1]

    coins1 = copy.copy(coins)  # Backup  
    
    list_of_trees = build_money_change_trees(money, coins)
    
#    print("==========================================\n")

    print("money = ", money, ", coins = ", coins1)
    print()
    print("list_of_trees = ")
    for i in range(len(list_of_trees)):
        print("  ", list_of_trees[i])
        print()

        
    change_list = changes_from_trees(list_of_trees)

    n_changes = len(change_list)
    
    print()
    print("change_list = \n")
    for i in range(len(change_list)):
        print(sum(change_list[i]), change_list[i])

    print()
    print("Total of changes: ", n_changes)



