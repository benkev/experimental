help_txt = ''' 
How many ways can you exchange $1 for 1, 2, 5, 20, and 50 cent coins?
'''
import numpy as np


#
# Powers in power series (1 + x^
#
a50 = [0, 50, 100]
a20 = [0, 20, 40, 60, 80, 100]
a10 = [i for i in range(0, 110, 10)]
a5 =  [i for i in range(0, 105, 5)]
a2 =  [i for i in range(0, 102, 2)]
a1 =  [i for i in range(0, 101, 1)]

a = np.zeros(101, dtype=int)
a[0] = 1

for i in range(len(a50)):
    p = a50[i]
    for j in range(len(a20)):
        q = a20[j]
        if p+q != 0 and p+q <= 100:
            a[p+q] += 1

a5020 = np.where(a)[0]

print("a = \n", a)
print("a5020 = \n", a5020)

for i in range(len(a5020)):
    p = a5020[i]
    for j in range(len(a10)):
        q = a10[j]
        if p+q != 0 and p+q <= 100:
            a[p+q] += 1

a502010 = np.where(a)[0]

print("a = \n", a)
print("a502010 = \n", a502010)


for i in range(len(a502010)):
    p = a502010[i]
    for j in range(len(a5)):
        q = a5[j]
        if p+q != 0 and p+q <= 100:
            a[p+q] += 1

a5020105 = np.where(a)[0]

print("a = \n", a)
print("a5020105 = \n", a5020105)


for i in range(len(a5020105)):
    p = a5020105[i]
    for j in range(len(a2)):
        q = a2[j]
        if p+q != 0 and p+q <= 100:
            a[p+q] += 1

a50201052 = np.where(a)[0]

print("a = \n", a)
print("a50201052 = \n", a50201052)



for i in range(len(a50201052)):
    p = a50201052[i]
    for j in range(len(a1)):
        q = a1[j]
        if p+q != 0 and p+q <= 100:
            a[p+q] += 1

a502010521 = np.where(a)[0]

print("a = \n", a)
print("a502010521 = \n", a502010521)



