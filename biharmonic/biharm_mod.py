#
# biharm_mod.py
#
# This is a Python module.
#
# Solving biharmonic equation, or
#
# the Dirichlet problem for bihatmonic equation del4(U) = V
#
# The problem is reduced to solving a linear system of equationd for
# the interior values:
#
# 20Uij - 8(Ui+1j + Ui-1j + Uij+1 + Uij-1) +
#         2(Ui+1j+1 + Ui-1j+1 + Ui+1j-1 + Ui-1j-1) +
#           Ui+2j + Ui-2j + Uij+2 + Uij-2               = Vij
#
# The stencil is:
#
#                1
#
#           2   -8    2
#
#       1  -8   20   -8   1
#
#           2   -8    2
#
#                1
#
#
from pylab import *
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D


def solve_biharm(G, B):
    """
    Solve biharmonic equation over domain G with borders and sources in B
    Returns copy of B with interior filled with solution

    G[m,n]: the domain. On entry, G must have zeros at the locations where
      the solution is sought (interior), and ones at the border and
      everywhere else. The interior may consist of several non-contiguous
      areas. Example:
      G = array(
      [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
       [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
       
      B[m,n]: border and sources. The border is counted to the depth 2 from
      inside the interior locations set in G to 1 (see the stencil). The
      numbers in B along the border are used as boundary values. All the
      numbers in the interior, set in B to 0, comprise the source.

      Upon return, G contains negatively enumerated border (starting from -1)
      and positively enumerated interior (starting from 1). B is unchanged.
    """
    # Assign a shorter name cat to concatenate():
    cat = concatenate;
    #
    # Enumerate interior and border (no gaps!)
    #
    b = where(G==1)
    p = where(G==0)    
    #r = where(G==1)
    np = len(p[0])
    nb = len(b[0])
    G[p] = arange(np) + 1      # Interior positively numbered
    G[b] = -(arange(nb) + 1)   # Border negatitively numbered


    p = where(G>0)   # Indices of the interior
    np = len(p[0])

    # RHS
    rhs = B[G>0]  # Sources from interior (borders have negative numbers)

    ib = array([],int); # Ordinals of equations with border values at their RHS
    jb = array([],int); # Ordinals of the border values
    v = array([],float) # Coefficients at the border elements

    #
    # Here and further: i changes in  North-South direction,
    #                  j changes in  West-East   direction,
    #

    #
    # Stencil shifts: stencil 
    # and coefficients: stcoef
    #
    stencil = array(((-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), \
                     (1,1), (-1,1), (-2,0), (2,0), (0,-2), (0,2))); 
    stcoef = array((-8., -8., -8., -8., 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0))

    # Connect to themselves with the 20.0 coefficient (diagonal, (i,i))
    ii = G[p]; ji = G[p]; s = repeat(20.0, np)

    for i in xrange(12):
        # Connect ij with ...
        t = stencil[i]
        Q = G[p[0]+t[0],p[1]+t[1]] # stencil-displaced interior
        q = find(Q>0);  # Variables' indices
        r = find(Q<0);  # Border elements' indices
        nq = len(q)  # Number of displaced interior elements
        nr = len(r)  # Number of border elements
        ii = cat((ii, G[p][q])); # Add equation numbers
        ji = cat((ji, Q[q])); # Add variable numbers
        s = cat((s, repeat(stcoef[i],nq)));
        # border
        ib = cat((ib,G[p][r]));  # Add equation numbers closest to the border
        jb = cat((jb,Q[r])); # Add negative -1-border numbers
        v = cat((v, repeat(stcoef[i],nr))); # Coefficients at the border elems


    #
    # Now ib holds numbers of equations, and jb holds corresponding (negative)
    # numbers of the border elements. For example, ib = 8 corresponds two
    # border numbers, jb = -9 and jb = -12. (Because ib = 8 is a corner element
    # of the interior)
    #

    ii -= 1; ji -= 1; # Make row-column indices start from 0
    #D = coo_matrix((s,(ii,ji))) # The matrix of the operator
    D = csr_matrix((s,(ii,ji))) # The matrix of the operator

    #
    # Add the values of the border to RHS
    #
    BGlt0 = B[G<0]  # Represent the border as a 1D array: rowwise
    for i in xrange(1,np+1):  # i runs over all the equation numbers (from 1)
        ieq = where(ib == i)  # indices for all border elems in rhs of ith eq.
        rhsi = sum(v[ieq]*BGlt0[-(jb[ieq]+1)])
        rhs[i-1] -= rhsi

    u = spsolve(D, rhs)

    # Map the solution u onto the grid
    U = zeros_like(B, float)
    U[G>0] = u[G[G>0]-1]
    U[b] = B[b]  # The border is unchanged; save it in U with the solution 
    return U

if __name__ == "__main__": 
    N = 51
    #
    # Create interior (a) and border (b) indices
    #
    G = ones((N,N),int)
    rng = range(N)
    x,y = meshgrid(rng, rng)
    a = where((x>1)&(x<N-2)&(y>1)&(y<N-2)) # Leave at least 2-elem-thick border
    G[a] = 0       # Seed interior with zeros
    G[10,10] = 0       # Seed interior with zeros

    b1 = where((x>18)&(x<32)&(y>17)&(y<33)) 
    G[b1] = 1      # Seed borders with ones
    B = zeros((N,N),float)
    B[b1] = 1.0   # Set internal border to ones, leaving external border at 0.

    B[10,10] = 0.3


    U = solve_biharm(G, B)

    fig = figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(x, y, U, rstride=1, cstride=1, cmap=cm.jet)
    show()


    ## fig = figure()
    ## ax = Axes3D(fig)
    ## ax.plot_surface(x, y, U, rstride=1, cstride=1, cmap=cm.jet)
## show()

