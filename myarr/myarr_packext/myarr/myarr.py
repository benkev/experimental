import numpy as np
import myarr

class Fatal_Error(): pass  # Exception for all fatal errors

class linsys():
    
    """
    Linear system of equations with square matrix
    """
    def __init__(self, matr, rhs=None):
        """
        
        """
        self.XCP = Fatal_Error()
        self.matr = matr
        self.rhs = rhs
        self.cond = np.linalg.cond(matr) # Matrix conditional number
        self.neqn = len(matr[:,0])
        self.inv = myarr.core.minv(matr)         # Matrix inverse

        # Check sizes
        if len(matr[0,:]) != self.neqn:
            raise self.XCP   # Non-square matrix
        if self.rhs and len(rhs[:]) != self.neqn:
            raise self.XCP   # RHS and matrix differ in size

    def solve(self, rhs=None):
        if (rhs == None) and (self.rhs == None):
            raise self.XCP   # No rhs
        else:
            self.rhs = rhs
        x = np.empty(self.neqn, dtype=np.double) # Solution
        x[:] = 0.0
        for i in xrange(self.neqn):
            for j in xrange(self.neqn):
                x[j] += self.matr[i,j]*self.rhs[j]
        return x
