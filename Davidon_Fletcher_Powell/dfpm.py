from pylab import *
import sys
from sys import argv

def fun(x, n):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + 5


def dfp(N, X, gradToler, fxToler, DxToler, MaxIter, myFx):
    """
    Function dfp performs multivariate optimization using the
    Davidon-Fletcher-Powell method.

    Input

    N - number of variables
    X - array of initial guesses 
    gradToler - tolerance for the norm of the slopes
    fxToler - tolerance for function
    DxToler - array of delta X tolerances
    MaxIter - maximum number of iterations
    myFx - name of the optimized function

    Output

    X - array of optimized variables
    F - function value at optimum
    Iters - number of iterations
    """
    tr = transpose;  # Just for brevity :)

    if type(X) != matrix:
        X = matrix(X).T
    if type(DxToler) != matrix:
        DxToler = matrix(DxToler).T
        
    B = mat(eye(N,N));

    bGoOn = True;
    Iters = 0;
    # calculate initial gradient
    grad1 =  FirstDerivatives(X, N, myFx);
    grad1 = tr(grad1);

    while bGoOn:

        Iters = Iters + 1;
        if Iters > MaxIter:
            break;

        print 'grad1 =', grad1
        S = -B*grad1;
        S = tr(S) / norm(S); # normalize vector S

        lamb = 1;
        print 'myFx.__name__ = ', myFx.__name__
        lamb = linsearch(X, N, lamb, S, myFx);
        # calculate optimum X[] with the given Lamb
        d = lamb * S;
        X = X + d;
        # get new gradient
        grad2 =  FirstDerivatives(X, N, myFx);

        grad2 = tr(grad2);
        g = grad2 - grad1;
        grad1 = grad2;

        # test for convergence
        for i in xrange(N):
            if abs(d(i)) > DxToler(i):
                break

        if norm(grad1) < gradToler:
            break

        # B = B + lamb * (S * tr(S)) / (tr(S) * g) - ...
        #     (B * g) * (B * tr(g)) / (tr(g) * B * g);
        x1 = (S * tr(S));
        x2 = (S * g);
        B = B + lamb * x1 * 1 / x2;
        x3 = B * g;
        x4 = tr(B) * g;
        x5 = tr(g) * B * g;
        B = B - x3 * tr(x4) / x5;

    F = myFx(X, N);

    return X, F, Iters

#
# Function value along direction DeltaX
#

def myFxEx(N, X, DeltaX, lamb, myFx):
    X = X + lamb * DeltaX;
    y = myFx(X, N);
    return y

#
# Numerical derivative of myFx() at point x
#

def FirstDerivatives(X, N, myFx):
#    FirstDerivX = zeros(N, float)
    derivX = mat(zeros(N, float)).T
    #print '1 derivX = ', derivX
    for iVar in xrange(N):  
        #print '2 derivX = ', derivX
        xt = X[iVar];
        h = 0.01 * (1 + abs(xt));
        X[iVar] = xt + h;
        fp = myFx(X, N);
        X[iVar] = xt - h;
        fm = myFx(X, N);
        X[iVar] = xt;
        #print '3 derivX = ', derivX
#        FirstDerivX[iVar] = (fp - fm) / 2 / h;    
        derivX[iVar] = (fp - fm) / 2 / h;    

    return derivX

#
# Linear search
#

def linsearch(X, N, lamb, D, myFx):
    MaxIt = 100;
    Toler = 0.000001;
    
    iter = 0;
    bGoOn = True;
    while bGoOn:
        iter = iter + 1;
        if iter > MaxIt:
            lamb = 0;
            break
   
    h = 0.01 * (1 + abs(lamb));
    print 'D = ', D
    print 'type(D) = ', type(D)
    f0 = myFxEx(N, X, D, lamb, myFx);
    fp = myFxEx(N, X, D, lamb+h, myFx);
    fm = myFxEx(N, X, D, lamb-h, myFx);
    deriv1 = (fp - fm) / 2 / h;
    deriv2 = (fp - 2 * f0 + fm) / h**2;
    diff = deriv1 / deriv2;
    lamb = lamb - diff;

    print 'f0 = ', f0, ', fp = ', fp, ', fm = ', fm, \
          ', deriv1 = ', deriv1, ',  deriv2 = ', deriv2, \
          'diff = ', diff, ', lamb = ', lamb
    
    if abs(diff) < Toler:
      bGoOn = False;
    
    return lamb

