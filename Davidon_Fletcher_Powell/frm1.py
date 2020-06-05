#
# Fletcher-Reeves method
#
from pylab import *
import sys
from sys import argv

## def fun(x, n):
##     return float((x[0] - 1)**2 + (x[1] - 2)**2 + 5)
## def derf(x, n):
##     return mat([[float(2*(x[0] - 1))], [float(2*(x[1] - 2))]])

#
# Rosenbroke function:
# z = 100.*((Y - X**2)**2 + (1. - X)**2)
# Minimum at [1,1]
# View:
# xr = linspace(-2,2,100)
# X,Y = meshgrid(xr, xr)
# z = 100.*((Y - X**2)**2 + (1. - X)**2)
# contour(X, Y, z, 1000)

## def fun(x, param=None): # Rosenbroke
##     return 100.*float((x[1] - x[0]**2)**2 + (1. - x[0])**2)

def fun(x, param=None): # Powell
    return float((x[0] + 10.*x[1])**2 + 5.*(x[2] - x[3])**2 + \
                      (x[1] - 2.*x[2])**4 + 10.*(x[0] - x[3])**4)



#
# Function value along direction DeltaX
#

def funEx(x, DeltaX, lamb, fun):
    x = x + lamb*DeltaX;
    y = fun(x);
    return y

#
# Numerical derivative of fun() at point x
#
def deriv(x, fun):
    n = size(x)
    df = mat(zeros(n, float)).T
    for i in xrange(n):
        xt = float(x[i])
        h = 0.000001*(1 + abs(xt))
        x[i] = xt + h
        f1 = fun(x)
        x[i] = xt - h
        f0 = fun(x)
        df[i] =  0.5*(f1 - f0)/h
    return df

#
# Linear search
#

def linsearch(X, N, lamb, D, fun):
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
    f0 = funEx(X, D, lamb, fun);
    fp = funEx(X, D, lamb+h, fun);
    fm = funEx(X, D, lamb-h, fun);
    df1 = 0.5*(fp - fm)/h;
    df2 = (fp - 2*f0 + fm)/h**2;
    print 'f0 = ', f0, ', fp = ', fp,  ', fm = ', fm,  \
          ', df1 = ', df1, ', df2 = ', df2
    dif = df1/df2;
    lamb = lamb - dif;

    if abs(dif) < Toler:
      bGoOn = False;
    
    return lamb


#def dfp(n, x, gradToler, fxToler, DxToler, MaxIter, fun):
n = 4
#x = [-1.2, 1.]
#x = [1.5, 2.]
x = [3., -1., 0., 1.]
print 'Initial x = ', x, ', n = ', n
gradToler = fxToler = 0.0001
DxToler = [0.0001, 0.0001, 0.0001, 0.0001]
MaxIter = 30
    ## """
    ## Function dfp performs multivariate optimization using the
    ## Davidon-Fletcher-Powell method.

    ## Input

    ## n - number of variables
    ## x - array of initial guesses 
    ## gradToler - tolerance for the norm of the slopes
    ## fxToler - tolerance for function
    ## DxToler - array of delta x tolerances
    ## MaxIter - maximum number of iterations
    ## fun - name of the optimized function

    ## Output

    ## x - array of optimized variables
    ## F - function value at optimum
    ## Iters - number of iterations
    ## """
tr = transpose;  # Just for brevity :)

if type(x) != matrix:
    x = matrix(x).T
#if type(DxToler) != matrix:
#    DxToler = matrix(DxToler).T

bGoOn = True;
Iters = 0;
# calculate initial gradint
g1 =  deriv(x, fun);
print 'Initial g1 =\n', g1
#grad1 = tr(grad1);

#sys.exit(0)

while bGoOn:

    Iters = Iters + 1;
    if Iters > MaxIter:
        break;
    print '%d-th iteration' % Iters 

    d = -copy(g1);
    d = d/norm(d); # normalize vector d

    lamb = 1;
    lamb = linsearch(x, n, lamb, d, fun);
    # calculate optimum x[] with the given Lambda
    x = x + lamb*d;

    print 'x = \n', x
    print 'F(x) =', fun(x)
    #print 'lamb = ', lamb
    print 'v = \n', v
    print 'DxToler = ', DxToler
    #print 'u = \n', u
    print 'g1 = \n', g1

    # test for convergence
    for i in xrange(n):
        if abs(v[i]) > DxToler[i]:  # If 
            break

    if norm(g1) < gradToler:
        break

    # H = H + lamb * (v * tr(v)) / (tr(v) * u) - ...
    #     (H * u) * (H * tr(u)) / (tr(u) * H * u);
    vvT = v*tr(v);     # $v v^T$   Gram matrix for v
    vTu = vdot(v,u);   # $v^T u$   float(tr(v)*u);
    H = H + vvT/vTu;
    Hu = H*u;
    HTu = tr(H)*u;
    uTHu = tr(u)*H*u;
    H = H - Hu*tr(HTu)/uTHu;

F = fun(x);

#return x, F, Iters

print 'Optimum F = %g at x = ' % F, array(x.T)
print '%d iterations' % (Iters-1) 

