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

def fun(x, param=None): # Rosenbroke
    return 100.*float((x[1] - x[0]**2)**2 + (1. - x[0])**2)

#def fun(x, param=None): # Powell
#    return float((x[0] + 10.*x[1])**2 + 5.*(x[2] - x[3])**2 + \
#                      (x[1] - 2.*x[2])**4 + 10.*(x[0] - x[3])**4)



#
# Function value along direction DeltaX
#

def dirfun(x, DeltaX, lamb, fun, param):
    x = x + lamb*DeltaX;
    y = fun(x, param);
    return y

#
# Numerical derivatives of fun() at point x
#
def grad(x, fun, param):
    n = size(x)
    df = mat(zeros(n, float)).T
    for i in xrange(n):
        xt = float(x[i])
        h = 0.000001*(1 + abs(xt))
        x[i] = xt + h
        f1 = fun(x, param)
        x[i] = xt - h
        f0 = fun(x, param)
        x[i] = xt
        df[i] =  0.5*(f1 - f0)/h
    return df
#
# Linear search for the minimun of the function phi(lambda),
#    phi(lambda) = f(x+lambda*d)
# using cubic interpolation.
# The search stops at the first lambda that improves
# the function value (i.e. if the function at x+lambda*d
# is smaller than at x).
#
# Inputs:
# x[n] - initial point
# lamb - initial step
# d[n] - direction of the search
# g[n] - gradient of fun() at x
# fun(x,param) - function of n arguments to be minimized along direction d
# param[] - arbitrary parameters of the function fun() 
#

def linsearch(x, lamb, d, g, fun, param):
    print '---------linsearch-------'
    maxiter = 5;
    Toler = 0.0001;
    niter = 0;

    p = copy(x)   # Left point p of the minimization interval
    q = copy(x)   # Right point q of the minimization interval
    r = copy(x)   # Potential minimum point r 
    fp = fun(p,param)

    print 'p: ', [p[i,0] for i in (0,1) ]
    print 'q:', [q[i,0] for i in (0,1) ]
    print 'x:', [x[i,0] for i in (0,1) ]
    print 'd:', [d[i,0] for i in (0,1) ]
    
    while niter < maxiter:          # Direction d is towards f() growth???
        Gp = vdot(g,d)   # Derivative phi'(p). Must be < 0, if correct
        print 'g = ', [g[i,0] for i in (0,1) ]
        print 'd = ', [d[i,0] for i in (0,1) ]
        print 'Gp = ', Gp
        if Gp < 0.:
            break
        #
        # Move backwards until d and g are counterdirected:
        #
        qx = abs(2.*fp/Gp)  # Step
        if qx > 1.: qx = 1.
        p[:] = p - qx*d
        fp = fun(p,param)
        g = grad(p, fun, param)
        print '+++ instability +++'
        niter = niter + 1

    #
    # Find next point q
    #
    qx = min(1., abs(2.*fp/Gp))  # Step
    hh = qx
    print 'abs(2.*fp/Gp) = ', abs(2.*fp/Gp)
    while True:         # Find next q point
        bb = hh
        #q[:] = p + bb*d
        q[:] = p + bb*d
        fq = fun(q, param)
        g = grad(q, fun, param)
        Gq = vdot(g,d)
        print 'hh = ', hh
        print 'p:', [p[i,0] for i in (0,1) ], ', fp = ', fp, ', Gp = ', Gp
        print 'q:', [q[i,0] for i in (0,1) ], ', fq = ', fq, ', Gq = ', Gq
        if Gq > 0. or fq > fp:  # The opposite point q is good
            print
            print 'Fairly good step hh = |q - p| found. ',
            print 'Go to cubic interpolation\n'
            break
        hh = 2.*hh   # Otherwise double the step

    #
    # Fairly good step hh = |q - p| found. Cubic interpolation:
    #
    while True:
        z = 3.*(fp - fq)/hh + Gp + Gq
        ww = z**2 - Gp*Gq
        if ww < 0: ww = 0.  # Prevent floating exception (or imaginary number)
        w = sqrt(ww)
        dd = hh*(1. - (Gq + w - z)/(Gq - Gp + 2.*w)) # Potential minimum 
        r[:] = p + dd*d
        fr = fun(r,param)
        g = grad(r, fun, param)
        Gr = vdot(g,d)   
        #print ' = ', hh, ', Gq = ', Gq, ', fp = ', fp, ', fq = ', fq
        print 'r:', [r[i,0] for i in (0,1) ], ', fr = ', fr, ', Gr = ', Gr,
        print ', dd (i.e. lambda) = ', dd      
        if fr <= fp and fr <= fq:   # Minimum found: exit
            hh = dd
            print '=========================>>> Minimum found. Search is over'
            break                
        if Gr > 0: # r is a point on the opposite side of the valley
            print 'r is a point on the opposite side of the valley'
            hh = dd
            q[:] = r
            #break
        else: # when Gr <= 0
            hh = hh - dd
            p[:] = r
            Gp = Gr
            
    return hh


       
    ## bGoOn = True;
    ## while bGoOn:
    ##     if niter > maxiter:
    ##         #lamb = 0;
    ##         break
    ##     h = 0.0001*(1 + abs(lamb));
    ##     f0 = dirfun(x, D, lamb, fun, param);
    ##     fp = dirfun(x, D, lamb+h, fun, param);
    ##     fm = dirfun(x, D, lamb-h, fun, param);
    ##     df1 = 0.5*(fp - fm)/h;
    ##     df2 = (fp - 2*f0 + fm)/h**2;
    ##     dif = df1/df2;
    ##     lamb = lamb - dif;
    ##     if abs(dif) < Toler:
    ##         bGoOn = False;
    ##     niter = niter + 1;
    ##     print 'niter = ', niter, ', lamb = ', lamb
    return lamb


#
# Linear search for the minimun of f(x+lam*d)
# It is reduced to search for zero of f':
#     lambda[k+1] = lambda[k] - f"(k)/f'(k)
#

def linsearch1(x, lamb, D, g, fun, param):
    MaxIt = 20;
    Toler = 0.0001;
    niter = 0;
    bGoOn = True;
    while bGoOn:
        if niter > MaxIt:
            #lamb = 0;
            break
        h = 0.0001*(1 + abs(lamb));
        f0 = dirfun(x, D, lamb, fun, param);
        fp = dirfun(x, D, lamb+h, fun, param);
        fm = dirfun(x, D, lamb-h, fun, param);
        df1 = 0.5*(fp - fm)/h;
        df2 = (fp - 2*f0 + fm)/h**2;
        dif = df1/df2;
        lamb = lamb - dif;
        if abs(dif) < Toler:
            bGoOn = False;
        niter = niter + 1;
        print 'niter = ', niter, ', lamb = ', lamb
    return lamb


## n = 2
## x = [-1.2, 1.]
## #x = [1.5, 2.]
## #x = [-.5, -2.]
## #x = [3., -1., 0., 1.]
## print 'Initial x = ', x, ', n = ', n
## gtol = 0.01
## xtol = 0.00001
## maxiter = 300

def dfp(fun, x, param, gtol, xtol, maxiter):
    """
    Function dfp performs multivariate optimization using the
    Davidon-Fletcher-Powell method.

    Input
    fun - name of the optimized function
    x - array of initial guesses
    param - additional function parameters
    gtol - tolerance for the norm of the slopes
    xtol - delta x tolerance
    maxiter - maximum number of iterations

    Output
    x - array of optimized variables
    F - function value at optimum
    niter - number of iterations
    """
    tr = transpose;  # Just for brevity :)

    n = len(x)
    if type(x) != matrix:
        x = matrix(x).T

    H = mat(eye(n,n)); # Initial guess for the Hessian - identity matrix
    success = False;
    niter = 0;
    # calculate initial gradint
    g1 =  grad(x, fun, param);
    print 'Initial g1 =\n', g1

    while True:
        niter = niter + 1;
        if niter > maxiter:
            break;
        print '%d-th iteration' % niter 

        d = -H*g1;
        d = d/norm(d); # normalize vector d

        lamb = 1;
        lamb = linsearch(x, lamb, d, fun, param);
        # calculate optimum x[] with the given Lambda
        v = lamb*d;
        x = x + v;
        # get new gradient
        g2 =  grad(x, fun, param);
        u = g2 - g1;
        g1 = g2;

        # test for convergence
        if norm(v) < xtol or norm(g1) < gtol:
            success = True;
            break

        # H = H + (v * tr(v)) / (tr(v) * u) - ...
        #     (H * u) * (H * tr(u)) / (tr(u) * H * u);
        vvT = v*tr(v);     # $v v^T$   Gram matrix for v
        vTu = vdot(v,u);   # $v^T u$   float(tr(v)*u);
        H = H + vvT/vTu;
        Hu = H*u;
        HTu = tr(H)*u;
        uTHu = tr(u)*H*u;
        H = H - Hu*tr(HTu)/uTHu;

    F = fun(x, param);
    return x, F, niter



#  -------------- MAIN ------------------------------------------

#def dfp(n, x, gtol, fxtol, dxtol, MaxIter, fun):

n = 2
x = [-1.2, 1.]
#x = [1.5, 2.]
x = [-.5, -2.]
#x = [3., -1., 0., 1.]
param = None
print 'Initial x = ', x, ', n = ', n
gtol = 0.01
dxtol = 0.00001
MaxIter = 300

## """
## Function dfp performs multivariate optimization using the
## Davidon-Fletcher-Powell method.

## Input
## n - number of variables
## x - array of initial guesses 
## gtol - tolerance for the norm of the slopes
## fxtol - tolerance for function value
## dxtol - delta x tolerance
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

H = mat(eye(n,n)); # Initial guess for the Hessian
print 'Initial H =\n', H

success = False;
Iters = 0;
# calculate initial gradint
g1 =  grad(x, fun, param);
print 'Initial g1 =\n', g1

while True:

    Iters = Iters + 1;
    if Iters > MaxIter:
        break;
    print '%d-th iteration' % Iters 

    d = -H*g1;
    d = d/norm(d); # normalize vector d

    lamb = 1;
    lamb = linsearch(x, lamb, d, g1, fun, param);
    #sys.exit(0)
    # calculate optimum x[] with the given Lambda
    v = lamb*d;
    x = x + v;
    # get new gradient
    g2 =  grad(x, fun, param);

    u = g2 - g1;
    g1 = g2;

    print 'x = \n', x
    print 'F(x) =', fun(x, param)
    print 'lamb = ', lamb
    print 'v = \n', v
    print 'dxtol = ', dxtol
    print 'g1 = \n', g1
    print 'norm(g1) = ', norm(g1)
    print 'norm(v) = ', norm(v)
    
    # test for convergence
    if norm(v) < dxtol or norm(g1) < gtol:
        success = True;
        break

    # H = H + (v * tr(v)) / (tr(v) * u) - ...
    #     (H * u) * (H * tr(u)) / (tr(u) * H * u);
    vvT = v*tr(v);     # $v v^T$   Gram matrix for v
    vTu = vdot(v,u);   # $v^T u$   float(tr(v)*u);
    H = H + vvT/vTu;
    Hu = H*u;
    HTu = tr(H)*u;
    uTHu = tr(u)*H*u;
    H = H - Hu*tr(HTu)/uTHu;
    print 'Improved H: \n', H

F = fun(x, param);


print 'Optimum F = %g at x = ' % F, array(x.T)
print '%d iterations' % (Iters-1) 

x = [-.5, -2.]
x, F, niter = dfp(fun, x, param, gtol, dxtol, maxiter)

print 'Optimum F = %g at x = ' % F, array(x.T)
print '%d iterations' % (Iters-1) 
