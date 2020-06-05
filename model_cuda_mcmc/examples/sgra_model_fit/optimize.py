#
# Optimization module
# Implements unconstrained optimization methods:
# fr():   Fletcher-Reeves method of conjugate gradients
# dfp():  Davidon-Fletcher-Powell method 
#
# linsearch(): Linear search method based on cubic interpolation
#              is used by both.
# Created by Leonid Benkevitch, 13 May 2012
#

from pylab import *
import sys

def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans[::-1])


def grad(x, fun, param):
    """
    Numerical derivatives of fun() at point x
    """
    n = size(x)
    df = zeros(n, float)
    for i in xrange(n):
        xt = float(x[i])
        h = 0.00000001*(1 + abs(xt))
        x[i] = xt + h
        f1 = fun(x, param)
        x[i] = xt - h
        f0 = fun(x, param)
        x[i] = xt
        df[i] =  0.5*(f1 - f0)/h
    return df


def linsearch(x, d, g, fun, param):
    """
    Linear search for the minimun of the function phi(lambda),
       phi(lambda) = f(x+lambda*d)
    using cubic interpolation.
    The search stops at the first lambda that improves
    the function value (i.e. if the function at x+lambda*d
    is smaller than at x).

    Inputs:
      x[n] - initial point
      lam - initial step
      d[n] - direction of the search
      g[n] - gradient of fun() at x
      fun(x,param) - function of n arguments to be minimized along direction d
      param[] - arbitrary parameters of the function fun()

    Outputs:
      Function returns (1) the optimal step lam that provides min f
                       (2) the f value
                       (3) and the error status: 0-OK, 1-
      x[n] - the optimal point, x = x + lam*d. 
      g[n] - the gradient of f at optimal point.
      (Note: after return from linsearch() x and g are already at minimum!)
    """
    err = 0
    maxiter = 5;
    Toler = 0.0001;
    n = size(x)   # Number of dimensions
    p = copy(x)   # Left point p of the minimization interval
    q = copy(x)   # Right point q of the minimization interval
    r = copy(x)   # Potential minimum point r 
    fp = fun(p,param)
    #print 'Inside linsearch: fun(p,param) = %g' % fp
    #print 'Inside linsearch: p = ', p

    #
    # Check if starting from point p we search in a correct direction,
    # i.e. the one generally opposite to that of the gradient. In this
    # case the directional derivative of f(x), Gp < 0.
    # If not, move backwards while Gp > 0.
    # 
    #
    wrongdir = True
    maxpiter = 25;
    niter = 0;
    while niter < maxpiter:      # Direction d is towards f() growth???
        Gp = vdot(g,d)   # Derivative phi'(p). Must be < 0, if correct
        if Gp < 0.:
            wrongdir = False
            break
        #
        # Move backwards until d and g are counterdirected:
        #
        qx = abs(2.*fp/Gp)  # Step
        if qx > 1.: qx = 1.
        p[:] = p - qx*d
        fp = fun(p,param)
        g[:] = grad(p, fun, param)
        print '+++ instability +++'
        niter = niter + 1

    if wrongdir:
        print 'Unable to find step lam minimizing function. Exiting'
        #sys.exit(0)
        err = 1
        return 0, fun(x,param), err

    #
    # Find next point q
    #
    maxqiter = 10;
    niter = 0;
    qfound = False
    hh = min(1., abs(2.*fp/Gp))  # Step
    # Find next q point
    while niter < maxqiter:        
        q[:] = p + hh*d
        fq = fun(q, param)
        g[:] = grad(q, fun, param)
        Gq = vdot(g,d)
        print 'fp = %g, fq = %g, Gp = %g, Gq = %g' % (fp, fq, Gp, Gq)
        if Gq > 0. or fq > fp:  # The opposite point q is good
            qfound = True
            break
        hh = 2.*hh   # Otherwise double the step
        niter = niter + 1

    if not qfound:
        print 'Unable to find point q. Exiting'
        #sys.exit(0)
        err = 2
        return 0, fun(x,param), err

    #
    # Fairly good step hh = |q - p| found. Cubic interpolation:
    #
    maxriter = 20
    niter = 0
    minfound = False
    while niter < maxriter:
        z = 3.*(fp - fq)/hh + Gp + Gq
        ww = z**2 - Gp*Gq
        if ww < 0: ww = 0.  # Prevent floating exception (or imaginary number)
        w = sqrt(ww)
        dd = hh*(1. - (Gq + w - z)/(Gq - Gp + 2.*w)) # Potential minimum
        r[:] = p + dd*d
        fr = fun(r,param)           # fr = phi(dd)
        g[:] = grad(r, fun, param)
        Gr = vdot(g,d)
        print 'dd = ', dd, ', r = ', r, ', fr = ', fr, ', Gr = ', Gr
        if fr <= fp and fr <= fq:   # Minimum found: exit
            hh = dd
            minfound = True
            print 'MinFound; grad = ', g
            break                
        if Gr > 0: # r is a point on the opposite side of the valley
            hh = dd
            q[:] = r
        else: # when Gr <= 0
            hh = hh - dd
            p[:] = r
            Gp = Gr
        niter = niter + 1

    if not minfound:
        print 'Cubic interpolation failed to provide function minimum.'
        
    x[:] = r
    return hh, fr, 0




def linsearch1(x, d, g, fun, param):  # newer version
    """
    Linear search for the minimun of the function phi(lambda),
       phi(lambda) = f(x+lambda*d)
    using cubic interpolation.
    The search stops at the first lambda that improves
    the function value (i.e. if the function at x+lambda*d
    is smaller than at x).

    Inputs:
      x[n] - initial point
      lam - initial step
      d[n] - direction of the search
      g[n] - gradient of fun() at x
      fun(x,param) - function of n arguments to be minimized along direction d
      param[] - arbitrary parameters of the function fun()

    Outputs:
      Function returns (1) the optimal step lam that provides min f
                       (2) the f value
                       (3) and the error status: 0-OK, 1-
      x[n] - the optimal point, x = x + lam*d. 
      g[n] - the gradient of f at optimal point.
      (Note: after return from linsearch() x and g are already at minimum!)
    """
    err = 0
    maxiter = 5;
    Toler = 0.0001;
    n = size(x)   # Number of dimensions
    p = copy(x)   # Left point p of the minimization interval
    q = copy(x)   # Right point q of the minimization interval
    r = copy(x)   # Potential minimum point r 
    fp = fun(p,param)
    #print 'Inside linsearch: fun(p,param) = %g' % fp
    #print 'Inside linsearch: p = ', p

    #
    # Check if starting from point p we search in a correct direction,
    # i.e. the one generally opposite to that of the gradient. In this
    # case the directional derivative of f(x), Gp < 0.
    # If not, move backwards while Gp > 0.
    # 
    #
    wrongdir = True
    maxpiter = 25;
    niter = 0;
    while niter < maxpiter:      # Direction d is towards f() growth???
        Gp = vdot(g,d)   # Derivative phi'(p). Must be < 0, if correct
        if Gp < 0.:
            wrongdir = False
            break
        #
        # Move backwards until d and g are counterdirected:
        #
        qx = abs(0.2*fp/Gp)  # Step
        if qx > 1.: qx = 1.
        p[:] = p - qx*d
        fp = fun(p,param)
        g[:] = grad(p, fun, param)
        print '+++ instability +++'
        niter = niter + 1

    if wrongdir:
        print 'Unable to find step lam minimizing function. Exiting'
        #sys.exit(0)
        err = 1
        return 0, fun(x,param), err

    #
    # Find next point q
    #
    maxqiter = 10;
    niter = 0;
    qfound = False
    hh = min(1., abs(2.*fp/Gp))  # Step
    bb = hh
    # Find next q point
    while niter < maxqiter:        
        q[:] = p + hh*d
        fq = fun(q, param)
        g[:] = grad(q, fun, param)
        Gq = vdot(g,d)
        print 'fp = %g, fq = %g, Gp = %g, Gq = %g' % (fp, fq, Gp, Gq)
        if Gq > 0. or fq > fp:  # The opposite point q is good
            qfound = True
            break
        hh = 2.*hh   # Otherwise double the step
        niter = niter + 1

    if not qfound:
        print 'Unable to find point q. Exiting'
        #sys.exit(0)
        err = 2
        return 0, fun(x,param), err

    #
    # Fairly good step hh = |q - 0| found.
    # Here hh is scalar q; dd is scalar r; 
    # Cubic interpolation:
    #
    maxriter = 20
    niter = 0
    minfound = False
    ee = 1.     # Precision
    while niter < maxriter:
        z = 3.*(fp - fq)/hh + Gp + Gq
        ww = z**2 - Gp*Gq
        if ww < 0: ww = 0.  # Prevent floating exception (or imaginary number)
        w = sqrt(ww)
        dd = hh*(1. - (Gq + w - z)/(Gq - Gp + 2.*w)) # Potential minimum
        r[:] = p + dd*d
        fr = fun(r,param)           # fr = phi(dd)
        g[:] = grad(r, fun, param)
        Gr = vdot(g,d)
        print 'dd = ', dd, ', r = ', r, ', fr = ', fr, ', Gr = ', Gr
        
        if (fr <= fp and fr <= fq) or (abs(Gr) < ee):   # Minimum found: exit
            hh = dd
            minfound = True
            print 'MinFound; grad = ',g,', |grad| = %g, Gr = %g' % (norm(g), Gr)
            break

        if Gr > 0: # r is a point on the opposite side of the valley
            hh = dd
            q[:] = r
            Gq = Gr
        else: # when Gr <= 0
            hh = hh - dd
            p[:] = r
            fp = fr
            Gp = Gr
        niter = niter + 1

    if not minfound:
        print 'Cubic interpolation failed to provide function minimum.'
        
    x[:] = r
    return hh, fr, 0





def dfp(fun, x, param, gtol, maxiter):
    """
    Davidon-Fletcher-Powell optimization method
    """
    itr = 0
    rst = 0
    not_optimum = True

    n = len(x)
    x = array(x, float)
    d = zeros(n, float)
    g = zeros(n, float)
    g1 = zeros(n, float)
    u = zeros(n, float)
    v = zeros(n, float)
    H = eye(n,n) # Initial guess for the Hessian is identity matrix
    vvT = zeros_like(H)
    A = zeros_like(H)
    B = zeros_like(H)
    uTH = zeros_like(x)
    Hu = zeros_like(x)

    while not_optimum:
        g[:] =  grad(x, fun, param)
        gnorm = norm(g)
        if gnorm < 1e-12: # We are on the "horizontal flat surface"
            gnorm = 0.
            f = fun(x, param)
            print 'We are on the "horizontal flat surface"'
            return x, f, gnorm
        
        d[:] = -dot(H,g)       # d = -H*g, new direction
        g1[:] = g

        lam, f, err = linsearch1(x, d, g1, fun, param); 

        print 'Step #%03d: f(x) = %g \nat x = ' % (itr+1, f)
        for i in xrange(n):
            print '%9.2e ' % x[i],
        else: print '\r'

        print 'lam = ', lam
        
        print 'grad(f(x)) = '
        for i in xrange(n):
            print '%9.2e ' % g1[i],
        print ', |grad(f)| = %9.2e' % norm(g1)

        if err or (lam == 0.):
            break
        
        gnorm = norm(g1)
        if gnorm < gtol:
            g[:] = g1
            not_optimum = False
            break

        v[:] = lam*d
        u[:] = g1 - g
        g[:] = g1
        # H = H + (v.v')/(v'.u) - (H.u.u'.H)/(u'.H.u):
        vvT[:] = outer(v,v)
        vTu = dot(v,u)
        A[:] = vvT/vTu
        Hu[:] = dot(H,u)
        uTH[:] = dot(u,H)
        B[:] = outer(Hu,uTH)/dot(uTH,u)
        H[:] = H + A - B
        print 'Inverse Hessian matrix approximation, H^(-1) = '
        for i in xrange(n):
            for j in xrange(n):
                print '%9.2e ' % H[i,j],
            else: print '\r'
        print 'cond(H) = ', cond(H)
        en = eig(H)[0]
        if len(where(imag(en) != 0.)[0]) != 0: # Complex eigenvalues?
            print 'COMPLEX eigenvalues:'
        elif len(where(en <= 0.)[0]) != 0: # Negative or zero eigenvalues?
            print 'NON-POSITIVE definite matrix. Eigenvalues:'
            for i in xrange(n): print '%9.2e ' % en[i],
            print '\r'
            print 'lam = ', lam
            #sys.exit(0)
        else:
            print 'POSITIVE definite matrix. Eigenvalues:'
            for i in xrange(n): print '%9.2e ' % en[i],
            print '\r'
        itr = itr + 1
        if itr > maxiter:
            'Limit on %d iterations exceeded; exiting.' % itr
            break
    
    if not_optimum:
        print '+++ Optimum not found! +++'
        
    return x, f, gnorm





def fr(fun, x, param, gtol, maxiter):
    """
    Fletcher-Reeves optimization method of conjugate gradients
    """
    itr = 0
    rst = 0
    not_optimum = True

    n = len(x)
    x = array(x, float)          # Position vector
    d = zeros(n, float)   # Direction of search vector
    g = zeros(n, float)   # Gradient of f(x)
    g1 = zeros(n, float)  # Next step gradient

    while not_optimum:
        if itr != 0:
            rst = rst + 1
            #print '=== Restart ==='
        g[:] =  grad(x, fun, param)
        d[:] = -g
        for i in xrange(n):
            itr = itr + 1
            g1[:] = g
            lam, f = linsearch(x, d, g1, fun, param); 
            print 'Step #%03d: f(x) = %g at x = ' % (itr+1, f), x,
            print ', grad(f(x)) = ', g
            #print 'Step #%03d: chi^2 = %g at zsp = ' % (itr+1, f), x,
            #print ', grad(f(x)) = ', g
            gnorm = norm(g1)
            if gnorm < gtol:
                g[:] = g1
                not_optimum = False
                break
            d[:] = -g1 + (dot(g1,g1)/dot(g,g))*d # A conjugate direction
            g[:] = g1
            itr = itr + 1
            if itr > maxiter:
                print 'Limit on %d iterations exceeded.' % itr
                break
    if not_optimum:
        print '+++ Optimum not found! +++'

    return x, f, gnorm


