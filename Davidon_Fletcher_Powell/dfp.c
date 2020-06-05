/*
 * Davidon-Fletcher-Powell method
 */
#include <stdio.h>
#include <math.h>
 

/*
 * Numerical derivatives of f() at point x
 */
void deriv(double (*f)(double *), double *x, double *df, int n) {
  /* n is number of dimensions */
  double h, xt, f0, f1;
  int i;

  for (i = 0; i < n; i++) {
    xt = x[i];
    h = 0.000001*(1. + fabs(xt));
    x[i] = xt + h;
    f1 = f(x);
    x[i] = xt - h;
    f0 = f(x);
    df[i] =  0.5*(f1 - f0)/h;
  }
}

int dfp(double (*f)(double *), 
	 double *x, double dxtol, double gtol, int maxiter, int n)
{
  int niter = 0;
  g1 =  deriv(f, x, df, n);
  while (1) {
    symvmul(H, g1, d);  /* d = -H*g1; direction vector */
    normalize(d);       /* d = d/norm(d); #- normalize vector d */
    lamb = 1;
    lamb = linsearch(f, x, lamb, d);
                                 /* v = lamb*d */;
                                  /* x = x + v; */
   }

  return 0;
}


int main() {
  return 0;
}
