////////////////////////////////////////////////////////////////////////////////
// File: minimize_down_the_line.c                                             //
// Routines:                                                                  //
//    Minimize_Down_the_Line                                                  //
////////////////////////////////////////////////////////////////////////////////

#include <float.h>                          // required for DBL_EPSILON
#include <math.h>                           // required for sqrt()

struct arg_list {
   double* x;
   double* v;
   double (*f)(double*);
   double *y;
   int n;
};

static struct arg_list *args = 0;

//                        Externally Defined Routines 

int Min_Search_Parabolic_Extrapolation(double (*f)(double), double x[],
               double fx[], double a, double b, double cut_off_scale_factor);
int Min_Search_Parabolic_Interpolation(double (*f)(double), double x[],
                                             double fx[], double tolerance );
void Ray_in_n_Space(double v[], double initial_pt[], double direction[],
                                                    double distance, int n );

//                        Internally Defined Routines 

static double g( double lambda );

////////////////////////////////////////////////////////////////////////////////
//  int Minimize_Down_the_Line(double (*f)(double *), double x[], double fx,  //
//                       double *p, double v, double y[], double cutoff,      //
//                       double cutoff_scale_factor, double tolerance, int n) //
//                                                                            //
//  Description:                                                              //
//     Given a function f:R^n -> R, a point x in R^n, and a direction v       //
//     this function minimizes g(p) = f(x - pv) for 0 < p <= cutoff.  In      //
//     particular, if v is the gradient of f, then as long as v != 0, this    //
//     function searches for a minimum of f in the direction opposite to that //
//     of the direction of the grad f.                                        //
//                                                                            //
//     This routine uses parabolic extrapolation to bracket a local minimum   //
//     followed by parabolic interpolation to refine the estimate.            //
//                                                                            //
//  Arguments:                                                                //
//     double (*f)(double *)                                                  //
//        Pointer to a user-defined function of a real n-dimensional vector   //
//        returning a real number (type double).                              //
//     double x[]                                                             //
//        On input, x[] is the initial point to begin the search for a        //
//        minimum of f(x - pv) with respect to p for p > 0.  The array x      //
//        should be dimensioned n in the calling routine.                     //
//     double fx                                                              //
//        On input, fx is the value of f(x[]).                                //
//     double *p                                                              //
//        On output, *p is the value of p which minimizes f(x - pv) for p > 0.//
//     double v[]                                                             //
//        On input, v is an "unnormalized" direction.  The search starts at   //
//        the point x in the direction -v.  The array v should be dimensioned //
//        n in the calling routine.                                           //
//     double y[]                                                             //
//        On output, y is the point x - pv where 0 < p <= cutoff is the value //
//        of p which minimizes f(x - pv).  The array y should be dimensioned  //
//        n in the calling routine.                                           //
//     double cutoff                                                          //
//        The upper bound of the parameter p.                                 //
//     double cutoff_scale_factor                                             //
//        A parameter which limits the displacement in any single step during //
//        the parabolic extrapolation phase.                                  //
//     double tolerance                                                       //
//        A parameter which controls the termination of the procedure.        //
//        The procedure is terminated when the relative length of the interval//
//        of uncertainty to the magnitude of its mid-point is less than or    //
//        equal to tolerance.  If a nonpositive number is passed, tolerance   //
//        is set to sqrt(DBL_EPSILON).                                        //
//     int    n                                                               //
//        On input, n is the dimension of x, v, and y.                        //
//                                                                            //
//  Return Values:                                                            //
//                                                                            //
//  Example:                                                                  //
//     extern double f(double*);                                              //
//     #define N                                                              //
//     double x[N], double v[N], double y[N];                                 //
//     double p, fx;                                                          //
//     double cutoff, cutoff_scale_factor, tolerance;                         //
//                                                                            //
//     (your code to create vectors x, v, evaluate fx = f(x), set cutoff,)    //
//     (cutoff_scale_factor and the tolerance.                           )    //
//                                                                            //
//     Minimize_Down_the_Line(f, x, fx, &p, v, y, cutoff, cutoff_scale_factor,//
//                                                             tolerance, n); //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

int Minimize_Down_the_Line(double (*f)(double *), double x[], double fx, 
                          double *p, double v[], double y[], double cutoff, 
                          double cutoff_scale_factor, double tolerance, int n)
{
   struct arg_list g_args;
   struct arg_list *old_args;
   double lambda[3];
   double g_lambda[3];
   int k = 0;
    
   g_args.x = x;
   g_args.v = v;
   g_args.f = f;
   g_args.y = y;
   g_args.n = n;

                            // Bracket the minimum 

   lambda[0] = 0.0;
   lambda[2] = 1.0;
   old_args = args;
   args = &g_args;
   g_lambda[0] = fx;
   g_lambda[2] = g(lambda[2]);
   *p = (g_lambda[0] < g_lambda[2]) ? 0.0 : 1.0;
   args = &g_args;
   k = Min_Search_Parabolic_Extrapolation(g, lambda, g_lambda, 0.0, cutoff,
                                                         cutoff_scale_factor);
   if ( k < 0 ) k = 0;
   else { 
                        // Interpolate for the minimum 
     // The minimum is located between lambda[0] < lambda[1] < lambda[2],
     // where g(lambda[1]) < g(lambda[0]) and g(lambda[1]) < g(lambda[2]),
     // i.e. f(x - lambda[0]*v) > f(x - lambda[1]*v) < f(x - lambda[2]*v).

      if (tolerance <= 0.0) tolerance = sqrt(DBL_EPSILON);
      args = &g_args;
      k = Min_Search_Parabolic_Interpolation(g, lambda, g_lambda, tolerance );
      *p = 0.5 * (lambda[0] + lambda[2]);
   }

   Ray_in_n_Space(y, x, v, -(*p), n );
   args = old_args;
   return k;
}


////////////////////////////////////////////////////////////////////////////////
//  static double g( double lambda )                                          //
//                                                                            //
//  Description:                                                              //
//                                                                            //
//  Return Values:                                                            //
//     g(lambda) = f(x - lambda * v)                                          //
////////////////////////////////////////////////////////////////////////////////
static double g( double lambda ) 
{
   Ray_in_n_Space(args->y, args->x, args->v, -lambda, args->n );

   return (args->f)(args->y);
}
