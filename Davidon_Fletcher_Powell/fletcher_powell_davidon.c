////////////////////////////////////////////////////////////////////////////////
// File: fletcher_powell_davidon.c                                            //
// Routines:                                                                  //
//    Fletcher_Powell_Davidon                                                 //
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>                       // required for malloc() and free()

//                        Externally Defined Routines 

void Subtract_Vectors(double w[], double u[], double v[], int n);
void Copy_Vector(double *d, double *s, int n);
double Inner_Product(double u[], double v[], int n);
int Minimize_Down_the_Line(double (*f)(double *), double x[], double fx, 
                          double *p, double v[], double y[], double cutoff,
                        double cutoff_scale_factor, double tolerance, int n);
void Identity_Matrix_ut(double *A, int n);
double* Multiply_Sym_Matrix_by_Vector_ut(double *u, double *A, double* v,
                                                                      int n);
//                        Internally Defined Routines 

static void Update_H(double *H, double* dx, double* dg, double* Hg, int n);

////////////////////////////////////////////////////////////////////////////////
//  int Fletcher_Powell_Davidon( double (*f)(double *),                       //
//            void (*df)(double*, double*),                                   //
//            int (*Stopping_Rule)(double*, double*, int, int),               //
//            double *a, double *dfa, double line_search_cutoff,              //
//            double line_search_cutoff_scale_factor,                         //
//            double line_search_tolerance, int n)                            //
//                                                                            //
//  Description:                                                              //
//     The Fletcher-Powell-Davidon method is a Quasi-Newton method for        //
//     locating a local minimum of a function f:R^n -> R.  Given an initial   //
//     point a in R^n and setting the inverse of the Hessian of f at the      //
//     location of the minimum to the identity matrix, the method then uses   //
//     the current estimate of the inverse of the Hessian of f and grad f     //
//     at the current point x to determine a search direction, then minimizes //
//     the function f along the line in the search direction.  The inverse of //
//     the Hessian is then updated and a new search direction is calculated.  //
//     After n searches, the inverse of the Hessian is reset to the identity  //
//     matrix and the process is repeated until halted by the user-defined    //
//     stopping rule.                                                         //
//                                                                            //
//  Arguments:                                                                //
//     double (*f)(double *)                                                  //
//        Pointer to a user-defined function of a real n-dimensional vector   //
//        returning a real number (type double).                              //
//     void (*df)(double*, double*)                                           //
//        The gradient of the user-defined function f above. "df(a,dfa)"      //
//        calculates the gradient of the function f:R^n -> R at the point a   //
//        and returns the gradient in the array dfa.                          //
//     int (*Stopping_Rule)(double*, double*, int, int),                      //
//        Pointer to a user-defined function which controls the iteration of  //
//        the Fletcher-Powell-Davidon method.  If the stopping rule returns a //
//        non-zero integer, then the process is halted; otherwise if it       //
//        returns a zero, then the process continues iterating.               //
//        The arguments are: the difference between two successive estimates  //
//        of the location of the minimum, the value of the gradient at the    //
//        current estimate of the location of the minimum, the total number of//
//        iterations performed, and n, the dimension of x.                    //
//     double *a                                                              //
//        On input, a is the initial point to start the iteraction. On output,//
//        a is the final point before the iteration is halted.                //
//     double *dfa                                                            //
//        On output, *dfa is the vector df(a) = grad f(a).                    //
//     double line_search_cutoff                                              //
//        The maximum value of the parameter p for doing a line search        //
//        in the direction opposite to that of grad f starting at the point a.//
//        I.e. search for the min{ f(a - pv): 0 < p <= line_search_cutoff},   //
//        where v = grad f.                                                   //
//     double line_search_cutoff_scale_factor                                 //
//        A parameter which limits the displacement in any single step during //
//        the parabolic extrapolation phase of the search for a minimum of f  //
//        along the line segment a - p grad f, 0 < p <= line_search_cutoff.   //
//     double line_search_tolerance                                           //
//        A parameter which controls the termination of the line-search       //
//        procedure.  The line-search is terminated when the relative length  //
//        of the interval of uncertainty to the magnitude of its mid-point is //
//        less than or equal to tolerance.  If a nonpositive number is passed,//
//        tolerance is set to sqrt(DBL_EPSILON).                              //
//     int    n                                                               //
//        The dimension of the argument of f(), the dimension of a, etc.      //
//                                                                            //
//  Return Values:                                                            //
//    -1    Not enough memory.                                                //
//    Other The return value of the user-supplied Stopping_Rule.              //
//                                                                            //
//  Example:                                                                  //
//     extern double f(double*);                                              //
//     extern void df(double*, double*);                                      //
//     extern int Stopping_Rule(double*, double*, int, int);                  //
//     #define N                                                              //
//     double dfa, line_search_cutoff, line_search_cutoff_scale_factor;       //
//     double line_search_tolerance;                                          //
//     double a[N];                                                           //
//                                                                            //
//     (your code to initialize the vector a, set line_search_cutoff, )       //
//     (line_search_cutoff_scale_factor, and line_search_tolerance.   )       //
//                                                                            //
//                                                                            //
//     err = Fletcher_Davidon_Powell(f, df, Stopping_Rule, a, &dfa,           //
//                      line_search_cutoff, line_search_cutoff_scale_factor,  //
//                                                line_search_tolerance, N);  //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

int Fletcher_Powell_Davidon( double (*f)(double *), 
                           void (*df)(double*, double*),
                           int (*Stopping_Rule)(double*, double*, int, int),
                           double *a, double *dfa, double line_search_cutoff,
                           double line_search_cutoff_scale_factor,
                           double line_search_tolerance, int n )
{
   double *x;
   double *dx;
   double *dg;
   double *H;
   double *Hg;
   double *v;
   double p;
   int iteration = 0;
   int count_between_resets = n;
   int err = 0;

   x = (double *) malloc( n * sizeof(double) );
   v = (double *) malloc( n * sizeof(double) );
   dx = (double *) malloc( n * sizeof(double) );
   dg = (double *) malloc( n * sizeof(double) );
   Hg = (double *) malloc( n * sizeof(double) );
   H = (double *) malloc( ((n * (n + 1) ) >> 1) * sizeof(double) );
 
   if (H == NULL) err = -1;
   else {
      df(a, dfa);
      do {
         iteration++;
         count_between_resets++;
         if (count_between_resets > n) {
            Identity_Matrix_ut( H, n );
            count_between_resets = 1;
         }
         Multiply_Sym_Matrix_by_Vector_ut(v, H, dfa, n);
         Minimize_Down_the_Line(f, a, f(a), &p, v, x, line_search_cutoff, 
                   line_search_cutoff_scale_factor, line_search_tolerance, n);
         Subtract_Vectors(dx, x, a, n);
         Copy_Vector(a, x, n);
         Copy_Vector(x, dfa, n);
         df(a, dfa);
         Subtract_Vectors(dg, dfa, x, n);
         Update_H(H, dx, dg, Hg, n);
         err = Stopping_Rule(dx, dfa, iteration, n);
      } while ( !err );
   }
   free(Hg);
   free(H);
   free(dg);
   free(dx);
   free(v);
   free(x);

   return err;
}


static void Update_H(double *H, double* dx, double* dg, double* Hg, int n)
{
   double *Hdg;
   double dxtdg;
   double dgtHdg;
   double *pH;
   int i, j;

   dxtdg = Inner_Product(dx, dg, n);
   Multiply_Sym_Matrix_by_Vector_ut(Hg, H, dg, n);
   dgtHdg = Inner_Product(dg, Hg, n);

   dxtdg = 1.0 / dxtdg;
   dgtHdg = 1.0 / dgtHdg;

   pH = H;
   for (i = 0; i < n; i++)
      for (j = i; j < n; j++)
         *pH++ += dxtdg * dx[i] * dx[j] - dgtHdg * Hg[i] * Hg[j];
}
