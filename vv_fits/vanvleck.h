#include <math.h>

#ifndef NAN
#  define NAN (0.0/0.0)
#endif
 
#define pi M_PI
#define TRUE  1
#define FALSE 0
#define inv_RAND_MAX  1./((double)RAND_MAX + 1.)
#define RANDU()  rand()*inv_RAND_MAX  /* Uniformly distributed in [0..1] */
//#define NS 301    /* Old Number of rh ticks in stdev dimension */
// #define NS 315    /* Number of rh ticks in stdev dimension */
// #define NQS 178   /* Number of qsig rows: qsig[NQS][NQC] */

#define NQC 5     /* Number of qsig columns: qsig[NQS][NQC] */
#define NR 101    /* Number of rh ticks in rho dimension */
/* New sizes: */
#define NS  995    /* Number of rh ticks in stdev dimension */
#define NQS 498    /* Number of qsig rows: qsig[NQS][NQC] */

double randn(void);
double adc(double x, int nbit);
double vvfun(double xi, double s1, double s2, int nbit);
double fun(double x, void *params);
int inds(double sig[], double s);
int load_rho_table(double *sig, double *rho, double (*rh)[NS][NR]);
int load_qsig_table(double (*qsig)[NQC]);
int load_rho_tablef(float *sig, float *rho, float (*rh)[NS][NR]);
int load_qsig_tablef(float (*qsig)[NQC]);
int load_vvslopes(double (*vvslp)[12][12]);
double bilin(double *sig, double (*rh)[NS][NR], double s1, double s2, int ir);
double corr_estim(double sig[], double rho[], double (*rh)[NS][NR], \
		  double s1, double s2, double xy);

double qstd(double sigma, int nbit);
double std_estim(double (*qsig)[NQC], int nbit, double squant);
double mean(int n, double x[n]);
double std(int n, double x[n]);
void log10space(double start, double stop, long np, double sp[np]);
void logspace(double base, double start, double stop, long np, double sp[np]);
double gauss(void);
size_t inline locate(size_t n, double tb[n], double x);
double lininterp(size_t n, double xx[n], double yy[n], double x);
double inline lininterp_vv(size_t n, double qcov[n], double rho[n], double q);
