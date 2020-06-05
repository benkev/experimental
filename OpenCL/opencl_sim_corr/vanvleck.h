/* #include <math.h> */
/* #include <curand_kernel.h> */
 
#define pi M_PI
#define TRUE  1
#define FALSE 0
#define inv_RAND_MAX  1./((double)RAND_MAX + 1.)
#define RANDU()  rand()*inv_RAND_MAX  /* Uniformly distributed in [0..1] */
//#define NS 301    /* Old Number of rh ticks in stdev dimension */
/* #define NS 315    /\* Number of rh ticks in stdev dimension *\/ */
/* #define NR 101    /\* Number of rh ticks in rho dimension *\/ */
/* #define NQS 178   /\* Number of qsig rows: qsig[NQS][NQC] *\/ */
/* #define NQC 5     /\* Number of qsig columns: qsig[NQS][NQC] *\/ */
#ifndef NAN
#  define NAN (0.0/0.0)
#endif

typedef struct calc_corr_input {
    int nseq;
    long nrand;
    int nbit;
    double sx;
    double sy;
    double a;
    double b;
    } calc_corr_input;

typedef struct calc_corr_rnd {
    unsigned long long seed;
    curandState  rndst;
} calc_corr_rnd;

typedef struct calc_corr_result {
    double r;
    double qr;
    double acx;
    double acy;
} calc_corr_result;


//double randn(void);
double vvfun(double xi, double s1, double s2, int nbit);
//double fun(double x, void *params);
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
double mean(int n, double x[]);
double stdev(int n, double x[]);
void log10space(double start, double stop, long np, double sp[]);
void logspace(double base, double start, double stop, long np, double sp[]);
double gauss(void);
long inline locate(size_t n, double tb[], double x);
double lininterp(size_t n, double xx[], double yy[], double x);
double lininterp_vv(size_t n, double qcov[], double rho[], double q);

/* __device__ double adc(double x, int nbit); */
/* __global__ void setup_randgen(calc_corr_rnd rnd[], int nseq); */
/* __global__ void calc_corr(calc_corr_input ci, calc_corr_rnd rnd[],  */
/*                           calc_corr_result cr[], int nseq); */

