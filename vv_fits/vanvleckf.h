#include <math.h>
 
#define pi M_PI
#define TRUE  1
#define FALSE 0
#define inv_RAND_MAX  1./((double)RAND_MAX + 1.)
#define RANDU()  rand()*inv_RAND_MAX  /* Uniformly distributed in [0..1] */

// #define NS 315    /* Number of rh ticks in stdev dimension */
// #define NQS 178   /* Number of qsigf rows: qsig[NQS][NQC] */

#define NQC 5     /* Number of qsig columns: qsig[NQS][NQC] */
#define NR 101    /* Number of rh ticks in rho dimension */
/* New sizes: */
#define NS  995    /* Number of rh ticks in stdev dimension */
#define NQS 498    /* Number of qsig rows: qsig[NQS][NQC] */


float randnf(void);
float adcf(float x, int nbit);
int load_rho_tablef(char *dir, float *sig, float *rho, float (*rh)[NS][NR]);
int load_qsig_tablef(char *dir, float (*qsig)[NQC]);
float bilinf(float *sig, float (*rh)[NS][NR], float s1, float s2, int ir);
float corr_estimf(float sig[], float rho[], float (*rh)[NS][NR], \
		  float s1, float s2, float xy);
double vvfun(double xi, double s1, double s2, int nbit);

float qstdf(float sigma, int nbit);
float std_estimf(float (*qsig)[NQC], int nbit, float squant);
int std_estimatef(int nqs, int nqc, float (*qsig)[nqc], int nbit,
				  int nsquant, float *squant, float *sanalog); 
void corr_estimatef(int ns, int nr, float sig[], float rho[], 
		   float (*xytbl)[ns][nr], int nxy, 
		   float s1[nxy], float s2[nxy], 
					float xys[nxy], float cors[nxy]);
float bilinearf(int ns, int nr, float *sig, float (*xy)[ns][nr], 
				float s1, float s2, int ir);
