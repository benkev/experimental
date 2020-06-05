#include <stdlib.h>
#include <stdio.h>
//#include <plot.h>
#include "mwac_utils.h"
#include "antenna_mapping.h"


void fill_mapping_matrix() {

    /*
     * The corr_mapping[inp1][inp2] matrix contains internal PFB (ant1, ant2, pol1, pol2) numbers.
     * The internal PFB numbers are used to address data inside of the GPU fits files.
     *
     * If the real-world antenna numbers and polarizations are ant1R, ant2R, pol1R, pol2R, then
     * they may be found inside the gpu fits files as
     *
     * (ant1, ant2, pol1, pol2) = corr_mapping[ant1R*npol + pol1R][ant2R*npol + pol2R]
     * 
     */

	extern map_t corr_mapping[NINPUT][NINPUT];
	extern int pfb_output_to_input[NINPUT];
	extern int single_pfb_mapping[64];
	extern int npol;
	extern int nsta;

	int stn1 = 0, stn2 = 0;
	int pol1 = 0, pol2 = 0;
    int inp1 = 0, inp2 = 0, inp = 0;
	int out1 = 0, out2 = 0;
	int p = 0, npfb = 4;
	
	//	 Output matrix has ordering
	//	 [channel][station][station][polarization][polarization][complexity]

	for (p=0; p<npfb; p++) {
		for (inp=0; inp<64; inp++) {
			pfb_output_to_input[(p*64) + inp] = single_pfb_mapping[inp] + (p*64);
		}
	}

			 
	for (stn1 = 0; stn1 < nsta; stn1++) {
		for (stn2 = 0; stn2 < nsta; stn2++) {
			for (pol1 = 0; pol1 < npol; pol1++) {
				for (pol2 = 0; pol2 < npol; pol2++) {
					out1 = stn1 * npol + pol1;
					out2 = stn2 * npol + pol2;
                    inp1 = pfb_output_to_input[out1];
                    inp2 = pfb_output_to_input[out2];
					/*
                      fprintf(stdout,
                      "stn1 %d pol1 %d stn2 %d pol2 %d map to out1 %d and out2 %d\n",
                      stn1, pol1, stn2, pol2, out1, out2);
                      fprintf(stdout,
                      "these map to PFB input numbers: %d and %d\n", inp1, inp2);
                    */
					corr_mapping[inp1][inp2].stn1 = stn1; // this should give us the pfb input
					corr_mapping[inp1][inp2].stn2 = stn2; // from the real world
                    corr_mapping[inp1][inp2].pol1 = pol1;
                    corr_mapping[inp1][inp2].pol2 = pol2;
				}
			}
		}
	}

}

void get_baseline(int st1, int st2, int pol1, int pol2, float complex *data,
		float complex *baseline) {

	int i, j, k, l, m;
	float complex *in, *out;
	extern int npol;
	extern int nsta;
	extern int nfreq;
	in = data;
	out = baseline;

	for (i = 0; i < nfreq; i++) {
		for (j = 0; j < nsta; j++) {
			for (k = 0; k < nsta; k++) {
				for (l = 0; l < npol; l++) {
					for (m = 0; m < npol; m++) {
						if (j == st1 && k == st2) {
							if (l == pol1 && m == pol2) {

								*out = *in;
								out++;
								// fprintf(stdout,"%f %f\n",crealf(*in),cimagf(*in));
							}

						}
						in++;
					}

				}
			}
		}
	}
}


/* 
 * For ifreq in [0 .. nfreq],
 * write the covariances from data[ifreq, ist1, ist2, ipol1, ipol2]
 * into adjacent words in baseline[ifreq].
 *
 */
void get_baseline_lu(int st1, int st2, int pol1, int pol2, float complex *data,
		float complex *baseline) {

    int i=0;
    float complex *in, *out;	
        
    extern int npol;
    extern int nsta;
    extern int nfreq;

    off_t in_index=0,offset,stride;

    in = data;
    out = baseline;

	/* direct lookup */
    offset = (st1*nsta*npol*npol) + (st2*npol*npol) + (pol1*npol) + pol2; /* [st1,st2,pol1,pol2] */
    stride = (nsta*nsta*npol*npol);
    for (i=0;i<nfreq;i++) {
        in_index = i*stride + offset;
        out[i] = in[in_index];
    }
}

void get_baseline_r(int st1, int st2, int pol1, int pol2, float complex *data,
		float complex *reorder,int npol, int nsta, int nfreq,int true_st1,int true_st2,
		int true_pol1,int true_pol2,int conjugate) {

	int i=0;
	float complex *in, *out;
	size_t out_index =0, in_index=0;;
	in = data;
	out = reorder;
	
/* direct lookup */

	for (i=0;i<nfreq;i++) {

		in_index = i*(nsta*nsta*npol*npol) + (st1*nsta*npol*npol) + 
                     (st2*npol*npol) + (pol1*npol) + pol2;
		out_index = i*(nsta*(nsta+1)*npol*npol/2) + (((true_st1*nsta) - 
                           ((true_st1+1)/2)*true_st1) + true_st2)*npol*npol + (pol1*npol) + pol2;
		if (!conjugate) {
			out[out_index] = in[in_index];
		}
		else {
			if (st2>st1) {
				out[out_index] = conj(in[in_index]);
			}
		}
	}

}
// full reorder using the correct mapping -- 
// takes the input cube and produces a packed triangular output
// in the correct order

// wacky packed tile order to packed triangular

void full_reorder(float complex *full_matrix_h, float complex *reordered)
{


	extern int npol;
	extern int nsta;
	extern int nfreq;
	extern map_t corr_mapping[NINPUT][NINPUT];
	
	int t1=0;
	int t2=0;
	int p1=0;
	int p2=0;

	long long baseline_count = 0;

	for (t1 = 0; t1 < nsta; t1++) {
		for (t2 = t1; t2 < nsta; t2++) {
			for (p1 = 0;p1 < npol;p1++) {
				for (p2 =0; p2 < npol; p2++) {
					baseline_count++;

					int index1 = t1 * npol + p1;
					int index2 = t2 * npol + p2;
					/*
					   fprintf(stdout, "requesting ant1 %d ant 2 %d pol1 %d pol2 %d",
					   antenna1, antenna2, pol1, pol2);
					 */
					map_t the_mapping = corr_mapping[index1][index2];
					int conjugate = 0;
					/*
					   fprintf(stdout,
					   "input ant/pol combination decodes to stn1 %d stn2 %d pol1 %d pol2 %d\n",
					   the_mapping.stn1, the_mapping.stn2, the_mapping.pol1,
					   the_mapping.pol2);
					 */


					if (the_mapping.stn2 > the_mapping.stn1) {
						conjugate = 1;
					}
					else {
						conjugate = 0;
					}

					get_baseline_r(the_mapping.stn1, the_mapping.stn2, the_mapping.pol1,
							the_mapping.pol2, full_matrix_h, reordered, npol, nsta, 
                                   nfreq, conjugate, t1, t2, p1, p2);

				}
			}
		}
	}

	// now reoredered should contain a triagular packed array in the correct order
}



/*
 * Extracts the full matrix from the packed Hermitian form
 */

void extractMatrix(float complex *matrix, float complex *packed) {
	int f, i, j, pol1, pol2;

	extern int npol;
	extern int nsta;
	extern int nfreq;

	for (f = 0; f < nfreq; f++) {
		for (i = 0; i < nsta; i++) {
			for (j = 0; j <= i; j++) {
				int k = f * (nsta + 1) * (nsta / 2) + i * (i + 1) / 2 + j;
				for (pol1 = 0; pol1 < npol; pol1++) {
					for (pol2 = 0; pol2 < npol; pol2++) {
						int index = (k * npol + pol1) * npol + pol2;
						matrix[(((f * nsta + i) * nsta + j) * npol + pol1) * npol + pol2] = 
                            packed[index];
						matrix[(((f * nsta + j) * nsta + i) * npol + pol2) * npol + pol1] = 
                            conjf(packed[index]);
					//	printf("f:%d s1:%d s2:%d %d p1:%d p2:%d %d\n",f,i,j,k,pol1,pol2,index);
					}
				}
			}
		}
	}

}
void extractMatrix_slow(float complex *matrix, float complex *packed) {
	int f, i, j, pol1, pol2;

	extern int npol;
	extern int nsta;
	extern int nfreq;
	int in_index=0;
	int out_index=0;
	int out_index_conj=0;

	for (f = 0; f < nfreq; f++) {
		for (i = 0; i < nsta; i++) {
			for (j = 0; j <= i; j++) {
				for (pol1 = 0; pol1 < npol; pol1++) {
					for (pol2 = 0; pol2 < npol; pol2++) {

						out_index = f*(nsta*nsta*npol*npol) + i*(nsta*npol*npol) + 
                                    j*(npol*npol) + pol1*(npol) + pol2;
						out_index_conj = f*(nsta*nsta*npol*npol) + j*(nsta*npol*npol) + 
                                    i*(npol*npol) + pol1*(npol) + pol2;
 						matrix[out_index] = packed[in_index];
						matrix[out_index_conj] = conjf(packed[in_index]);
						in_index++;
					}
				}
			}
		}
	}

}
