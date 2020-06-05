#ifdef __OPENCL_VERSION__
typedef struct rand64State {
    ulong s[4];
} rand64State;

typedef struct calc_corr_input {
    unsigned long seed;
    long nrand;
    int nseq;
    int nbit;
    double sx;
    double sy;
    double a;
    double b;
    } calc_corr_input;

typedef struct calc_corr_result {
    double r;
    double qr;
    double acx;
    double acy;
} calc_corr_result;


#else
typedef struct rand64State {
    cl_ulong s[4];
} rand64State;

typedef struct calc_corr_input {
    cl_ulong seed;
    cl_long nrand;
    cl_int nseq;
    cl_int nbit;
    cl_double sx;
    cl_double sy;
    cl_double a;
    cl_double b;
    } calc_corr_input;

typedef struct calc_corr_result {
    cl_double r;
    cl_double qr;
    cl_double acx;
    cl_double acy;
} calc_corr_result;
#endif
