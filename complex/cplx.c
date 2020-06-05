#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <tgmath.h>
//#define PI 3.14159265

int main(int argc, char *argv[]) {
    //complex double a, b, c;
    double PI = acos(-1);
    double complex a, b, c;

    a = -I*PI/4.;
    b = exp(a);
    c = cexp(a);
    printf("Re[b]=%g, Im[b]=%g\n", creal(b), cimag(b));
    printf("Re[c]=%g, Im[c]=%g\n", creal(c), cimag(c));
    printf("Re[I]=%g, Im[I]=%g\n", creal(I), cimag(I));
    printf("sizeof(I)=%ld, sizeof(cexp(I))=%ld, sizeof(exp(I))=%ld\n", 
	   sizeof(I), sizeof(cexp(I)), sizeof(exp(I)));
    
    double complex z = cexp(I * PI/4.); // Euler's formula
    printf("exp(i*pi) = %.1f%+.1fi\n", creal(z), cimag(z));

    return 0;
}
