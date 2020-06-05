#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


typedef union intdouble {
    double d;
    uint32_t i[2];
} intdouble;

typedef union intfloat {
    float f;
    uint32_t i;
} intfloat;


float as_float(uint32_t i) {
    union {
        uint32_t i;
        float f;
    } pun = { i };
    
    return pun.f;
}
 
double as_double(uint64_t i) {
    union {
        uint64_t i;
        double f;
    } pun = { i };

    return pun.f;
}

int main(int argc, char *argv[]) {

    intdouble x;
    intfloat y;
    uint32_t i0[] = {1, 1}; 
    uint32_t i1[] = {0, 0};
    int i, n = 2;
    ulong l;
    uint iii;
    
    /* for (i=0; i<n; i++) { */
    /*     x.i0 = i0[i]; */
    /*     x.i1 = i1[i]; */
    /*     /\* x.d = 123456.; *\/ */
    /*     printf("%d %d - %d %d\n", i0[i], i1[i], x.i0, x.i1); */
    /*     printf("x.i0 = 0x%08d, x.i1 = 0x%08d, x.d = %22.16e = %a\n", */
    /*            x.i0, x.i1, x.d, x.d); */
    /* } */
    printf("\n");
    
    /* x.i[0] = 0; */
    /* for (i=0; i<32; i++) { */
    /*     x.i[1] = (uint)1 << i; */
    /*     printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n", */
    /*            x.i[0], x.i[1], x.d, x.d); */
    /* } */
    
    printf("\n");
    printf("----------------------------------------\n");
    printf("\n");
    
    /* x.i[0] = 0; x.i[1] = (uint)1 << 30; */
    /* // x.d = 1.; */
    /* printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n", */
    /*            x.i[0], x.i[1], x.d, x.d); */

    /* x.i[0] = 0; x.i[1] = (uint)1 << 23; */
    x.d = 1.;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    x.d = 2.;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
    x.d = 3.;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
    x.d = 0.5;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
    x.d = 0.25;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
    x.d = 0.1;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
    y.f = 0.f;
    printf("y.i = 0x%08x, y.f = %14.7e = %a\n",
               y.i, y.f, y.f);
    
    y.f = -0.f;
    printf("y.i = 0x%08x, y.f = %14.7e = %a\n",
               y.i, y.f, y.f);
    
    x.d = -1.;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    x.d = -2.;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
    x.d = -3.;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
    x.d = -0.5;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
    x.d = -0.25;
    printf("x.i0 = 0x%08x, x.i1 = 0x%08x, x.d = %22.16e = %a\n",
               x.i[0], x.i[1], x.d, x.d);
    
   
    printf("\n\n");
    printf("Double as two int32:\n");
    printf("\n");
    
    intdouble a;
    double s = 1./256.;
    
    for (i=0; i<=256; i++) {
        a.d = 1. + i*s;
        printf("a.i[1] = 0x%08x, a.i[0] = 0x%08x, a.d = %22.16e = %a\n",
               a.i[1], a.i[0], a.d, a.d);
    }

    printf("\n\n");
    printf("Float as single int32:\n");
    printf("\n");
    
    intfloat b;
    float bs = 1./256.;
    
    for (i=0; i<=256; i++) {
        b.f = 1.f + i*bs;
        printf("b.i = 0x%08x, b.f = %14.7e = %a\n",
               b.i, b.f, b.f);
    }

    printf("\n\n");
    printf("Float as single int32 for mantissa with lead zeros:\n");
    printf("\n");
    

    //b.i = 0b0011;
    for (i=0; i<=22; i++) {
        b.i = (uint32_t)0b0011 << i;
        b.i = 0x3F800000U | b.i;
        printf("b.i = 0x%08x, b.f = %14.7e = %a\n",
               b.i, b.f, b.f);
    }

     

    /* return 0; */

    /* for (i=0; i<10; i++) { */
    /*     y.i = i; */
    /*     printf("%d %d\n", i, y.i); */
    /*     printf("y.i = 0x%08d, y.f = %22.16e = %a\n", */
    /*            y.i, y.f, y.f); */
    /* } */
    
    return 0;
}
