#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int i, n;
    double an;
    for (i=0; i<100; i++) {
        an = ceil((-3 + sqrt(9 + 8*i))/2.);
        n = lrint(an);
        printf("i=%d, an=%4.1f, n=%d\n", i, an, n);
    }
}
