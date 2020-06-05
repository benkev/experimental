#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <locale.h>
#include <float.h>
#include <math.h>

int main() {

    printf("\n");
    printf("FLT_EPSILON = %e\n", FLT_EPSILON);
    printf("DBL_EPSILON = %e\n", DBL_EPSILON);
    printf("log(DBL_EPSILON) = %e\n", log(DBL_EPSILON));
    printf("log(DBL_MIN) = %e\n", log(DBL_MIN));
    printf("log(DBL_MIN/1e10) = %e\n", log(DBL_MIN/1e10));
    printf("\n");
    
    return 0;
}
