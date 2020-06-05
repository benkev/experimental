#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
/* #include <limits.h> */


uint64_t s[4];

uint64_t next(void);
void jump(void);
void long_jump(void);

int main() {
    
    uint64_t seed = 12345;
    /* extern uint64_t s[4]; */

    s[0] = seed;
    printf("State: %21lu ", s[0]);
    for (int i=1; i<4; i++) {
        s[i] = s[i-1]*69069;
        printf("%21lu ", s[i]);
    }
    printf("\n\n");

    for (int i=0; i<100; i++) {
        printf("%3d %21lu \n", i, next());

    }
    
    return 0;
}
