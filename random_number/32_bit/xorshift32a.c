#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint32_t state;

uint32_t xorshift32a(void) {
    int s = __builtin_bswap32(state * 1597334677);
    state ^= state << 25;
    state ^= state >> 7;
    state ^= state << 2;
    return state + s;
}

int main(int argc, char *argv[]) {
    uint i, nrand;
    uint32_t rn = 0;
    FILE* fh = fopen("xorshift32a_n1e7.txt", "w");

    if (argc >= 2)
        nrand = atoi(argv[1]);
    else
        nrand = 100;
    
    state = 123456789;
    
    fprintf(fh, "%08u\n", nrand);
    printf("%08u\n", nrand);

    for (i=0; i<nrand; i++) {
        rn = xorshift32a();
        fprintf(fh, "%08u\n", rn);
        //printf("%08u\n", rn);
    }
    fclose(fh);
}
