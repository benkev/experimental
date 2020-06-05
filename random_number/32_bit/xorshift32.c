#include <stdio.h>
#include <stdlib.h>

uint32_t state;


uint32_t xorshift32a(void) {
    int s = __builtin_bswap32(state * 1597334677);
    state ^= state << 25;
    state ^= state >> 7;
    state ^= state << 2;
    return state + s;
}

int main(int argc, char *argv[]) {
}
