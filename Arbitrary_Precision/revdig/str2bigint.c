#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

uint bmul(uint a, uint b[], int nd);
uint badd(uint a, uint b[], int nd);

int main(int argc, char *argv[]) {

    // char str[] = "65535";
    // int ns = 5, nd = 2;
    char str[] = "34143735";
    int ns = 8, nd = 2;
    uint b[2] = {0, 0};
    uint a = 10, s, c; 
    ulong bl;

    // for (int i = ns-1; i > -1; i--) {
    for (int i = 0; i < ns; i++) {
        s = str[i] - '0';
        printf("i = %d, s = %d, str[i] = %c, bl = %lu\n", i, s, str[i], bl);

        c = bmul(a, b, nd);
        for (int id=nd-1; id>-1; id--) printf("%5u  ", b[id]);
        printf(" -- mul\n");
        
        c = badd(s, b, nd);
        for (int id=nd-1; id>-1; id--) printf("%5u  ", b[id]);
        printf(" -- add\n");
    }
    
    bl = 65536*b[1] + b[0];
    printf("c = %d, bl = %lu\n\n", c, bl);


    return 0;

}



uint badd(uint a, uint b[], int nd) {

    uint c = 0; // Carry
    uint bc;    // = b + c

    b[0] += a;    
    for (int i = 0; i < nd; i++) {
        bc = b[i] + c;
        b[i] = bc % 65536;
        c =    bc / 65536;
    }
    
    return c;   
}



uint bmul(uint a, uint b[], int nd) {
    
    uint c = 0; // Carry
    uint abc;   // = a*b + c

    for (int i = 0; i < nd; i++) {
        abc = a * b[i] + c;
        b[i] = abc % 65536;
        c =    abc / 65536;
    }
    
    return c;   
}

