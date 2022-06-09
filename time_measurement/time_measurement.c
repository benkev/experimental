/*
 * clock() returns the number of clock ticks since the program started 
 *   executing. On Linux, you will get CPU time, while on Windows you will 
 *   get wall time. Therefore, here, in Linux, it does not count the sleep()
 *   time. 
 * time() only returns full seconds since the Epoch time in UTC only.
 *
 * time_t is actually the same as long int, so you can print it directly 
 *   with printf().
 *
 */

#include <stdio.h>
#include <time.h>
#include <unistd.h>

int main(){
    clock_t stop, start = clock();
    clock_t tic, toc;

    time(&tic);
    
    sleep(2.78);
    
    stop = clock();

    time(&toc);
    
    double elapsed = (double)(stop - start)*1000.0 / CLOCKS_PER_SEC;
    double cps = (double) CLOCKS_PER_SEC;
    
    
    printf("Time elapsed in ms: %f\n", elapsed);
    printf("CLOCKS_PER_SEC: %f\n", cps);

    printf("Time measured: %ld seconds.\n", toc - tic);
    
}




