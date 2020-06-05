#include <stdio.h>
//#include <stdio.h>

/*
 * From K&R 
 * qsort: sort v[left]...v[right] into increasing order
 */
static void swap(int v[], int i, int j);

static void qsort(int v[], int left, int right) {
    int i, last;

    if (left >= right) /* do nothing if array contains */
	return;
    /* fewer than two elements */
    swap(v, left, (left + right)/2); /* move partition elem */
    last = left;
    /* to v[0] */
    for (i = left + 1; i <= right; i++) /* partition */
	if (v[i] < v[left])
	    swap(v, ++last, i);
    swap(v, left, last);
    /* restore partition elem */
    qsort(v, left, last-1);
    qsort(v, last+1, right);

}


/* 
 *swap: interchange v[i] and v[j] 
 */

static void swap(int v[], int i, int j) {
    int temp;
    temp = v[i];
    v[i] = v[j];
    v[j] = temp;
}


int main(int argc, char *argv[]) {
    
    //int v[] = {6, 53, 49, 9, 90, 33, 91, 84, 5, 1};
    //int v[] = {533, 498};
    //int v[] = {33, 498};
    int v[100];
    int i, n;
    //n = sizeof(v)/sizeof(int);

    for (i=1; i<argc; i++) v[i-1] = atoi(argv[i]);

    if (argc == 1) return 0;

    n = argc - 1;   

    puts(" ");
    puts("Unsorted:");
    for (i=0; i<n; i++) printf("%d  ", v[i]);
    puts("\n");

    printf("n=%d; qsort(v, %d, %d);\n", n, 0, n-1);
    
    qsort(v, 0, n-1);
    
    puts("Sorted:");
    for (i=0; i<n; i++) printf("%d  ", v[i]);
    puts("\n");

    return 0;
}







