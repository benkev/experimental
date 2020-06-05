#include <stdio.h>
#include <stdlib.h>

/*
 * From K&R 
 * qsort: sort v[left]...v[right] into increasing order
 */
static void swap(int v[], int i, int j);

static void qsort1(int v[], int left, int right) {
    int i, last;

    if (left >= right) /* do nothing if array contains */
	return;        /* fewer than two elements */
    
    swap(v, left, (left + right)/2); /* move partition elem */
    last = left;
    /* to v[0] */
    for (i = left + 1; i <= right; i++) /* partition */
	if (v[i] < v[left])
	    swap(v, ++last, i);
    swap(v, left, last);
    /* restore partition elem */
    qsort1(v, left, last-1);
    qsort1(v, last+1, right);

}

static void argqsort(int v[], int idx[], int left, int right) {
    /*
     * From K&R, modified.
     * When sorting array v[], this finction swaps its locations, and 
     * similtaneously it swaps the same locations of array idx[]. 
     * If array idx[] contained sequential integers, 0, 1, 2, ...,
     * upon finishing it will contain the sorting index.
     */
    int i, last, last1, mddl;

    if (left >= right) /* do nothing if array contains */
	return;        /* fewer than two elements */
                       
    /* Here left < right: */
    mddl = (left + right)/2;
    swap(v,   left, mddl); /* move partition elem */
    swap(idx, left, mddl); /* move partition elem in index */
    last = left;
    /* to v[0] */
    for (i = left + 1; i <= right; i++) /* partition */
	if (v[i] < v[left]) {
	    last1 = last + 1;
	    swap(v,   last1, i);
	    swap(idx, last1, i);
	    last = last1;
	}
    swap(v,   left, last);
    swap(idx, left, last);
    /* restore partition elem */
    argqsort(v, idx, left, last-1);
    argqsort(v, idx, last+1, right);

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

static void argsort(int v[], int vs[], int ix[], int n) {
    /*
     * Sorts array v[] of n elements, preserving it in v[].
     * Returns sorted array in vs[], and the sort index in ix[],
     * such that v[ix[j]] = vs[j], j = 0 .. n-1.
     *
     * Arrays vs[] and ix[] of size n must be provided by the caller.
     */
    int i;
    for (i=0; i<n; i++) {
	ix[i] = i;       /* Sequence 0 to n-1 */
	vs[i] = v[i];    /* Copy of the unsorted aray v[] */
    }
    argqsort(vs, ix, 0, n-1);
}




int main(int argc, char *argv[]) {
    
    int v[100];
    int *vs;
    int *ix;
    int i, j, n;

    if (argc == 1) return 0;

    for (i=1; i<argc; i++) {
	v[i-1] = atoi(argv[i]);
    }
    n = argc - 1; 
  
    vs = (int *) malloc(n*sizeof(int));
    ix = (int *) malloc(n*sizeof(int));

    puts(" ");
    puts("Unsorted v[]:");
    for (i=0; i<n; i++) printf("%d  ", v[i]);
    puts("\n");
    puts("Index before sorting:");
    for (i=0; i<n; i++) printf("%d  ", ix[i]);
    puts("\n");

    printf("argsort(v, vs, ix, n);\n\n", n, 0, n-1);
    
    argsort(v, vs, ix, n);

    puts("Sort index :");
    for (i=0; i<n; i++) printf("%d  ", ix[i]);
    puts("\n");
    
    puts("Sorted:");
    for (i=0; i<n; i++) printf("%d  ", vs[i]);
    puts("\n");
    puts("v[ix[i]]:\n");
    for (i=0; i<n; i++) printf("%d  ", v[ix[i]]);
    puts("\n");

    free(ix);
    free(vs);

    return 0;
}







