Since C99, 13 years now, C has 2D arrays with dynamical bounds. If you want to
avoid that such beast are allocated on the stack (which you should), you can
allocate them easily in one go as the following 

double (*A)[n] = malloc(sizeof(double[n][n]));

and that's it. You can then easily use it as you are used for 2D arrays with
something like A[i][j]. And don't forget that one at the end 

free(A);

Randy Meyers wrote series of articles explaining variable length arrays (VLAs).
