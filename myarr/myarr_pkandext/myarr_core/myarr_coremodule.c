/*******************************************************
 * myarr_coremodule.c                                       *
 *                                                     *
 * An attempt to build a Python extension module that  *
 * makes use of the NumPy arrays                       *
 *******************************************************/
#include <stdio.h>
#include <Python.h>
#include "/usr/lib/python2.6/site-packages/numpy/core/include/numpy/arrayobject.h"
#include "math.h"

static void minv2(const int N, double a[N][N], double ainv[N][N]);


/*
 * PyArg_ParseTuple() function format string elements
 * O! 
 *    (object) [typeobject, PyObject *]
 *   Store a Python object in a C object pointer. 
 * This is similar to O, but takes two C arguments: the first is 
 * the address of a Python type object, the second is the address 
 * of the C variable (of type PyObject*) into which the object 
 * pointer is stored. If the Python object does not have the 
 * required type, TypeError is raised.
 * :
 *   The list of format units ends here; the string after the colon 
 * is used as the function name in error messages 
 * (the "associated value" of the exception that PyArg_ParseTuple() 
 * raises).
 * NumPy User Guide, r1.6
 */

/*
 * Matrix transpose
 * call from Python:
 * import myarr_core
 * a = array([[1., 2.], [-5., 7.], [4., 6.]])
 * b = myarr_core.trans(a)
 * Note: no conversion is made on input, so the input array must be 
 * STRICTLY 2-dimensional array of doubles! 
 * 
 */
/* static PyObject *tran(PyObject *self, PyObject *args) { */
PyObject *tran(PyObject *self, PyObject *args) {

  PyArrayObject *inMatrix = NULL, *outMatrix = NULL;
  double *pimx = NULL, *pomx = NULL;
  npy_intp dims[2], ncols, nrows, nelem;
  int i, j;
 
  /* Get arguments:  */
  if (!PyArg_ParseTuple(args, "O!:tran", &PyArray_Type, &inMatrix))
    return NULL;

  /* safety checks */
  if (inMatrix == NULL) return NULL;

  //if (PyArray_NDIM(inMatrix) != 2) {  
  if (inMatrix->nd != 2) {  
    PyErr_Format(PyExc_ValueError, "Array has wrong dimension (%d)",
		 inMatrix->nd); 
    return NULL;
  }

  ncols = PyArray_DIM(inMatrix,1); /* dimensions of the input matrix */
  nrows = PyArray_DIM(inMatrix,0); /* dimensions of the input matrix */
  nelem = ncols*nrows;
  pomx = (double *) malloc(nelem*sizeof(double));
  pimx = (double *) PyArray_DATA(inMatrix);

  /* Matrix transpose */
  for (i = 0; i < nrows; i++) {
    for (j = 0; j < ncols; j++) { 
      pomx[j*nrows+i] = pimx[i*ncols+j];
    }
  }

  printf("Input:\n");
  for (i = 0; i < nrows; i++) { 
    for (j = 0; j < ncols; j++) {
      printf("%g ", pimx[i*ncols+j]);
    }
    printf("\n");
  }

  printf("Output:\n");
  for (i = 0; i < ncols; i++) { 
    for (j = 0; j < nrows; j++) {
      printf("%g ", pomx[i*nrows+j]);
    }
    printf("\n");
  }

  dims[0] = ncols;
  dims[1] = nrows;
  /* outMatrix = (PyArrayObject*) PyArray_FromDims(2, dims, NPY_DOUBLE); */
  outMatrix = (PyArrayObject *)
               PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, pomx);
  /* outMatrix->data = pomx; */

  return PyArray_Return(outMatrix); 
  //  return (PyObject *) outMatrix;
}


/*
 * Converting an arbitrary sequence object
 *
 * The main routine for obtaining an array from any Python object that 
 * can be converted to an array is PyArray_FromAny. This function is 
 * very flexible with many input arguments. Several macros make it easier
 * to use the basic function. PyArray_FROM_OTF is arguably the most useful 
 * of these macros for the most common uses. It allows you to convert an 
 * arbitrary Python object to an array of a specific builtin data-type 
 * ( e.g. float), while specifying a particular set of requirements ( e.g. 
 * contiguous, aligned, and writeable). The syntax is
 *
 * PyObject *PyArray_FROM_OTF(PyObject* obj, int typenum, int requirements)
 *
 * Return an ndarray from any Python object, obj, that can be converted to 
 * an array. The number of dimensions in the returned array is determined 
 * by the object. The desired data-type of the returned array is provided 
 * in typenum which should be one of the enumerated types. The requirements 
 * for the returned array can be any combination of standard array flags. 
 * Each of these arguments is explained in more detail below. You receive a
 * new reference to the array on success. On failure, NULL is returned and 
 * an exception is set. 
 *
 * obj
 *     The object can be any Python object convertable to an ndarray. 
 * If the object is already (a subclass of) the ndarray that satisfies 
 * the requirements then a new reference is returned. Otherwise, a new 
 * array is constructed. The contents of obj are copied to the new array 
 * unless the array interface is used so that data does not have to be 
 * copied. Objects that can be converted to an array include: 
 * 1) any nested sequence object, 
 * 2) any object exposing the array interface, 
 * 3) any object with an __array__ method (which should return an ndarray), 
 * and 
 * 4) any scalar object (becomes a zero-dimensional array). Sub-classes 
 * of the ndarray that otherwise fit the requirements will be passed 
 * through. If you want to ensure a base-class ndarray, then use 
 * NPY_ENSUREARRAY in the requirements flag. A copy is made only if 
 * necessary. If you want to guarantee a copy, then pass in NPY_ENSURECOPY
 * to the requirements flag.
 *
 * typenum
 *
 *     One of the enumerated types or NPY_NOTYPE if the data-type should 
 * be determined from the object itself. The C-based names can be used:
 *    NPY_BOOL, NPY_BYTE, NPY_UBYTE, NPY_SHORT, NPY_USHORT, NPY_INT,
 *    NPY_UINT, NPY_LONG, NPY_ULONG, NPY_LONGLONG, NPY_ULONGLONG,
 *    NPY_DOUBLE, NPY_LONGDOUBLE, NPY_CFLOAT, NPY_CDOUBLE,
 *    NPY_CLONGDOUBLE, NPY_OBJECT.
 * Alternatively, the bit-width names can be used as supported on the 
 * platform. For example:
 *    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64, NPY_UINT8, NPY_UINT16,
 *    NPY_UINT32, NPY_UINT64, NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64,
 *    NPY_COMPLEX128.
 * The object will be converted to the desired type only if it can be 
 * done without losing precision. Otherwise NULL will be returned and 
 * an error raised. Use NPY_FORCECAST in the requirements flag to 
 * override this behavior.
 *
 * requirements
 *    The memory model for an ndarray admits arbitrary strides in each 
 * dimension to advance to the next element of the array. Often, however, 
 * you need to interface with code that expects a C-contiguous or a 
 * Fortran-contiguous memory layout. In addition, an ndarray can be 
 * misaligned (the address of an element is not at an integral multiple 
 * of the size of the element) which can cause your program
 * to crash (or at least work more slowly) if you try and dereference 
 * a pointer into the array data. Both of these problems can be solved 
 * by converting the Python object into an array that is more "well behaved"
 * for your specific usage. 
 *
 * The requirements flag allows specification of what kind of array is 
 * acceptable. If the object passed in does not satisfy this requirements 
 * then a copy is made so that the returned object will satisfy the 
 * requirements. Thus ndarray can use a very generic pointer to memory. 
 * This flag allows specification of the desired properties of the 
 * returned array object. All of the flags are explained in the
 * detailed API chapter. 
 *
 * The flags most commonly needed are NPY_IN_ARRAY, NPY_OUT_ARRAY,
 * and NPY_INOUT_ARRAY:
 *
 * NPY_IN_ARRAY
 *    Equivalent to NPY_CONTIGUOUS | NPY_ALIGNED. This combination of 
 *    flags is useful for arrays that must be in C-contiguous order 
 *    and aligned. These kinds of arrays are usually input arrays 
 *    for some algorithm.
 * NPY_OUT_ARRAY
 *    Equivalent to NPY_CONTIGUOUS | NPY_ALIGNED | NPY_WRITEABLE. 
 *    This combination of flags is useful to specify an array that is 
 *    in C-contiguous order, is aligned, and can be written to as well. 
 *    Such an array is usually returned as output (although normally 
 *    such output arrays are created from scratch).
 * NPY_INOUT_ARRAY
 *    Equivalent to NPY_CONTIGUOUS | NPY_ALIGNED | NPY_WRITEABLE |
 *    NPY_UPDATEIFCOPY. This combination of flags is useful to specify 
 *    an array that will be used for both input and output. If a copy 
 *    is needed, then when the temporary is deleted (by your use of 
 *    Py_DECREF at the end of the interface routine), the temporary 
 *    array will be copied back into the original array passed in. Use 
 *    of the UPDATEIFCOPY flag requires that the input object is already 
 *    an array (because other objects cannot be automatically updated
 *    in this fashion). If an error occurs use PyArray_DECREF_ERR (obj) 
 *    on an array with the NPY_UPDATEIFCOPY flag set. This will delete 
 *    the array without causing the contents to be copied back into the 
 *    original array. Other useful flags that can be OR’d as additional 
 *    requirements are:
 * NPY_FORCECAST
 *    Cast to the desired type, even if it can’t be done without losing 
 *    information.
 * NPY_ENSURECOPY
 *    Make sure the resulting array is a copy of the original.
 * NPY_ENSUREARRAY
 *    Make sure the resulting object is an actual ndarray and not a 
 *    sub-class.
 *
 * Note: Whether or not an array is byte-swapped is determined by the 
 * data-type of the array. Native byte-order arrays are always 
 * requested by PyArray_FROM_OTF and so there is no need for a 
 * NPY_NOTSWAPPED flag in the requirements argument. There is also no 
 * way to get a byte-swapped array from this routine.
 * NumPy User Guide, r1.6
 */



/*
 * Matrix inversion 
 * call from Python:
 *
 * import myarr_core
 * a = array([[1., 2., -5.], [-1., 0., 7.], [4., -3., 6.]])
 * a
 * array([[ 1.,  2., -5.],
 *       [-1.,   0.,  7.],
 *       [ 4.,  -3.,  6.]])
 * b = myarr_core.minv(a)
 * b
 * array([[ 0.28378378,  0.04054054,  0.18918919],
 *        [ 0.45945946,  0.35135135, -0.02702703],
 *        [ 0.04054054,  0.14864865,  0.02702703]])
 *
 * The input argument if minv() can be of any type convertible to 
 * array of doubles. This is ensured by the use of conversion 
 * routine (or macro)  PyArray_FROM_OTF(). Thus, the following 
 * sequence of Python commands will work, too:
 *
 * b = [[1, 2, -5], [-1, 0, 7], [4, -3, 6]] # a list with integers
 * b
 * [[1, 2, -5], [-1, 0, 7], [4, -3, 6]]
 * c = myarr_core.minv(b)
 * c
 * array([[ 0.28378378,  0.04054054,  0.18918919],
 *        [ 0.45945946,  0.35135135, -0.02702703],
 *        [ 0.04054054,  0.14864865,  0.02702703]])
 *
 */
/* static PyObject *minv(PyObject *self, PyObject *args) { */
PyObject *minv(PyObject *self, PyObject *args) {

  PyObject *inArg = NULL; /* Any object convertible to ndarray */
  PyArrayObject *outMatrix = NULL, *inMatrix = NULL;
  double *pimx, *pomx;
  npy_intp dims[2], ncols, nrows, nelem;
  int i, j;

  /* Get arguments:  */
  if (!PyArg_ParseTuple(args, "O:minv", &inArg))
    return NULL;

  /* safety check */
  if (inArg == NULL) return NULL;

  /* 
   * Convert anything to matrix of doubles 
   */
  inMatrix = (PyArrayObject *)
              PyArray_FROM_OTF(inArg, NPY_DOUBLE, NPY_IN_ARRAY);

  if (inMatrix == NULL) {  
    PyErr_Format(PyExc_ValueError, 
		 "The argument cannot be converted to an array"); 
    return NULL;
  }

  /* safety check */
  if (PyArray_NDIM(inMatrix) != 2) {  
    PyErr_Format(PyExc_ValueError, "Array has wrong dimension (%d)",
		 (int)PyArray_NDIM(inMatrix)); 
    return NULL;
  }

  ncols = PyArray_DIM(inMatrix,1); /* dimensions of the input matrix */
  nrows = PyArray_DIM(inMatrix,0); /* dimensions of the input matrix */

  if (ncols != nrows) {  
    PyErr_Format(PyExc_ValueError, 
		 "Array is not square: nrows = %d, ncols = %d",
		 (int)nrows, (int)ncols); 
    return NULL;
  }

  nelem = ncols*nrows;
  pomx = (double *) malloc(nelem*sizeof(double));
  pimx = (double *) PyArray_DATA(inMatrix);

  minv2(nrows, pimx, pomx); /* Call the actual inversion routine */

  dims[0] = nrows;
  dims[1] = nrows;

  /* outMatrix = (PyArrayObject*) PyArray_FromDims(2, dims, NPY_DOUBLE); */
  outMatrix = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, pomx);
    /* outMatrix->data = pomx; */

  return (PyObject *) PyArray_Return(outMatrix); 
  //  return (PyObject *) outMatrix;
}


/*
 * The second argument passed in to the Py_InitModule function is 
 * a structure that makes it easy to to define functions in
 * the module. In the example given above, the myarr_core_methods
 * structure would have been defined earlier in the file 
 * (usually right before the init{name} subroutine) to:
 * NumPy User Guide, r1.6
 */

/* ==== methods table ====================== */
static PyMethodDef myarr_core_methods[] = {
  {"tran", tran, METH_VARARGS, "Matrix transpose: use tran(A)"},
  {"minv", minv, METH_VARARGS, "Matrix inverse: use minv(A)"},
  {NULL, NULL, 0, NULL}
};

/*
 * There is exactly one function that must be defined in your C-code 
 * in order for Python to use it as an extension module.
 * The function must be called init{name} where {name} is the name of 
 * the module from Python.
 *    Here: initmyarr_core().
 * This function must be declared so that it 
 * is visible to code outside of the routine. Besides adding the methods 
 * and constants you desire, this subroutine must also contain calls 
 * to import_array() and/or import_ufunc() depending on which C-API 
 * is needed. Forgetting to place these commands will show itself as 
 * an ugly segmentation fault (crash) as soon as any C-API subroutine 
 * is actually called. It is actually possible to have multiple 
 * init{name} functions in a single file in which case multiple modules 
 * will be defined by that file. However, there are some tricks to get 
 * that to work correctly and it is not covered here.
 * NumPy User Guide, r1.6
 */
/* ==== Initialize ====================== */
PyMODINIT_FUNC initmyarr_core()  {
	Py_InitModule("myarr_core", myarr_core_methods);
	import_array();  // for NumPy
}










// 
// Matrix inversion  
//  
static void minv2(const int N, double a[N][N], double ainv[N][N]) {
  // 
  // Matrix inversion by solving N systems of linear equations 
  //     a*ainv = I 
  // for ainv, where a is NxN matrix, and I is the identity matrix 
  // (all zeroes except the diagonal elements, which are ones)
  // Input:
  //   N: system size;
  //   a: matrix N by N
  // Output:
  //   a: inverse of a.
  //
  // This is a pure Gauss algorithm. The program does not check if the 
  // matrix a is well-conditioned. It does not check if a diagonal element 
  // is zero or not before division. The purpose of this program is 
  // "SPEED at the expence of REAIABILITY", whether you like it or not :)
  //
  int i, j, k, l, kp1, Nm1;
  double c, akk;
  double eye[N][N]; // Identity matrix 

  //
  // Prepare the identity matrix
  //
  for(i = 0; i < N; i++) 
    for(j = 0; j < N; j++) 
      if (i == j) eye[i][j] = 1.0; else eye[i][j] = 0.0;

  //
  // Reduce system to upper-triangle form
  //
  Nm1 = N - 1;
  for(k = 0; k < Nm1; k++) {
    kp1 = k + 1;
    akk = a[k][k]; // Just to save time not accessing the a array
    for(i = kp1; i < N; i++) {
      c = a[i][k]/akk;
      for(j = kp1; j < N; j++) {
	a[i][j] -= c*a[k][j];
      }
      for(l = 0; l < N; l++) eye[i][l] -= c*eye[k][l];
    }
  }

  //
  // Back substitution run
  //
  for(l = 0; l < N; l++) 
    ainv[Nm1][l] = eye[Nm1][l]/a[Nm1][Nm1]; // Find the last roots
  
  for(i = Nm1-1; i >= 0; i--) {
    for(l = 0; l < N; l++) {
      c = 0.0;
      for(j = i+1; j < N; j++) {
	c = c + a[i][j]*ainv[j][l];
      }
      ainv[i][l] = (eye[i][l] - c)/a[i][i];
    }
  }
}












