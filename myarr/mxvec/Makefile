#
# Makefile for libmxv.a, a small matrix-vector library
# used in the raytrace Python extension module
# Created March 15, 2011 by Leonid Benkevitch. 
#
CFLAGS = -g # -Wall
#CPPFLAGS =
LDLIBS = -lm

CC = gcc

MXVOBJS=\
cross_product.o dot_product.o linsolve.o linsolve2.o maxval.o minval.o \
minv.o minv2.o mmul.o mvmul.o sum.o sum_squares.o print1d.o print2d.o

install: libmxv.a

libmxv.a: $(MXVOBJS)
	ar cvr libmxv.a $(MXVOBJS)
	ranlib libmxv.a
	mv libmxv.a   ..
clean:
	 rm -f *.o *.a
