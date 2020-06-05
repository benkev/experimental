OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

all: vector_add.c
	gcc -Wall -g vector_add.c -o vector_add $(OPTIONS)

clean:
	rm -rf vector_add
