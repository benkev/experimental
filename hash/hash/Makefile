#
# Makefile for building two demonstrational programs 
# implementing the hash addressing.
#
# hashc (source hash.c) is written in C.
# hashcpp (source hash.cpp) is written in C++.
#


.PHONY: clean

all: hashc hashcpp

hashc:
	gcc -g -Wall  hash.c -o hashc

hashcpp:
	g++ -g -Wall  hash.cpp -o hashcpp

clean:
	rm -f hashc hashcpp \#*\# *\~