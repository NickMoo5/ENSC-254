#compile and link the application
all: test

#run the application
run: test
	./test

#link the application
#input: test.o
#output using -o: test
#-g: include debug information in the output .o file
test: test.o
	gcc -o test test.o

#compile the application
#input: test.c
#output: test.o
#-c: compile the input
#-g: include debug information in the output .o file
test.o: test.c types.h
	gcc -g -c test.c

#remove built files
clean:
	rm -rf test test.o output.txt *~
