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
	g++ -o test test.o

#compile the application
#input: test.cpp
#output: test.o
#-c: compile the input
#-g: include debug information in the output .o file
test.o: test.cpp types.h
	g++ -g -c test.cpp

#remove built files
clean:
	rm -rf test test.o output.txt *~
