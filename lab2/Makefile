SOURCES := utils.c disasm.c emulator.c riscv.c
HEADERS := types.h utils.h riscv.h
PWD := $(shell pwd)
CUNIT := -L $(PWD)/CUnit-install/lib -I $(PWD)/CUnit-install/include -llibcunit
CFLAGS := -g  -Wall

all: riscv

riscv: $(SOURCES) $(HEADERS)
	gcc $(CFLAGS) -o $@ $(SOURCES)

test-utils: test_utils.c utils.c $(HEADERS)
	gcc $(CFLAGS) -DTESTING -o test-utils test_utils.c utils.c $(CUNIT)
	./test-utils
	rm -f test-utils

clean:
	rm -f riscv
	rm -f *.o *~
	rm -f test-utils
	rm -f code/out/*.solution code/out/*/*.solution
	rm -f code/out/*.trace code/out/*/*.trace

deepclean: clean
	rm -rf CUnit-install
