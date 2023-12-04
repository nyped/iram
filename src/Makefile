CC := gcc
OFLAGS := -Ofast -march=native -funroll-loops -ftree-vectorize
IDIR := include
CFLAGS := -Wall -g -Werror -pedantic $(OFLAGS) -I$(IDIR)
LDFLAGS := -lcblas -lm -llapacke -lbsd
LDFLAGS += -fsanitize=address -pg
EXE := file_test test

.PHONY: all clean clear data

all: $(EXE)

%.o: %.c include/%.h

file_test: file_test.o tools.o arnoldi.o ra.o iram.o qr.o

test: test.o tools.o arnoldi.o ra.o iram.o qr.o

clean:
	rm -f *.o

clear: clean
	rm -f $(EXE)

data: rand_test
	./$< 1000 10 5 1000 | tee convergence.dat
