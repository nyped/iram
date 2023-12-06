CC := gcc
OFLAGS := -Ofast -march=native -funroll-loops -ftree-vectorize
IDIR := include
CFLAGS := -Wall -Werror -pedantic $(OFLAGS) -I$(IDIR)
LDFLAGS := -lcblas -lm -llapacke
LDFLAGS += -fsanitize=address -pg -g
EXE := file_test test

.PHONY: all clean clear data

all: $(EXE)

%.o: %.c include/%.h

file_test: tools.o arnoldi.o ra.o iram.o qr.o

test: test.o tools.o arnoldi.o ra.o iram.o qr.o

clean:
	rm -f *.o

clear: clean
	rm -f $(EXE)
