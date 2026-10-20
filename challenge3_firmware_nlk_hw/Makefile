CC      ?= clang
CFLAGS  := -std=c99 -O2 -Wall -Wextra -Werror -pedantic -ffunction-sections -fdata-sections

all: test_nlk_hw

test_nlk_hw: nlk_hw.c test_nlk_hw.c nlk_hw.h nlk_peripherals.h nlk_errors.h
	$(CC) $(CFLAGS) nlk_hw.c test_nlk_hw.c -o test_nlk_hw

test: test_nlk_hw
	./test_nlk_hw

clean:
	rm -f test_nlk_hw
