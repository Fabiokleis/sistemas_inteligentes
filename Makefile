CC = clang
CFLAGS = -std=c11 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wswitch-enum -g
LDFLAGS = -lm

all: gen

.PHONY: gen

run: gen
	./gen

main.o: main.c
	$(CC) $(CFLAGS) -c $<

gen: main.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o gen main.o

clean:
	rm -f gen *.o
