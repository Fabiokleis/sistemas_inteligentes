CC = clang
CFLAGS = -std=c11 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wswitch-enum -g
LDFLAGS = -lm

all: gen temp

.PHONY: all

run-gen: gen
	./gen

run-temp: temp
	./temp

gen.o: gen.c
	$(CC) $(CFLAGS) -c $<

gen: gen.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o gen gen.o

temp.o: temp.c
	$(CC) $(CFLAGS) -c $<

temp: temp.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o temp temp.o

clean:
	rm -f gen temp *.o
