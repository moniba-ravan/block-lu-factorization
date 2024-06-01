CC = gcc
CFLAGS = -Wall -Wextra -fopenmp

all: lufactorization

lufactorization: lufactorization.c
	$(CC) $(CFLAGS) -o lufactorization lufactorization.c

clean:
	rm -f lufactorization
