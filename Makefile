CC = gcc
CFLAGS = -Wall -Wextra -fopenmp -O3 -ftree-vectorize -mavx -fopt-info-vec
 
all: parallel_lu seriel_lu

parallel_lu: parallel_lu.c
	$(CC) $(CFLAGS) -o parallel_lu parallel_lu.c


lu_serial: seriel_lu.c
	$(CC) $(CFLAGS) -o seriel_lu seriel_lu.c

clean:
	rm -f parallel_lu seriel_lu 
