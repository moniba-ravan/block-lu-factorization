CC = gcc
CFLAGS = -Wall -Wextra -fopenmp -O3 

all:lu_serial  parallel_lu

parallel_lu: parallel_lu.c
	$(CC) $(CFLAGS) -o parallel_lu parallel_lu.c

lu_serial: seriel_lu.c
	$(CC) $(CFLAGS) -o seriel_lu seriel_lu.c

try: try.c
	$(CC) $(CFLAGS) -o try try.c

clean:
	rm -f parallel_lu seriel_lu try
