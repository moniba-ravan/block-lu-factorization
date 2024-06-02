#!/bin/bash

# Define parameters
start=1000
end=30000
step=5000
block_size=100
n_threads=4

# Loop through values of N
for (( N = start; N <= end; N += step )); do
    # Execute the program
    echo "Running with N=$N"
    ./parallel_lu $N $block_size $n_threads
done
