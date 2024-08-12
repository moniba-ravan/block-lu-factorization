#!/bin/bash


# Matrix sizes
sizes=(3000)


# Block sizes
block_sizes=(100 250 500 1000 3000)


# Number of threads (adjust as needed)
threads=4


# Output directory
output_dir="cachegrind_results"
mkdir -p "$output_dir"


# Iterate over each matrix size
for size in "${sizes[@]}"; do
    echo "Running tests for matrix size $size"


    # Iterate over each block size
    for block_size in "${block_sizes[@]}"; do
        output_file="$output_dir/cachegrind_${size}_${block_size}.out"
        echo "Running cachegrind with matrix size $size and block size $block_size"
        
        # Run Valgrind with Cachegrind
        valgrind --tool=cachegrind --cachegrind-out-file="$output_file" ./parallel_lu "$size" "$block_size" "$threads"
        
        echo "Results saved to $output_file"
    done
done


echo "All tests completed."


