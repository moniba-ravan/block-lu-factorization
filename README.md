# Block LU-Factorization

## Overview

This repository contains the implementation of Block LU-Factorization in C using OpenMP for parallel computation. Block LU-Factorization extends the traditional LU factorization by splitting matrices into smaller blocks, enabling parallel computation and efficient memory utilization.


## Experiments

Various experiments were conducted to test the implementation's performance on different matrix sizes and block sizes, using different numbers of threads for parallel execution. The results showed significant performance improvements with parallel implementation.



### Dependencies

- GCC compiler with OpenMP support
- Python (for initial serial implementation and comparison)

### Usage
Compile and Run the executable:
    
    
    ./parallel_lu N block_size n_threads
    
    ./seriel_lu N 
    
    

### Acknowledgements

This project is part of the High Performance and Parallel Computing course (1TD064) at Uppsala University.

