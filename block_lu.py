import numpy as np
import time
import random
import sys


def lu(A):
    """
    Perform serial LU decomposition on matrix A and overwrite A.
    
    Parameters:
    - A: numpy.ndarray, input matrix to be decomposed
    
    Returns:
    None
    """
    n = A.shape[0]
    for k in range(n):
        for i in range(k + 1, n):
            A[i, k] = A[i, k] / A[k, k]
        
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] -= A[i, k] * A[k, j]

def back_substitution(U, Y):
    """
    Perform back substitution to solve XU = Y for X,
    where U is an upper triangular matrix, and overwrite Y.
    
    Parameters:
    - U: numpy.ndarray, upper triangular matrix
    - Y: numpy.ndarray, right-hand side matrix
    
    Returns:
    None
    """
    num_vectors, m = Y.shape
    for row in range(num_vectors):
        Y[row, 0] /= U[0, 0]
        for j in range(1, m):
            y = Y[row, j]
            for i in range(0, j):
                y -= U[i, j] * Y[row, i]
            Y[row, j] = y / U[j, j]

def forward_substitution(L, Y):
    """
    Perform forward substitution to solve LX = Y for X,
    where L is a lower triangular matrix, and overwrite Y.
    
    Parameters:
    - L: numpy.ndarray, lower triangular matrix
    - Y: numpy.ndarray, right-hand side matrix
    
    Returns:
    None
    """
    n, num_vectors = Y.shape
    for col in range(num_vectors):
        for i in range(1, n):
            y = Y[i, col]
            for j in range(0, i):
                y -= L[i, j] * Y[j, col]
            Y[i, col] = y

def block_lu(N, block_size, A):
    """
    Perform LU decomposition on a matrix A using a block-wise approach.
    
    Parameters:
    - N: int, size of the square matrix
    - block_size: int, size of each block
    - num_block: int, number of blocks
    - A: numpy.ndarray, input matrix to be decomposed
    
    Returns:
    None
    """
    for idx in range(0, N, block_size):
        block_kk = A[idx: idx + block_size, idx: idx + block_size]
        lu(block_kk)
        
        for i in range(idx + block_size, N, block_size):
            block_ik = A[i: i + block_size, idx: idx + block_size]
            back_substitution(block_kk, block_ik)
            
        for j in range(idx + block_size, N, block_size):
            block_kj = A[idx: idx + block_size, j: j + block_size]
            forward_substitution(block_kk, block_kj)

        for i in range(idx + block_size, N, block_size):
            for j in range(idx + block_size, N, block_size):
                block_ij = A[i: i + block_size, j: j + block_size]
                block_ik = A[i: i + block_size, idx: idx + block_size]
                block_kj = A[idx: idx + block_size, j: j + block_size]
                block_ij -= block_ik @ block_kj

def write_to_file(serial_or_parallel, N, block_size, runtime):
    with open("runtimes.txt", "a") as file:
        file.write(f"{serial_or_parallel}, {N}, {block_size}, {runtime}\n")

if __name__ == "__main__":
  
    if len(sys.argv) != 3:
        print("Usage: python script.py N block_size")
        sys.exit(1)

    # Get N and block_size from command-line arguments
    N = int(sys.argv[1])
    block_size = int(sys.argv[2])

    # A = np.array([[1, 2, 3],[3, 1, 4], [5, 3, 1]], dtype=float)

    A = np.zeros((N, N))
    random.seed()
    for i in range(N):
        for j in range(N):
            A[i][j] = random.randint(1, 100)

    # print("Original matrix A:")
    # print(A)

    start_time = time.time()
    block_lu(N, block_size, A)
    end_time = time.time()

    # print("\nLU-decomposed matrix A:")
    # print(A)

    runtime = end_time - start_time
    print("Execution Time:", runtime, "seconds")
    write_to_file(0, N, block_size, runtime)
    

