import numpy as np
import sys
from block_lu import *
import scipy as sp
def extraxt_LU_from_compact(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = A[i, j]
            else:
                U[i, j] = A[i, j]
    return L, U


def check_matrix(A, B):
    print(A)
    print(B)
    
    if np.allclose(A, B):
        # if two matrix are almost equal!
        print(f"Equal!")
    else: 
        print(f"NOT Equal!")

if __name__ == "__main__":
  
    if len(sys.argv) != 3:
        print("Usage: python test.py N block_size")
        sys.exit(1)

    # Get N and block_size from command-line arguments
    N = int(sys.argv[1])
    block_size = int(sys.argv[2])

    

    A = np.zeros((N, N))
    random.seed()
    for i in range(N):
        for j in range(N):
            A[i][j] = random.randint(1, 100)

    A = np.array([[1, 2, 3, 4, 5, 6, 7, 8], 
                  [4, 3, 2, 4, 2, 2, 2, 1],
                  [2, 2, 1, 4, 5, 1, 2, 2],
                  [4, 2, 1, 5, 1, 1, 1, 1],
                  [3, 3, 3, 3, 6, 5, 5, 5],
                  [2, 2, 2, 2, 4, 4, 4, 2],
                  [3, 2, 1, 4, 3, 2, 1, 4],
                  [1, 5, 6, 7, 8, 5, 4, 3]], dtype=float)

    # Keep the origin matrix A
    origin_A = A.copy()
    print("Original matrix A:")
    print(origin_A)

    # Calculate
    start_time = time.time()
    ## My solution
    block_lu(N, block_size, A)

    ## build-in func
    # sp.linalg.lu(A, overwrite_a=True)
    end_time = time.time()

    print("\nLU-decomposed matrix A:")
    print(A)
    

    # Evaluate the solution
    # Correctness
    L, U = extraxt_LU_from_compact(A)
    made_A = L @ U
    print("Made A:")
    print(made_A)
    check_matrix(origin_A, made_A)

    # Solution Runtime 
    runtime = end_time - start_time
    print("Execution Time:", runtime, "seconds")

    write_to_file(0, N, block_size, runtime)