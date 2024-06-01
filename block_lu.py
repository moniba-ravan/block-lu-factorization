import numpy as np

def lu(A):
    """
    Perform decompositon on A
    and rewrite A
    """
    n = A.shape[0]
    for k in range(n):
        for i in range(k + 1, n):
            A[i, k] = A[i, k] / A[ k, k]   
        
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] -= A[ i, k] * A[ k, j]

def back_substitution(U, Y):
    """
    Perform back substitution to solve XU = Y for X,
    where U is an upper triangular matrix.
    """
    num_vectors, m = Y.shape

    # Each column/vector in X and Y
    for row in range(num_vectors):
        Y[row, 0] /= U[0, 0]
    
        # Perform back substitution
        for j in range(1, m):
            y = Y[row, j]
            for i in range(0, j):
                y -= U[i, j] * X[row, i]
            y[row, j] = y / U[j, j]
 
def forward_substitution(L, Y):
    """
    Perform forward substitution to solve LX = Y for X,
    where  is an Lower triangular matrix.
    """
    n, num_vectors = Y.shape
    X = np.zeros_like(Y, dtype=np.float64)

    # Each column/vector in X and Y
    for col in range(num_vectors):

        # Perform back substitution
        for i in range(1, n):
            y = Y[i, col]
            for j in range(0, i):
                y -= L[i, j] * Y[j, col]
            Y[i, col] = y

def block_lu(N, block_size, num_block, A):
    #each block idx
    for idx in range(0, N, block_size):
        block_kk = A[idx: idx + block_size, idx: idx + block_size]
        lu(block_kk)
        for i in range(idx+block_size, N, block_size):
            block_ik = A[i: i + block_size, idx: idx + block_size]
            back_substitution(block_kk, block_ik)

        for j in range(idx+block_size, N, block_size):
            block_kj = A[idx:idx + block_size, j: j + block_size]
            forward_substitution(block_kk, block_kj)
        
        for i in range(idx+block_size, N, block_size):
            for j in range(idx+block_size, N, block_size):
                block_ij = A[i: i + block_size, j: j + block_size]
                block_ik = A[i: i + block_size, idx: idx + block_size]
                block_kj = A[idx:idx + block_size, j: j + block_size]
                block_ij -= block_ik @ block_kj



N = 3
block_size = 3
num_block = N / block_size

A = np.array([[1, 2, 3],[3, 1, 4], [5, 3, 1]], dtype=float)

B = np.array([14, 17, 14], dtype=float)
B = np.column_stack((B, B))

print(A)
block_lu(N, block_size, num_block, A)
print(A)