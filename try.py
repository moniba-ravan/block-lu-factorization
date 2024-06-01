import numpy as np
from scipy.linalg import lu

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
    return A
def back_substitution(U, Y):
    """
    Perform back substitution to solve UX = Y for X,
    where  is an upper triangular matrix.
    """
    n, num_vectors = Y.shape
    X = np.zeros_like(Y, dtype=np.float64)

    # Each column/vector in X and Y
    for col in range(num_vectors):
        X[n-1, col] = Y[n-1, col] / U[n-1, n-1]
    
        # Perform back substitution
        for i in range(n-2, -1, -1):
            y = Y[i, col]
            for j in range(n-1, i, -1):
                y -= U[i, j] * X[j, col]
            X[i, col] = y / U[i, i]
    
    return X
def forward_substitution(L, Y):
    """
    Perform forward substitution to solve LX = Y for X,
    where  is an Lower triangular matrix.
    """
    n, num_vectors = Y.shape
    X = np.zeros_like(Y, dtype=np.float64)

    # Each column/vector in X and Y
    for col in range(num_vectors):
        X[0, col] = Y[0, col]
    
        # Perform back substitution
        for i in range(1, n):
            y = Y[i, col]
            for j in range(0, i):
                y -= L[i, j] * X[j, col]
            X[i, col] = y
    return X


A = np.array([[7, 2, 5, 0, 1, 9, 0, 3],
                   [7, 7, 3, 0, 1, 0, 0, 6],
                   [5, 3, 8, 4, 1, 5, 9, 7],
                   [9, 7, 6, 3, 3, 0, 8, 7],
                   [4, 5, 4, 7, 3, 7, 1, 2],
                   [2, 1, 2, 7, 0, 4, 6, 6],
                   [3, 1, 0, 1, 9, 0, 1, 4],
                   [8, 8, 3, 1, 3, 6, 0, 1]], dtype=float)
A = np.array([[1, 2, 3],[3, 1, 4], [5, 3, 1]], dtype=float)

C = np.array([14, 17, 14], dtype=float)
C = np.column_stack((C, C))

A = lu(A)
print(A)
X = forward_substitution(A,C)
print(X)
X = back_substitution(A, X)

print(X)
# np.set_printoptions(precision=3, suppress=True)
# print(A)
