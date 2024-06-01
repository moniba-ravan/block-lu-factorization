import numpy as np

A = np.array([[4, 3], [6, 3]], dtype=float)
A = np.array([[2.0, 3.0, 1.0],
              [6.0, 8.0, 3.0],
              [2.0, 5.0, 2.0]])
# P, L, U = np.linalg.lu(A)
print(A)
n = 3
for k in range(n):
    for i in range(k + 1, n):
        print(i, k, A[i, k], A[k, k], A[i, k]/A[k, k]) 
        A[i, k] = A[i, k] / A[ k, k]   
        print(A) 
    
    for i in range(k + 1, n):
        for j in range(k + 1, n):
            print(" > ", i, j, k, A[i, j], A[i, j] - A[ i, k] * A[ k, j]) 
            
            A[i, j] -= A[ i, k] * A[ k, j]
            print(A)
