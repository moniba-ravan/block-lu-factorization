#include <stdio.h>
#include <omp.h>

#define N 2
#define BLOCK_SIZE 1

// Function to perform LU factorization with block decomposition
void block_lu_factorization(double A[N][N]) {
    // Iterate over blocks
    for (int block = 0; block < N; block += BLOCK_SIZE) {
            
        // Factorize A_kk to L_kk, U_kk) 
        #pragma omp parallel for
        for (int k = block; k < block + BLOCK_SIZE; k++) {
            for (int i = k + 1; i < block + BLOCK_SIZE; i++) {
                // Factorize A[i][j] into L_kk and U_kk
                A[i][k] /= A[k][k];
                for (int j = k + 1; j < block + BLOCK_SIZE ; j++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
                
            }
        }
        
        // Solve matrix equations to find L_ik and U_ki in parallel
        // Update A_ik and A_ki in parallel
        #pragma omp parallel for collapse(2)
        for (int i = block + BLOCK_SIZE; i < N; i += BLOCK_SIZE) {
            for (int j = block; j < block + BLOCK_SIZE; j++) {
                // Solve matrix equation L_ik * U_kk = A_ik
                for (int p = block; p < block + BLOCK_SIZE; p++) {
                    for (int q = block; q < block + BLOCK_SIZE; q++) {
                        A[i][j] -= A[i][p] * A[p][q] * A[q][j];
                    }
                }
            }
        }
        
        // Compute A’ _ij = A_ij - L_ikU_kj for the remaining blocks
        #pragma omp parallel for collapse(2)
        for (int i = block + BLOCK_SIZE; i < N; i += BLOCK_SIZE) {
            for (int j = block + BLOCK_SIZE; j < N; j += BLOCK_SIZE) {
                // Update the matrix A' with A_ij - L_ikU_kj
                for (int p = block; p < block + BLOCK_SIZE; p++) {
                    for (int q = block; q < block + BLOCK_SIZE; q++) {
                        for (int r = i; r < i + BLOCK_SIZE; r++) {
                            for (int s = j; s < j + BLOCK_SIZE; s++) {
                                A[r][s] -= A[r][p] * A[p][q] * A[q][s];
                            }
                        }
                    }
                }

            }
        }

    }
}


int main() {
    // Initialize matrix A
    // double A[N][N] = {
    //     {7, 2, 5, 0, 1, 9, 0, 3},
    //     {7, 7, 3, 0, 1, 0, 0, 6},
    //     {5, 3, 8, 4, 1, 5, 9, 7},
    //     {9, 7, 6, 3, 3, 0, 8, 7},
    //     {4, 5, 4, 7, 3, 7, 1, 2},
    //     {2, 1, 2, 7, 0, 4, 6, 6},
    //     {3, 1, 0, 1, 9, 0, 1, 4},
    //     {8, 8, 3, 1, 3, 6, 0, 1}
    // };
    double A[N][N] = {{4.0, 4.0},
                      {4.0, 4.0}};
    
    
    // Perform block LU factorization
    block_lu_factorization(A);
    
    // Print the resulting matrix A
    printf("Resulting Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f\t", A[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}
