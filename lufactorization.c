#include <stdio.h>
#include <omp.h>

#define N 3

int main() {
    double A[N][N] = {{2.0, 3.0, 1.0},
                      {6.0, 8.0, 3.0},
                      {2.0, 5.0, 2.0}};

    printf("Original Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f\t", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Parallel region for LU decomposition
    #pragma omp parallel
    {
        // LU Decomposition
        for (int k = 0; k < N - 1; k++) {
            // Parallel loop for forward elimination
            #pragma omp for
            for (int i = k + 1; i < N; i++) {
                A[i][k] = A[i][k] / A[k][k];
            }

            // Parallel loop for the remaining computations
            #pragma omp for
            for (int i = k + 1; i < N; i++) {
                for (int j = k + 1; j < N; j++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
            }
        }
    } // End of parallel region

    printf("LU Decomposed Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f\t", A[i][j]);
        }
        printf("\n");
    }

    return 0;
}
