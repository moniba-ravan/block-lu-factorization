#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 3

double get_time_in_seconds() {
    return (double)clock() / CLOCKS_PER_SEC;
}

void printMatrix(double A[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f\t", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void lu(double A[N][N]) {
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

int main() {
    double A[N][N] = {{2.0, 3.0, 1.0},
                      {6.0, 8.0, 3.0},
                      {2.0, 5.0, 2.0}};

    printf("Original Matrix A:\n");
    printMatrix(A);

    double start_time = get_time_in_seconds();
    // LU Decomposition
    lu(A);
    double end_time = get_time_in_seconds();

    printf("LU Decomposed Matrix:\n");
    printMatrix(A);

    printf("Serial LU factorization time: %f seconds\n", end_time - start_time);

    return 0;
}
