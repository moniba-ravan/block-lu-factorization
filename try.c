#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

#define N 3
// void solveEquation(int n, float A[][n], float B[][n], float X[][n]) {
//  for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             X[i][j] = B[i][j];
//             for (int k = 0; k < i; k++) {
//                 X[i][j] -= A[i][k] * X[k][j];
//             }
//             X[i][j] /= A[i][i];
//         }
//     }
// }
// External function declaration
extern void dtrsm_(char const *, char const *, char const *, char const *,
    int const *, int const *, double const *, double const *, int const *,
    double *, int const *);
// Wrapper function to simplify calling dtrsm_
void solveEquation(int n, double A[][n], double C[][n]) {
    char SIDE = 'L'; // Multiply from the left
    char UPLO = 'L'; // Lower triangular matrix
    char TRANS = 'N'; // No transpose
    char DIAG = 'N'; // Non-unit triangular
    int M = n; // Number of rows of matrix B
    int W = n; // Number of columns of matrix B
    double ALPHA = 1.0; // Scalar multiplier for A
    int LDA = n; // Leading dimension of A
    int LDB = n; // Leading dimension of B

    // Call the external function
    dtrsm_(&SIDE, &UPLO, &TRANS, &DIAG, &M, &W, &ALPHA, &A[0][0], &LDA, &C[0][0], &LDB);
}
void printMatrix(int n, double mat[][n]) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%.2f\t", mat[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // int n = N; // Change the value of n as needed
    double A[N][N] = {{1, 0, 0},
                     {2, 3, 0},
                     {4, 5, 6}};
    double C[N][N] = {{1, 2, 3},
                     {4, 5, 6},
                     {7, 8, 9}};
    // double B[N][N];

    solveEquation(N, A, C);

    printf("Matrix B:\n");
    printMatrix(N, C);

    return 0;
}