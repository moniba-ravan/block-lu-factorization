#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void lu(double* A, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
        }
        for (int i = k + 1; i < n; i++) {
            #pragma omp parallel for
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

void back_substitution(double* U, double* Y, int num_vectors, int m) {
    for (int row = 0; row < num_vectors; row++) {
        Y[row * m + 0] /= U[0 * m + 0];
        for (int j = 1; j < m; j++) {
            double y = Y[row * m + j];
            #pragma omp parallel for reduction(-:y)
            for (int i = 0; i < j; i++) {
                y -= U[i * m + j] * Y[row * m + i];
            }
            Y[row * m + j] = y / U[j * m + j];
        }
    }
}

void forward_substitution(double* L, double* Y, int n, int num_vectors) {
    for (int col = 0; col < num_vectors; col++) {
        for (int i = 1; i < n; i++) {
            double y = Y[i * num_vectors + col];
            #pragma omp parallel for reduction(-:y)
            for (int j = 0; j < i; j++) {
                y -= L[i * n + j] * Y[j * num_vectors + col];
            }
            Y[i * num_vectors + col] = y;
        }
    }
}

void matrix_multiply(double* A, double* B, double* C, int n, int m, int p) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < m; k++) {
                C[i * p + j] += A[i * m + k] * B[k * p + j];
            }
        }
    }
}

void block_lu(int N, int block_size, double* A) {
    for (int idx = 0; idx < N; idx += block_size) {
        double* block_kk = &A[idx * N + idx];
        lu(block_kk, block_size);

        for (int i = idx + block_size; i < N; i += block_size) {
            double* block_ik = &A[i * N + idx];
            back_substitution(block_kk, block_ik, block_size, block_size);
        }

        for (int j = idx + block_size; j < N; j += block_size) {
            double* block_kj = &A[idx * N + j];
            forward_substitution(block_kk, block_kj, block_size, block_size);
        }

        for(int i = idx + block_size; i < N; i += block_size) {
            for(int j = idx + block_size; j < N; j += block_size) {
                double* block_ij = &A[i * N + j];
                double* block_ik = &A[i * N + idx];
                double* block_kj = &A[idx * N + j];

                double* temp = (double*)malloc(block_size * block_size * sizeof(double));
                matrix_multiply(block_ik, block_kj, temp, block_size, block_size, block_size);
                #pragma omp parallel for collapse(2)
                for (int ii = 0; ii < block_size; ii++) {
                    for (int jj = 0; jj < block_size; jj++) {
                        block_ij[ii * N + jj] -= temp[ii * block_size + jj];
                    }
                }
                free(temp);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s N block_size n_threads\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]); // Matrix size
    int block_size = atoi(argv[2]); // Block size
    int n_threads = atoi(argv[3]); // Number of threads

    omp_set_num_threads(n_threads); 

    double* A = (double*)malloc(N * N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 100 + 1;  // Random values between 1 and 100
        }
    }
    // double A[9] = {
    //     1, 2, 3,
    //     3, 1, 4,
    //     5, 3, 1
    // };

    printf("Original matrix A:\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", A[i * N + j]);
    //     }
    //     printf("\n");
    // }

    double start_time = omp_get_wtime();
    block_lu(N, block_size, A);
    double end_time = omp_get_wtime();

    printf("\nLU-decomposed matrix A:\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", A[i * N + j]);
    //     }
    //     printf("\n");
    // }
    printf("\nExecution Time: %f seconds\n", end_time - start_time);
    return 0;
}
