#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void lu(double* A, int n) {
    /*
    Perform serial LU-decomposition on matrix A
    and overwrite A.
    
    Parameters:
        A : double pointer
            Pointer to the matrix A.
        n : int
            Size of the matrix A.
    */
    for (int k = 0; k < n; k++) {
        // vectorized
        #pragma omp simd
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
        }
        #pragma omp for schedule(dynamic)
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -=  A[i * n + k] * A[k * n + j];
            }
        }
    }
}

void back_substitution(double* U, double* Y, int num_vectors, int m) {
    /*
    Perform back substitution to solve XU = Y for X,
    where U is an upper triangular matrix,
    and overwrite Y.
    
    Parameters:
        U : double pointer
            Pointer to the upper triangular matrix U.
        Y : double pointer
            Pointer to the matrix Y.
        num_vectors : int
            Number of vectors in Y.
        m : int
            Size of the matrix Y.
    */
    #pragma omp parallel for
    for (int row = 0; row < num_vectors; row++) {
        Y[row * m + 0] /= U[0 * m + 0];
        for (int j = 1; j < m; j++) {
            double y = Y[row * m + j];
            for (int i = 0; i < j; i++) {
                y -= U[i * m + j] * Y[row * m + i];
            }
            Y[row * m + j] = y / U[j * m + j];
        }
    }
}

void forward_substitution(double* L, double* Y, int n, int num_vectors) {
    /*
    Perform forward substitution to solve LX = Y for X,
    where L is a lower triangular matrix,
    and overwrite Y.
    
    Parameters:
        L : double pointer
            Pointer to the lower triangular matrix L.
        Y : double pointer
            Pointer to the matrix Y.
        n : int
            Size of the matrix L.
        num_vectors : int
            Number of vectors in Y.
    */
    #pragma omp parallel for
    for (int col = 0; col < num_vectors; col++) {
        for (int i = 1; i < n; i++) {
            double y = Y[i * num_vectors + col];
            for (int j = 0; j < i; j++) {
                y -= L[i * n + j] * Y[j * num_vectors + col];
            }
            Y[i * num_vectors + col] = y;
        }
    }
}

void matrix_multiply(double* A, double* B, double* C, int n) {
    /*
    Perform matrix multiplication A * B = C,
    and overwrite matrix C.
    
    Parameters:
        A : double pointer
            Pointer to the matrix A.
        B : double pointer
            Pointer to the matrix B.
        C : double pointer
            Pointer to the result matrix C.
        n : int
            Number of rows, columns in A and B since they are square.
    */
    #pragma omp parallel for simd
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void block_lu(int N, int block_size, double* A) {
    /*
    Perform LU decomposition on a block-wise matrix A
    and overwrite A.

    Parameters:
        N : int
            Size of the matrix A.
        block_size : int
            Size of the block.
        A : double pointer
            Pointer to the matrix A.
    */
    for (int idx = 0; idx < N; idx += block_size) {

        double* block_kk = &A[idx * N + idx];
        lu(block_kk, block_size); 

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = idx + block_size; i < N; i += block_size) {
                double* block_ik = &A[i * N + idx];
                back_substitution(block_kk, block_ik, block_size, block_size);
            }

            #pragma omp for
            for (int j = idx + block_size; j < N; j += block_size) {
                double* block_kj = &A[idx * N + j];
                forward_substitution(block_kk, block_kj, block_size, block_size);
            }

            #pragma omp barrier

            #pragma omp for collapse(2) schedule(dynamic)
            for (int i = idx + block_size; i < N; i += block_size) {
                for (int j = idx + block_size; j < N; j += block_size) {
                    double* block_ij = &A[i * N + j];
                    double* block_ik = &A[i * N + idx];
                    double* block_kj = &A[idx * N + j];

                    double* temp = (double*)malloc(block_size * block_size * sizeof(double));
                    matrix_multiply(block_ik, block_kj, temp, block_size);
                    
                    #pragma omp simd collapse(2)
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
}

void display(double* A, int N) {
    /*
    Display the matrix A.
    
    Parameters:
        A : double pointer
            Pointer to the matrix A.
        N : int
            Size of the matrix A.
    */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i * N + j]);
        }
        printf("\n");
    }
}

void write_to_file(int serial_or_parallel, int N, int block_size, double runtime) {
    FILE *file = fopen("runtimes.txt", "a");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }
    
    fprintf(file, "%d, %d, %d, %.2f\n", serial_or_parallel, N, block_size, runtime);
    
    fclose(file);
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

    if (N == 3) {
        double temp[] = {1.0, 2.0, 3.0, 3.0, 1.0, 4.0, 5.0, 3.0, 1.0};
        for (int i = 0; i < N * N; i++) {
            A[i] = temp[i];
        }
    } else {
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = rand() % 100 + 1;  // Random values between 1 and 100
            }
        }
    }
    if (N == 3) {
        printf("Original matrix A:\n");
        display(A, N);
    }

    double start_time = omp_get_wtime();
    block_lu(N, block_size, A);
    double end_time = omp_get_wtime();

    if (N == 3) {
        printf("\nLU-decomposed matrix A:\n");
        display(A, N);
    }
   

    printf("\nExecution Time: %f seconds\n", end_time - start_time);
    write_to_file(n_threads, N, block_size, end_time - start_time);
    return 0;
}

            
