#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

void lu(int idx, double* A, int n, int block_size) {
    /*
    Perform serial LU-decomposition on matrix A
    and overwrite A.
    
    Parameters:
        A : double pointer
            Pointer to the matrix A.
        n : int
            Size of the matrix A.
    */
    for (int k = 0; k < block_size; k++) { 
        // vectorized
        #pragma omp simd
        for (int i = k + 1; i < block_size; i++) {
            if (A[(k + idx) * n + (k + idx)] == 0.0 )
                A[(i + idx) * n + (k + idx)] = 0.0;
            else A[(i + idx) * n + (k + idx)] /= A[(k + idx) * n + (k + idx)];
        }
        #pragma omp for schedule(dynamic)
        for (int i = k + 1; i < block_size; i++) {
            for (int j = k + 1; j < block_size; j++) {
                A[(i + idx) * n + (j + idx)] -=  A[(i + idx) * n + (k + idx)] * A[(k + idx) * n + (j + idx)];
            }
        }
    }
}

void back_substitution(int idx_i, int idx_j, double* A, int n, int block_size) {
    /*
    Perform back substitution to solve XU = Y for X,
    where U is an upper triangular matrix,
    and overwrite Y.
    Block U_(idx_i, idx_i) and Y_(idx_j, idx_i)
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
    // inside the block X[i][j] ?
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            double y = A[(i + idx_j) * n + ( j + idx_i)];
            for (int k = 0; k < j; k++) {
                // y -= X * U
                y -= A[(i + idx_j) * n + (k + idx_i)] * A[(k + idx_i) * n + (j + idx_i)]; 
            }
            // X_ij = y / U_jj
            if (A[(j + idx_i)* n + (j + idx_i)] == 0 )
                A[(i + idx_j) * n + ( j + idx_i)] = 0;
            else A[(i + idx_j) * n + ( j + idx_i)] = y / A[(j + idx_i)* n + (j + idx_i)];
        }
    }
}

void forward_substitution(int idx_i, int idx_j, double* A, int n, int block_size) {
    /*
    Perform forward substitution to solve LX = Y for X,
    where L is a lower triangular matrix,
    and overwrite Y.
    Block L_(idx_i, idx_i) and Y_(idx_i, idx_j)
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
    for (int j = 0; j < block_size; j++) {
        for (int i = 0; i < block_size; i++) {
            double y = A[(i + idx_i) * n + (j + idx_j)];
            for (int k = 0; k < i; k++) {
                
                // L_ik * X_kj
                y -= (A[(i + idx_i) * n + (k + idx_i)] * A[(k + idx_i) * n + (j + idx_j)]);
               
            }
            
            if (A[(i + idx_i) * n + (i + idx_i)] == 0.0 )
                A[(i + idx_i) * n + (j + idx_j)] = 0.0;
            else A[(i + idx_i) * n + (j + idx_j)] = y;
             
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

void matrix_multiply222(int idx_idx, int idx_i, int idx_j, double* A, double* C, int n, int block_size) {
    /*
    Perform matrix multiplication A * B = C,
    and overwrite matrix C.
    
    Parameters:
                    
        A : double pointer
            Pointer to the matrix A.
            double* block_ik = &A[i * N + idx];
                    
        B : double pointer
            Pointer to the matrix B.
            double* block_kj = &A[idx * N + j];

        C : double pointer
            Pointer to the result matrix C.
            temp

        n : int
            Number of rows, columns in A and B since they are square.
    */
    #pragma omp parallel for simd
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < block_size; k++) {
                // A@B
                sum += A[(i + idx_i) * n + (k + idx_idx)] * A[(k + idx_idx) * n + (j + idx_j)];
            }
            C[i * block_size + j] = sum;
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

        // block_kk 
        // printf("\nblock_kk matrix A:\n");
        // display(0, 0, A, N, N);
        lu(idx, A, N, block_size); 
        
        // printf("\lu:\n");
        // display(0, 0, A, N, N);
        #pragma omp parallel
        {
            #pragma omp for
            for (int j = idx + block_size; j < N; j += block_size) {
                // L_ji.U_ii = A_ji
                back_substitution(idx, j, A, N, block_size);
                
            }
            // printf("&d\n");
            // display(0, 0, A, N, N);
            #pragma omp for
            for (int j = idx + block_size; j < N; j += block_size) {
                // L_ii.U_ij = A_ij
                forward_substitution(idx, j, A, N, block_size);
                // printf("%d\n", j);
                // display(0, 0, A, N, N);
            }
            // printf("\nforward:\n");
            // display(0, 0, A, N, N);
            #pragma omp barrier

            #pragma omp for collapse(2) schedule(dynamic)
            for (int i = idx + block_size; i < N; i += block_size) {
                for (int j = idx + block_size; j < N; j += block_size) {
                    // 
                    

                    double* temp = (double*)malloc(block_size * block_size * sizeof(double));
                    matrix_multiply222(idx, i, j, A, temp, N, block_size);
                    // matrix_multiply(block_ik, block_kj, temp, block_size);
                    
                    #pragma omp simd collapse(2)
                    for (int ii = 0; ii < block_size; ii++) {
                        for (int jj = 0; jj < block_size; jj++) {
                            // double* block_ij = &A[i * N + j];
                            A[(ii + i)* N + (jj + j)] -= temp[ii * block_size + jj];
                        }
                    }
                    free(temp);
                }
            }

        }
    }
}

void display(int idx_i, int idx_j, double* A, int n, int block_size) {
    /*
    Display the matrix A.
    
    Parameters:
        A : double pointer
            Pointer to the matrix A.
        N : int
            Size of the matrix A.
    */
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            printf("%f ", A[((i + idx_i) * n + (j + idx_j))]);
        }
        printf("\n");
    }
}

void extract_LU_from_compact(double* A, int n, double* L, double* U) {
    /*
    Extract L and U from the compact matrix A.
    
    Parameters:
        A : double pointer
            Pointer to the compact matrix A stored in a 1D array.
        n : int
            Size of the matrix A.
        L : double pointer
            Pointer to the output lower triangular matrix L.
        U : double pointer
            Pointer to the output upper triangular matrix U.
    */
    
    // Initialize L to identity matrix and U to zero matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                L[i * n + j] = 1.0;  // Identity matrix
            } else {
                L[i * n + j] = 0.0;
            }
            U[i * n + j] = 0.0;
        }
    }

    // Fill L and U based on A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) {
                L[i * n + j] = A[i * n + j];
            } else {
                U[i * n + j] = A[i * n + j];
            }
        }
    }
}

int are_matrices_approx_equal(double** A, double** B, int n, double tol) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > tol) {
                return 0;
            }
        }
    }   
    return 1;
}

void check_matrix(double* A, double* B, int N, double tol) {
    /*
    Check if two matrices are approximately equal.

    Parameters:
        A : double pointer
            Pointer to the matrix A.
        B : double pointer
            Pointer to the matrix B.
        N : int
            Size of the matrices.
        tol : double
            Tolerance for comparing floating point numbers.
    */
    // printf("Matrix A:\n");
    // display(0, 0, A, N, N);
    // printf("Matrix B:\n");
    // display(0, 0, B, N, N);

    int approx_equal = 1;  // Assume matrices are approximately equal
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(A[i * N + j] - B[i * N + j]) > tol) {
                printf("%f \n", fabs(A[i * N + j] - B[i * N + j]));
                approx_equal = 0;
                break;
            }
        }
        if (!approx_equal) break;
    }
    
    if (approx_equal) {
        printf("Equal!\n");
    } else {
        printf("NOT Equal!\n");
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
    double* origin_A = (double*)malloc(N * N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

    if (N == 8) {
        // Verifying the correctness of the algorithm
        double temp[] = {1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 4, 2, 2, 2, 1, 2, 2, 1, 4, 5, 1, 2, 2, 4, 2, 1, 5, 1, 1, 1, 1, 3, 3, 3, 3, 6, 5, 5, 5, 2, 2, 2, 2, 4, 4, 4, 2, 3, 2, 1, 4, 3, 2, 1, 4,1, 5, 6, 7, 8, 5, 4, 3};
        // double temp[] = {1, 2, 3, 4, 4, 3, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2};
        
        for (int i = 0; i < N * N; i++) {
            origin_A[i] =  A[i] = temp[i];
            
        }
        
    } else {
        srand(time(NULL));
        for (int i = 0; i < N*N; i++) 
                origin_A[i] = A[i] = rand() % 100 + 1;  // Random values between 1 and 100
        
    }
    if (N <= 12) {
        printf("Original matrix A:\n");
        display(0, 0, origin_A, N, N);
    }

    double start_time = omp_get_wtime();
    block_lu(N, block_size, A);
    double end_time = omp_get_wtime();

    if (N <= 12) {
        printf("\nLU-decomposed matrix A:\n");
        display(0, 0, A, N, N);
    }
   
    // evaluate the solution
    // Correctness
    double* made_A = (double*)malloc(N * N * sizeof(double));
    double* L = (double*)malloc(N * N * sizeof(double));
    double* U = (double*)malloc(N * N * sizeof(double));
    if (made_A == NULL || L == NULL || U == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Extract L and U from A
    extract_LU_from_compact(A, N, L, U);

    // For demonstration purposes, recombine L and U to verify correctness
    matrix_multiply(L, U, made_A, N);
    free(L);
    free(U);
    double tol = 1e-4;
    check_matrix(origin_A, made_A, N, tol);
    printf("\nExecution Time: %f seconds\n", end_time - start_time);
    write_to_file(n_threads, N, block_size, end_time - start_time);
    return 0;
}

            
