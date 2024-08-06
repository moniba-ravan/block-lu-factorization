#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

void lu(int idx, double* A, int n, int block_size) {
    /*
    Perform LU decomposition on a block of the matrix A,
    where the block is defined by the starting index `idx` and
    the block size `block_size`. The matrix A is overwritten with
    the LU decomposition result.

    Parameters:
        idx : int
            Starting index for the block in the matrix A.
        A : double pointer
            Pointer to the matrix A (size n x n).
        n : int
            Size of the matrix A.
        block_size : int
            Size of the block for LU decomposition.
    */
    for (int k = 0; k < block_size; k++) {
        // Vectorized division step
        #pragma omp simd
        for (int i = k + 1; i < block_size; i++) {
            if (A[(k + idx) * n + (k + idx)] == 0.0 )
                A[(i + idx) * n + (k + idx)] = 0.0;
            else
                A[(i + idx) * n + (k + idx)] /= A[(k + idx) * n + (k + idx)];
        }
        #pragma omp for schedule(dynamic)
        for (int i = k + 1; i < block_size; i++) {
            for (int j = k + 1; j < block_size; j++) {
                A[(i + idx) * n + (j + idx)] -= A[(i + idx) * n + (k + idx)] * A[(k + idx) * n + (j + idx)];
            }
        }
    }
}

void back_substitution(int idx_i, int idx_j, double* A, int n, int block_size) {
    /*
    Perform back substitution to solve XU = Y for X,
    where U is an upper triangular matrix and Y is the
    matrix to be updated. The block of U and Y is defined by
    the indices `idx_i` and `idx_j`, respectively.

    Parameters:
        idx_i : int
            Starting row index of the block in matrix U.
        idx_j : int
            Starting column index of the block in matrix Y.
        A : double pointer
            Pointer to the matrix A, which contains the upper triangular matrix U.
        n : int
            Size of the matrix A.
        block_size : int
            Size of the block for back substitution.
    */
    #pragma omp parallel for
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            double y = A[(i + idx_j) * n + (j + idx_i)];
            for (int k = 0; k < j; k++) {
                y -= A[(i + idx_j) * n + (k + idx_i)] * A[(k + idx_i) * n + (j + idx_i)];
            }
            if (A[(j + idx_i) * n + (j + idx_i)] == 0)
                A[(i + idx_j) * n + (j + idx_i)] = 0;
            else
                A[(i + idx_j) * n + (j + idx_i)] = y / A[(j + idx_i) * n + (j + idx_i)];
        }
    }
}

void forward_substitution(int idx_i, int idx_j, double* A, int n, int block_size) {
    /*
    Perform forward substitution to solve LX = Y for X,
    where L is a lower triangular matrix and Y is the matrix
    to be updated. The block of L and Y is defined by the indices
    `idx_i` and `idx_j`, respectively.

    Parameters:
        idx_i : int
            Starting row index of the block in matrix L.
        idx_j : int
            Starting column index of the block in matrix Y.
        A : double pointer
            Pointer to the matrix A, which contains the lower triangular matrix L.
        n : int
            Size of the matrix A.
        block_size : int
            Size of the block for forward substitution.
    */
    #pragma omp parallel for
    for (int j = 0; j < block_size; j++) {
        for (int i = 0; i < block_size; i++) {
            double y = A[(i + idx_i) * n + (j + idx_j)];
            for (int k = 0; k < i; k++) {
                y -= A[(i + idx_i) * n + (k + idx_i)] * A[(k + idx_i) * n + (j + idx_j)];
            }
            if (A[(i + idx_i) * n + (i + idx_i)] == 0.0)
                A[(i + idx_i) * n + (j + idx_j)] = 0.0;
            else
                A[(i + idx_i) * n + (j + idx_j)] = y;
        }
    }
}

void matrix_multiply_LU(double* A, double* C, int n, int origin_n) {
    /*
    Perform matrix multiplication A * B = C, where both A and
    B are derived from the LU decomposition, and overwrite matrix C.

    Parameters:
        A : double pointer
            Pointer to the matrix A.
        C : double pointer
            Pointer to the result matrix C.
        n : int
            Size of the matrices A and B (they are square matrices).
    */
    #pragma omp parallel for simd
    for (int i = 0; i < origin_n; i++) {
        for (int j = 0; j < origin_n; j++) {
            double sum = 0.0, x = 0.0, y = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; k++) {
                x = y = 0.0;
                if (i == k)
                    x = 1.0;  // Identity matrix
                if (i > k)
                    x = A[i * n + k];
                if (k <= j)
                    y = A[k * n + j];
                sum += x * y;
            }
            C[i * origin_n + j] = sum;
        }
    }
}

void matrix_multiply(int idx_idx, int idx_i, int idx_j, double* A, double* C, int n, int block_size) {
    /*
    Perform matrix multiplication of blocks A and B to compute C,
    and overwrite matrix C.

    Parameters:
        idx_idx : int
            Starting index for the block in matrix A.
        idx_i : int
            Starting row index for the block in matrix A.
        idx_j : int
            Starting column index for the block in matrix B.
        A : double pointer
            Pointer to the matrix A.
        C : double pointer
            Pointer to the result matrix C.
        n : int
            Size of the matrix A.
        block_size : int
            Size of the block for matrix multiplication.
    */
    #pragma omp parallel for simd
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < block_size; k++) {
                sum += A[(i + idx_i) * n + (k + idx_idx)] * A[(k + idx_idx) * n + (j + idx_j)];
            }
            C[i * block_size + j] = sum;
        }
    }
}

void block_lu(int N, int block_size, double* A) {
    /*
    Perform block-wise LU decomposition on matrix A and overwrite A.

    Parameters:
        N : int
            Size of the matrix A.
        block_size : int
            Size of the block for LU decomposition.
        A : double pointer
            Pointer to the matrix A.
    */

    for (int idx = 0; idx < N; idx += block_size) {

        
        lu(idx, A, N, block_size); 
        
        
        #pragma omp parallel
        {
            #pragma omp for
            for (int j = idx + block_size; j < N; j += block_size) {
                // L_ji.U_ii = A_ji
                back_substitution(idx, j, A, N, block_size);
                
            }
            

            
            #pragma omp for
            for (int j = idx + block_size; j < N; j += block_size) {
                // L_ii.U_ij = A_ij
                forward_substitution(idx, j, A, N, block_size);
                ;
            }
            

            #pragma omp barrier

            #pragma omp for collapse(2) schedule(dynamic)
            for (int i = idx + block_size; i < N; i += block_size) {
                for (int j = idx + block_size; j < N; j += block_size) {
                   
                    double* temp = (double*)malloc(block_size * block_size * sizeof(double));
                    matrix_multiply(idx, i, j, A, temp, N, block_size);
                    
                    
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
    

    int approx_equal = 1;  // Assume matrices are approximately equal
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(A[i * N + j] - B[i * N + j]) > tol) {
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
    /*
    ...
    
    Parameters:
       
    */
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
    
    int origin_N = N;
    if (N % block_size){
        // if N is not divisible by block_size 
        // we increase to the closest number which is divible by block_size.
        N = ((int)(N/block_size) + 1) * block_size;
        // printf("%d %d\n", origin_N, N);
    }

    omp_set_num_threads(n_threads); 

    double* A = (double*)malloc(N * N * sizeof(double));
    double* origin_A = (double*)malloc(origin_N * origin_N * sizeof(double));
    
    if (A == NULL || origin_A == NULL ) {
        printf("Memory allocation failed\n");
        return -1;
    }
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) 
        for (int j = 0; j < N; j++){
            if( i >= origin_N || j >= origin_N)
                A[i * N + j] = 0.0;
            else origin_A[i * origin_N + j] = A[i * N + j] = (double)( rand() % 1000) + 1.0;  // Random values between 1 and 1000
        }
        
    
    
    
    double start_time = omp_get_wtime();
    block_lu(N, block_size, A);
    double end_time = omp_get_wtime();
    
    
   
    // evaluate the solution
    // Correctness
    double* made_A = (double*)malloc(origin_N * origin_N * sizeof(double));
    if (made_A == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }
    // For demonstration purposes, recombine L and U to verify correctness

    // made_A <- L@U which L and U are stored in A
    matrix_multiply_LU(A, made_A, N, origin_N);
    
    double tol = 1e-9;
    printf("\n> Check if L @ U is equal to A: ");
    check_matrix(origin_A, made_A, origin_N, tol);

    printf("\n> Execution Time: %f seconds\n", end_time - start_time);
    if( origin_N < 5 ){
        printf("\n> Original Matirx A:\n");
        display(0, 0, origin_A, origin_N, origin_N);
        printf("\n> L & U stored in one Matrix:\n");
        display(0, 0, A, N, origin_N);
    }

    // Store the timing 
    write_to_file(n_threads, origin_N, block_size, end_time - start_time);

    free(A);
    free(origin_A);
    free(made_A);
    return 0;
}

            
