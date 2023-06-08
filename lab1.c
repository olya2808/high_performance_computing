#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Filling matrix with random double numbers from 0 to 9
void fill_matrix(double* H, int M, int N)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M * N; ++i)
    {
        H[i] = (rand() % 9);
    }
}

// Printing the matrix
void print_matrix(double* H, int M, int N)
{
    int i, j;
    for (j = 0; j < M; ++j)
    {
        for (i = 0; i < N; ++i)
        {
            if (i != N - 1)
            {
                printf("%f\t", H[i * M + j]);
            }
            else
            {
                printf("%f\n", H[i * M + j]);
            }
        }
    }
    printf("\n");
}

// Sequential blas_dgemm 
void blas_dgemm_pos(int M, int N, int K, double* A, double* B, double* copy, int alpha, int beta)
{
    int i, j;
    double* tmp;
    // The result of alpha * A * B is stored in tmp
    tmp = (double*) calloc (M * K, sizeof(double));

    // alpha * A * B
    for ( j = 0; j < M * K; ++j)
    {
        tmp[j] = 0;
        for ( i = 0; i < N; ++i)
        {
            tmp[j] += (alpha * A[i * M + j % M] * B[(j / M) * N + i]);
        }
    }

    // tmp + beta * C
    for (int j = 0; j < M * K; ++j)
    {
        copy[j] = tmp[j] + beta * copy[j];
    }
    //print_matrix(C, M, K);

    // Deallocating the memory
    free(tmp);
}

// Parallel blas_dgemm
void blas_dgemm(int M, int N, int K, double* A, double* B, double* C, double alpha, int beta)
{
    int i, j, num;
    double *tmp, *max, maximum;
    maximum = 0;
    // The result of alpha * A * B is stored in tmp
    tmp = (double*) calloc (M * K, sizeof(double));
    max = (double*) calloc (num, sizeof(double));
    // alpha * A * B
    #pragma omp parallel for shared (A, B) private(j, i)
    for ( j = 0; j < M * K; ++j)
    {
        num = omp_get_num_threads();
        tmp[j] = 0;
        for ( i = 0; i < N; ++i)
        {
            tmp[j] += (alpha * A[i * M + j % M] * B[(j / M) * N + i]);
        }
    }

    // tmp + beta * C
    #pragma omp parallel for shared (C) private(j)
    for (j = 0; j < M * K; ++j)
    {
        C[j] = tmp[j] + beta * C[j];
        // printf("Elements in thread %d: %f\n",omp_get_thread_num(), C[j]);
        // //Additional task to find max element in each thread and max element among all threads
        // if (max[omp_get_thread_num()] < C[j])
        // {
        //     max[omp_get_thread_num()] = C[j];
        // }
    }
    printf("\n");
    //Printing array of maxes in each thread
    // for (i = 0; i < num; ++i)
    // {
    //     printf("max for thread %d is: %f\n", i + 1, max[i]);
    // }
    // maximum = max[0];
    // for (i = 1; i < num; ++i)
    // {
    //     if (max[i] > maximum)
    //     {
    //         maximum = max[i];
    //     }
    // }
    // printf("Resulting max = %f\n", maximum);
    // Deallocating the memory
    free(tmp);
    free(max);
}

// Creating the identity matrix
void identity_matrix(double* H, int M, int N)
{
    int i;
    #pragma omp parallel for
    for (int i = 0; i < M * N; ++i)
    {
        if (i % M != i / M)
        {
            H[i] = 0;
        }
        else
        {
            H[i] = 1;
        }
    }
}

// Comparing matrices
void compare_matrices(double* C, double* copy, int M, int K)
{
    int i, flag;
    double eps;
    flag = 0;
    eps = 0.001;
    for (i = 0; i < M * K; ++i)
    {
        if (abs(C[i] - copy[i]) >= eps)
        {
            flag = 1;
            break;
        }
    }
    if (flag == 0)
    {
        printf("Resuls of sequantial and parallel multiplications are equal \n");
    }
    else
    {
        printf("Resuls of sequantial and parallel multiplications are different \n");
    }
}

int main()
{
    int M, N, K, beta, i, threads;
    double start, end, alpha;
    double *A, *B, *C, *I, *copy;

    M = 5;
    N = 4;
    K = 3;
    alpha = 0.5;
    beta = 0;

    // Allocating memory for matrices, matricess are stored in column-major order
    A = (double*) calloc (M * N, sizeof(double));
    B = (double*) calloc (N * K, sizeof(double));
    C = (double*) calloc (M * K, sizeof(double));
    copy = (double*) calloc (M * K, sizeof(double));
    // Creating an identity matrix to compare with result
    I = (double*) calloc (M * K, sizeof(double));
    
    // Filling matrices with random numbers
    //fill_matrix(A, M, N);
    //fill_matrix(B, N, K);
    //fill_matrix(C, M, K);

    // Creating a copy of C matrix
    for (i = 0; i < M * K; ++i)
    {
        copy[i] = C[i];
    }

    // Creating identity matrices
    identity_matrix(A, M, N);
    identity_matrix(B, N, K);
    identity_matrix(C, M, K);
    //identity_matrix(I, M, K);

    // Printing matrices
    //print_matrix(A, M, N);
    //print_matrix(B, N, K);
    // print_matrix(C, M, K);

    // blas_dgemm_pos function
    start = omp_get_wtime();
    blas_dgemm_pos(M, N, K, A, B, copy, alpha, beta);
    end = omp_get_wtime();
    //print_matrix(copy, M, K);
    printf("Time of sequantial multiplication: %f \n", end - start);
    //printf("Result: \n");
    //print_matrix(copy, M, K);

    // blas_dgemm function
    start = omp_get_wtime();
    blas_dgemm(M, N, K, A, B, C, alpha, beta);
    end = omp_get_wtime();
    //print_matrix(C, M, K);
    printf("Time of parallel multiplication: %f \n", end - start); 
    printf("Result: \n");
    print_matrix(C, M, K);
    print("C\n%f", C);

    // Comparing results
    compare_matrices(C, copy, M, K);

    //printf("Expected identity matrix:\n");
    //print_matrix(I, M, K);

    // Deallocating the memory
    free(A);
    free(B);
    free(C);
    free(copy);
    //free(I);

    return 0;
}