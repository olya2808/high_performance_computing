#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 2
__device__ double result=0;
//#define pos 0

// Filling matrix with random double numbers from 0 to 9
void fill_matrix(double* H, int N)
{
    int i;
    for (i = 0; i < N * N; ++i)
    {
        H[i] = (rand() % 9);
    }
}

// Printing the matrix
void print_matrix(double* H, int N)
{
    int i, j;
    for (j = 0; j < N; ++j)
    {
        for (i = 0; i < N; ++i)
        {
            if (i != N - 1)
            {
                printf("%f\t", H[i * N + j]);
            }
            else
            {
                printf("%f\n", H[i * N + j]);
            }
        }
    }
    printf("\n");
}

// Sequential multiplication
void mult_pos(int N, double* A, double* B, double* C)
{
     for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {   
            for (int k = 0; k < N; k++) 
            {
                C[j * N + i] += A[k * N + i] * B[j * N + k];
            }
        }
    }
}

// Comparing matrices
void compare_matrices(double* C, double* hostC, int N)
{
    int i, flag;
    double eps;
    flag = 0;
    eps = 0.001;
    for (i = 0; i < N * N; ++i)
    {
        if (abs(C[i] - hostC[i]) >= eps)
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

__global__ void matrixmultiplication(double* A, double* B, double* C, int N) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
// вычисление элемента матрицы C
    if (r < N && c < N) 
    {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) 
        {
            sum += B[r * N + i] * A[i * N + c];
        }
    }
    C[r * N + c] = sum;
} 

__global__ void task(double* A, double* B, double* Res, int N) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
    //double result=0.0;
    int pos;

// вычисление элемента матрицы C
    if (blockIdx.x==blockIdx.y)
    {
            if (r < N && c < N) 
        {
            //printf("x=%d\ny=%d\n", blockIdx.x, blockIdx.y);
        // each thread computes one element of the block sub-matrix
            for (int i = 0; i < N; i++) 
            {
                sum += B[r * N + i] * A[i * N + c];
            }
            //if (r==c)
            {
                result+=sum;
                pos=blockIdx.x + blockIdx.y;
                printf("x=%d\n", blockIdx.x);
                printf("y=%d\n", blockIdx.y);
                printf("sum=%f\n", sum);
                printf("result=%f\n", result);
                printf("pos=%d\n", pos);
                Res[pos] = sum;
                printf("respos=%f\n", Res[0]);
            }
        
        }
    }
} 

int main(int argc, char* argv[])
{
    int N = 4;
    double *hostA;
    double *hostB;
    double *hostC;
    double *hostRes;
    double *C;
    float time;
    int num;
    num=(N*N)/(BLOCK_SIZE*BLOCK_SIZE);
    //memory on the host
    hostA = (double*) calloc(N*N, sizeof(double));
    hostB = (double*) calloc(N*N, sizeof(double));
    hostC = (double*) calloc(N*N, sizeof(double));
    hostRes = (double*) calloc(num, sizeof(double));
    C = (double*) calloc(N*N, sizeof(double));
    //matrix initialization
    fill_matrix(hostA, N);
    fill_matrix(hostB, N);
    // allocate device memory
    double *deviceA;
    cudaMalloc((void **)&deviceA, N*N * sizeof(double));
    double *deviceB;
    cudaMalloc((void **)&deviceB, N*N * sizeof(double));
    double * deviceC;
    cudaMalloc((void **)&deviceC, N*N * sizeof(double));
    double * deviceRes;
    cudaMalloc((void **)&deviceRes, num * sizeof(double));

    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    int N_new = 0;
    if (N % BLOCK_SIZE == 0)
        N_new = int (N / BLOCK_SIZE);
    else
        N_new = int (N / BLOCK_SIZE) + 1;
    dim3 blocksPerGrid = dim3(N_new,N_new);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0);
    cudaMemcpy(deviceA, hostA, N*N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N*N * sizeof(double), cudaMemcpyHostToDevice);
    matrixmultiplication<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, N);
    task<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceRes, N);
    cudaEventRecord( stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy(hostRes, deviceRes, num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostC, deviceC, N*N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventElapsedTime( &time, start, stop);
    printf("Time: %.2f \n",time);
    printf("Trace elements: \n");
    double result=0;
    for (int i=0;i<num;++i)
    {
        result+=hostRes[i];
        printf("%f\t", hostRes[i]);
    }
    printf("\nTrace: %f\n", result);
    //print_matrix(hostRes, N);

    //Sequantial multiplication
    mult_pos(N, hostA, hostB, C);
    
    printf("hostC:\n");
    print_matrix(hostC, N);
    //printf("C:\n");
    //print_matrix(C, N);
//comparison of results
    compare_matrices(C, hostC, N);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFree(deviceRes);
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostRes);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return 0;
}