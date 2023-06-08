#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 128

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

//CUDA multiplication
__global__ void matrixmultiplication(double* A, double* B, double* C, int N) 
{
    //Computing position of elemets
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
    //Computing elements for matrix C
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

int main(int argc, char* argv[])
{
    int N = 2048;
    double *hostA;
    double *hostB;
    double *hostC;
    double *C;
    float time;

    //Allocating memory for hosts
    hostA = (double*) calloc(N*N, sizeof(double));
    hostB = (double*) calloc(N*N, sizeof(double));
    hostC = (double*) calloc(N*N, sizeof(double));

    C = (double*) calloc(N*N, sizeof(double));

    //Filling matrices
    fill_matrix(hostA, N);
    fill_matrix(hostB, N);

    //Printg matrices A and B
    //printf("hostA:\n");
    //print_matrix(hostA, N);
    //printf("hostB:\n");
    //print_matrix(hostB, N);

    //Allocating memory for devices
    double *deviceA;
    cudaMalloc((void **)&deviceA, N*N * sizeof(double));
    double *deviceB;
    cudaMalloc((void **)&deviceB, N*N * sizeof(double));
    double * deviceC;
    cudaMalloc((void **)&deviceC, N*N * sizeof(double));

    //Defining dimensions of the block and the grid
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    int N_new = 0;
    if (N % BLOCK_SIZE == 0)
        N_new = int (N / BLOCK_SIZE);
    else
        N_new = int (N / BLOCK_SIZE) + 1;
    dim3 blocksPerGrid = dim3(N_new,N_new);

    //Defining and initializing variables for computing time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

//Recording the computing time
    cudaEventRecord( start, 0);

    //Copying matrices from host to device
    cudaMemcpy(deviceA, hostA, N*N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N*N * sizeof(double), cudaMemcpyHostToDevice);

    //Calling multiplication function
    matrixmultiplication<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, N);
     
    //Copying matrices from device to host
    cudaMemcpy(hostC, deviceC, N*N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord( stop, 0);
    cudaEventSynchronize(stop);

    //Computing time
    cudaEventElapsedTime( &time, start, stop);
    printf("Time: %.2f \n",time);

    //Sequantial multiplication
    mult_pos(N, hostA, hostB, C);
    
    //Printg result of CUDA multiplication
    //printf("hostC:\n");
    //print_matrix(hostC, N);

    //Printg result of sequential multiplication
    //printf("C:\n");
    //print_matrix(C, N);

//Comparing results of CUDA multiplication and sequential multiplication
    compare_matrices(C, hostC, N);

//Deallocationg the memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    free(hostA);
    free(hostB);
    free(hostC);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return 0;
}