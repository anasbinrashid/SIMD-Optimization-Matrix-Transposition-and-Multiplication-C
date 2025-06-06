#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define N 1024

void scalar_2Dimplementation(float **A, float **B, float **C) 
{
    float *A_T = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            A_T[j * N + i] = A[i][j];
        }
    }

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            C[i][j] = A_T[i * N + j] * B[i][j];
        }
    }

    free(A_T);
}

void scalar_1Dimplementation(float *A, float *B, float *C) 
{
    float *A_T = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            A_T[j * N + i] = A[i * N + j];
        }
    }

    for (int i = 0; i < N * N; i++) 
    {
        C[i] = A_T[i] * B[i];
    }

    free(A_T);
}

void simd_implementation(float **A, float **B, float **C) 
{
    float *A_T = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i += 8) 
    { 
        for (int j = 0; j < N; j++) 
        {
            __m256 row = _mm256_loadu_ps(&A[j][i]); 
            
            for (int k = 0; k < 8; k++) 
            {
                A_T[(i + k) * N + j] = ((float*)&row)[k];  
            }
        }
    }

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j += 8) 
        {  
            __m256 a = _mm256_loadu_ps(&A_T[i * N + j]);
            __m256 b = _mm256_loadu_ps(&B[i][j]);
            __m256 c = _mm256_mul_ps(a, b);
            _mm256_storeu_ps(&C[i][j], c);
        }
    }

    free(A_T);
}


int main() 
{
    srand(time(NULL));

    printf("\n\nN = %d", N);

    float **A = (float **)malloc(N * sizeof(float *));
    float **B = (float **)malloc(N * sizeof(float *));
    float **C = (float **)malloc(N * sizeof(float *));
    
    for (int i = 0; i < N; i++)
    {
        A[i] = (float *)malloc(N * sizeof(float));
        B[i] = (float *)malloc(N * sizeof(float));
        C[i] = (float *)malloc(N * sizeof(float));
    }

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            A[i][j] = (float)rand() / RAND_MAX;
            B[i][j] = (float)rand() / RAND_MAX;
        }
    }

    clock_t start = clock();
    scalar_2Dimplementation(A, B, C);
    clock_t end = clock();
    
    printf("\n\nScalar 2D time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    float *A_flat = (float *)malloc(N * N * sizeof(float));
    float *B_flat = (float *)malloc(N * N * sizeof(float));
    float *C_flat = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            A_flat[i * N + j] = A[i][j];
            B_flat[i * N + j] = B[i][j];
        }
    }

    start = clock();
    scalar_1Dimplementation(A_flat, B_flat, C_flat);
    end = clock();
    
    printf("\nScalar 1D time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    simd_implementation(A, B, C);
    end = clock();
    
    printf("\nSIMD time: %f seconds\n\n\n", (double)(end - start) / CLOCKS_PER_SEC);

    for (int i = 0; i < N; i++) 
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }

    free(A);
    free(B);
    free(C);
    free(A_flat);
    free(B_flat);
    free(C_flat);

    return 0;
}

