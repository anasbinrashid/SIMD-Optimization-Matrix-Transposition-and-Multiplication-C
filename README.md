# SIMD Optimization for Matrix Transposition and Element-wise Multiplication

A comprehensive implementation and performance analysis of matrix operations using SIMD (Single Instruction, Multiple Data) optimizations with AVX intrinsics in C.

## Overview

This project demonstrates the optimization of matrix transposition followed by element-wise multiplication using three different approaches:

1. **Scalar 2D Implementation** - Traditional nested loop approach with 2D arrays
2. **Scalar 1D Implementation** - Optimized scalar version using 1D arrays for better cache locality  
3. **SIMD Implementation** - Vectorized operations using AVX intrinsics for parallel processing

### Problem Statement
Given two square matrices A and B of size N×N:
- Compute the transpose of matrix A → A^T
- Perform element-wise multiplication: C = A^T × B
- Compare performance across different implementation strategies

## Performance Results

| Matrix Size | Scalar 2D (s) | Scalar 1D (s) | SIMD (s) | Speedup |
|-------------|---------------|---------------|----------|---------|
| 256×256     | 0.000517      | 0.000622      | 0.000172 | **3.00x** |
| 512×512     | 0.002585      | 0.002046      | 0.000987 | **2.61x** |
| 1024×1024   | 0.012183      | 0.014570      | 0.003914 | **3.11x** |
| 2048×2048   | 0.057914      | 0.069366      | 0.014464 | **4.00x** |

**Key Findings:**
- SIMD consistently achieves 3-4x performance improvement
- Performance gains increase with larger matrix sizes
- Dynamic memory allocation prevents stack overflow for large matrices

## Technical Implementation

### Memory Management
- **Dynamic Allocation**: Uses `malloc()` to handle large matrices on the heap
- **Prevents Stack Overflow**: Avoids segmentation faults with large matrix sizes
- **Contiguous Memory**: Ensures better cache performance

### SIMD Optimizations
- **AVX Intrinsics**: Processes 8 floating-point values simultaneously using `__m256`
- **Vectorized Operations**: 
  - `_mm256_loadu_ps()` for loading data
  - `_mm256_mul_ps()` for parallel multiplication
  - `_mm256_storeu_ps()` for storing results
- **Loop Unrolling**: Reduces loop overhead and improves throughput

### Cache Optimization
- **1D Array Layout**: Better memory locality compared to 2D arrays
- **Sequential Access Patterns**: Minimizes cache misses during operations

## Repository Structure

```
├── Code.c          # Complete implementation with all three methods
├── Report.pdf      # Detailed analysis and performance results
└── README.md       # This file
```

## Prerequisites

### Hardware Requirements
- CPU with AVX support (Intel Sandy Bridge or newer, AMD Bulldozer or newer)
- Sufficient RAM for large matrix operations

### Software Requirements
- GCC compiler with AVX support
- Linux/Ubuntu environment (tested on Ubuntu VM)
- Standard C libraries

## Compilation and Usage

### Compile the program:
```bash
gcc -o matrix_simd Code.c -mavx -O3
```

### Run the program:
```bash
./matrix_simd
```

### Compilation Flags Explained:
- `-mavx`: Enables AVX instruction set
- `-O3`: Maximum optimization level
- `-o matrix_simd`: Output executable name

## Understanding the Code

### Scalar 2D Implementation
```c
void scalar_2Dimplementation(float **A, float **B, float **C)
```
- Traditional approach using nested loops
- 2D array indexing with `A[i][j]`
- Baseline for performance comparison

### Scalar 1D Implementation  
```c
void scalar_1Dimplementation(float *A, float *B, float *C)
```
- Uses 1D arrays with manual indexing `A[i*N + j]`
- Better cache locality than 2D approach
- Demonstrates importance of memory layout

### SIMD Implementation
```c
void simd_implementation(float **A, float **B, float **C)
```
- Vectorized operations using AVX intrinsics
- Processes 8 elements per instruction
- Achieves significant performance improvements

## Key Learning Points

1. **Memory Layout Matters**: 1D arrays can outperform 2D arrays due to better cache locality
2. **SIMD Parallel Processing**: Processing multiple elements simultaneously provides substantial speedups
3. **Dynamic Memory Allocation**: Essential for handling large datasets without stack limitations
4. **Performance Scaling**: SIMD benefits increase with larger problem sizes

## Potential Improvements

- **Memory Alignment**: Use aligned memory allocation for optimal AVX performance
- **Error Handling**: Add checks for malloc failures and invalid inputs
- **Advanced SIMD**: Implement more sophisticated transposition algorithms
- **Benchmarking**: Add more comprehensive timing and profiling tools

## Educational Value

This project demonstrates:
- Practical application of SIMD programming concepts
- Performance optimization techniques in C
- Memory management best practices
- Comparative analysis methodologies
- Real-world parallel computing applications

## Contributing

Feel free to:
- Submit issues for bugs or improvements
- Propose optimizations or alternative implementations
- Add support for different matrix sizes or data types
- Enhance documentation or add examples
