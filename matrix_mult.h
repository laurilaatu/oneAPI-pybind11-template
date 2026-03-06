#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

// Declare the SYCL matrix multiplication function
void sycl_matrix_multiply(const float* A, const float* B, float* C, int M, int K, int N);

#endif // MATRIX_MULT_H