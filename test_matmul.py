import numpy as np
import time
import sycl_matmul

def run_test():
    # Define matrix dimensions
    M, K, N = 256, 256, 256
    
    # Initialize random float32 matrices
    print(f"Initializing matrices A({M}x{K}) and B({K}x{N})...")
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    # oneAPI SYCL Matmul
    print("Running SYCL matrix multiplication...")
    start_time = time.time()
    C_sycl = sycl_matmul.matmul(A, B)
    sycl_time = time.time() - start_time
    print(f"SYCL Time: {sycl_time:.4f} seconds")

    # Naive Python Nested Loops
    print("Running naive Python nested loops...")
    C_python = np.zeros((M, N), dtype=np.float32)
    start_time = time.time()
    
    # Triple nested loop for matrix multiplication
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C_python[i, j] += A[i, k] * B[k, j]
                
    python_time = time.time() - start_time
    print(f"Naive Python Time: {python_time:.4f} seconds")

    # Verification
    max_diff = np.max(np.abs(C_sycl - C_python))
    print(f"\nMax absolute difference: {max_diff}")
    
    # Allow a small tolerance for floating point rounding variations
    if np.allclose(C_sycl, C_python, atol=1e-4):
        print("SYCL output matches naive Python output.")
    else:
        print("Mismatch found between SYCL and naive Python outputs.")

if __name__ == "__main__":
    run_test()