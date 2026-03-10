#include "matrix_mult.h"
#include <sycl/sycl.hpp>
#include <iostream>

void sycl_matrix_multiply(const float* A, const float* B, float* C, int M, int K, int N) {
    // Select the default device (GPU if available, otherwise CPU)
    sycl::queue q(sycl::default_selector_v);

    // Create SYCL buffers. These manage data transfer between host (Python/CPU) and device
    sycl::buffer<float, 1> bufA(A, sycl::range<1>(M * K));
    sycl::buffer<float, 1> bufB(B, sycl::range<1>(K * N));
    sycl::buffer<float, 1> bufC(C, sycl::range<1>(M * N));

    // Submit a command group to the queue
    q.submit([&](sycl::handler& h) {
        // Request access to the buffers
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accB = bufB.get_access<sycl::access::mode::read>(h);
        auto accC = bufC.get_access<sycl::access::mode::write>(h);

        // Execute a 2D parallel for loop
        h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            float sum = 0.0f;
            
            // Dot product for the row of A and column of B
            for (int i = 0; i < K; ++i) {
                sum += accA[row * K + i] * accB[i * N + col];
            }
            accC[row * N + col] = sum;
        });
    });

    // Wait for the queue to finish executing before returning to Python
    q.wait();
}


void print_device() {

    sycl::queue q(sycl::default_selector_v);

    std::cout << "SYCL Selected Device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Is it a GPU? "
              << (q.get_device().is_gpu() ? "Yes" : "No") << "\n";
}