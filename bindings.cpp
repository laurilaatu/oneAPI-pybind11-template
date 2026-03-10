#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix_mult.h"
#include <stdexcept>

namespace py = pybind11;

// Wrapper function to handle numpy arrays
py::array_t<float> matmul_wrapper(
    py::array_t<float, py::array::c_style | py::array::forcecast> input1, 
    py::array_t<float, py::array::c_style | py::array::forcecast> input2) {
    
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.ndim != 2 || buf2.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be two");
    }

    if (buf1.shape[1] != buf2.shape[0]) {
        throw std::runtime_error("Matrix dimensions must match for multiplication");
    }

    int M = buf1.shape[0];
    int K = buf1.shape[1];
    int N = buf2.shape[1];

    // Allocate memory for the result array
    auto result = py::array_t<float>({M, N});
    py::buffer_info buf3 = result.request();

    // Extract raw pointers
    float* ptr1 = static_cast<float*>(buf1.ptr);
    float* ptr2 = static_cast<float*>(buf2.ptr);
    float* ptr3 = static_cast<float*>(buf3.ptr);

    // Call the SYCL implementation
    sycl_matrix_multiply(ptr1, ptr2, ptr3, M, K, N);

    return result;
}

// Define the Python module
PYBIND11_MODULE(sycl_matmul, m) {
    m.doc() = "SYCL-accelerated matrix multiplication using PyBind11";
    m.def("matmul", &matmul_wrapper, "Multiply two numpy arrays using oneAPI SYCL");
    m.def("print_device", &print_device, "Print information about the selected SYCL device");
}