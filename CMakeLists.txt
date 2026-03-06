cmake_minimum_required(VERSION 3.15)
project(sycl_matmul_project)

# Ensure Pybind11 is found (install via: pip install pybind11)
find_package(pybind11 REQUIRED)

# Create the Python module extension
pybind11_add_module(sycl_matmul bindings.cpp matrix_mult.cpp)

# Enable SYCL compilation flags for Intel's icpx compiler
target_compile_options(sycl_matmul PRIVATE -fsycl)
target_link_options(sycl_matmul PRIVATE -fsycl)