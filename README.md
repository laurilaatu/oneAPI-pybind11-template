# Simple project template for binding Python with SYCL

Input numpy arrays into SYCL C++ function

## Build and run the container
```
docker build -t oneapi-2025 --build-arg user=$USER .
docker run -p 127.0.0.1:8080:8080 -v /home/$USER/:/home/$USER/local -it oneapi-2025
```

### Initialize SYCL:

```
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force
```

### Build and run the project

```
mkdir build
cd build
CXX=icpx cmake ..
make -j4

cp sycl_matmul*.so ..
cd ../

python test_matmul.py

```

# Integrated Intel GPU

```
docker run -it  --device /dev/dri --group-add $(stat -c "%g" /dev/dri/renderD128) --group-add $(stat -c "%g" /dev/dri/card1) -v /home/$USER/:/home/$USER/local  -p 8080:8080 oneapi-2025
```

### Verify that GPU is found

```
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force

sycl-ls

# Example output
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Iris(R) Xe Graphics 12.3.0 [1.6.33578+15]
[opencl:cpu][opencl:0] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i7-1365U OpenCL 3.0 (Build 0) [2026.20.1.0.12_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO  [25.18.33578]

```

### Environment variable to control the used device (Defaults to GPU)

```
# GPU
ONEAPI_DEVICE_SELECTOR=opencl:gpu python test_matmul.py

# CPU
ONEAPI_DEVICE_SELECTOR=opencl:cpu python test_matmul.py
```