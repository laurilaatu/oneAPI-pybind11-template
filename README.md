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