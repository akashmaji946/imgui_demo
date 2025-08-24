#ifndef COMMONS_H
#define COMMONS_H

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <limits>
#include <utility>
#include <time.h>
#include <random>
#include <chrono>

#ifdef __CUDACC__
#define __CUDA__
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
static inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") at "
                  << file << ":" << line << " '" << func << "'\n";
        
        cudaDeviceReset();
        std::exit(99);
    }
}
#else
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#endif

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"

// #include "../include/Material.hpp"
// #include "../include/Sphere.hpp"
// #include "../include/Triangle.hpp"
// #include "../include/Plane.hpp"
// #include "../include/AABB.hpp"
// #include "../include/BVH.hpp"

using namespace std;

#endif // COMMONS_H