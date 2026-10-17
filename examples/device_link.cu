// Device-link anchor for the host-only example executables.
//
// bsmp is built as relocatable device code (CUDA_SEPARABLE_COMPILATION), so
// every executable that consumes bsmp.lib must be linked through nvcc's
// device-link step, which defines the __cudaRegisterLinkedBinary_* symbols
// referenced by bsmp's objects. CMake performs device linking only for
// targets that contain a CUDA source file, hence this tiny translation unit
// in every example target.
__device__ int bsmp_device_link_anchor() {
    return 0;
}