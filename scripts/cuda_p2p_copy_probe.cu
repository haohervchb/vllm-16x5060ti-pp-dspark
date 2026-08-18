// Validate that CUDA peer access moves real data, rather than only reporting
// the peer-access capability. Build with:
//   nvcc -O2 -arch=sm_120 scripts/cuda_p2p_copy_probe.cu -o /tmp/cuda_p2p_copy_probe
// Run with physical GPUs exposed in the desired order, for example:
//   CUDA_VISIBLE_DEVICES=0,1 /tmp/cuda_p2p_copy_probe 0 1

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d: %s failed: %s (%d)\n", __FILE__,     \
                         __LINE__, #call, cudaGetErrorString(status), status);   \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ void fill_sequence(unsigned int *data, size_t count)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        data[index] = static_cast<unsigned int>(index + 1);
}

__global__ void peer_copy(unsigned int *destination,
                          const unsigned int *source,
                          size_t count)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        destination[index] = source[index];
}

static int validate(const std::vector<unsigned int> &data)
{
    for (size_t index = 0; index < data.size(); ++index) {
        const unsigned int expected = static_cast<unsigned int>(index + 1);
        if (data[index] != expected) {
            std::fprintf(stderr,
                         "validation failed at element %zu: got %u, expected %u\n",
                         index, data[index], expected);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    int result = 0;
    const int source_device = argc > 1 ? std::atoi(argv[1]) : 0;
    const int destination_device = argc > 2 ? std::atoi(argv[2]) : 1;
    const size_t bytes = argc > 3 ? std::strtoull(argv[3], nullptr, 0)
                                  : 16ULL * 1024 * 1024;
    const size_t count = std::max<size_t>(1, bytes / sizeof(unsigned int));
    const size_t allocation_bytes = count * sizeof(unsigned int);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (source_device < 0 || destination_device < 0 ||
        source_device >= device_count || destination_device >= device_count ||
        source_device == destination_device) {
        std::fprintf(stderr, "invalid device pair %d -> %d (device count: %d)\n",
                     source_device, destination_device, device_count);
        return 2;
    }

    int source_can_access_destination = 0;
    int destination_can_access_source = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&source_can_access_destination,
                                       source_device, destination_device));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&destination_can_access_source,
                                       destination_device, source_device));
    std::printf("pair %d -> %d; capability: %d -> %d = %s, %d -> %d = %s\n",
                source_device, destination_device,
                source_device, destination_device,
                source_can_access_destination ? "YES" : "NO",
                destination_device, source_device,
                destination_can_access_source ? "YES" : "NO");
    if (!source_can_access_destination || !destination_can_access_source)
        return 3;

    unsigned int *source = nullptr;
    unsigned int *destination = nullptr;

    CUDA_CHECK(cudaSetDevice(source_device));
    CUDA_CHECK(cudaMalloc(&source, allocation_bytes));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(destination_device, 0));
    fill_sequence<<<(count + 255) / 256, 256>>>(source, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaSetDevice(destination_device));
    CUDA_CHECK(cudaMalloc(&destination, allocation_bytes));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(source_device, 0));
    CUDA_CHECK(cudaMemset(destination, 0, allocation_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Exercise peer writes from a kernel on the source GPU.
    CUDA_CHECK(cudaSetDevice(source_device));
    peer_copy<<<(count + 255) / 256, 256>>>(destination, source, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned int> host(count);
    CUDA_CHECK(cudaSetDevice(destination_device));
    CUDA_CHECK(cudaMemcpy(host.data(), destination, allocation_bytes,
                          cudaMemcpyDeviceToHost));
    if (validate(host) != 0) {
        std::fprintf(stderr, "FAIL: peer kernel wrote invalid data\n");
        result = 4;
    } else {
        std::printf("PASS: peer kernel copy validated (%zu bytes)\n",
                    allocation_bytes);
    }

    // Exercise the CUDA peer-copy API independently.
    CUDA_CHECK(cudaMemset(destination, 0, allocation_bytes));
    CUDA_CHECK(cudaMemcpyPeer(destination, destination_device,
                              source, source_device, allocation_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), destination, allocation_bytes,
                          cudaMemcpyDeviceToHost));
    if (validate(host) != 0) {
        std::fprintf(stderr, "FAIL: cudaMemcpyPeer returned invalid data\n");
        result = 5;
    } else {
        std::printf("PASS: cudaMemcpyPeer validated (%zu bytes)\n",
                    allocation_bytes);
    }

    CUDA_CHECK(cudaFree(destination));
    CUDA_CHECK(cudaSetDevice(source_device));
    CUDA_CHECK(cudaFree(source));
    return result;
}
