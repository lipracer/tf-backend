
#include "runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logger/logger.h"
#include "macro.h"

#ifdef USE_CUDA
    #include <cuda_runtime_api.h>
#else

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,     /**< Host   -> Host */
    cudaMemcpyHostToDevice = 1,   /**< Host   -> Device */
    cudaMemcpyDeviceToHost = 2,   /**< Device -> Host */
    cudaMemcpyDeviceToDevice = 3, /**< Device -> Device */
    cudaMemcpyDefault =
        4 /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

tfbe::runtime::RTErr_t cudaMalloc(void** ptr, size_t size)
{
    auto p = malloc(size);
    CHECK(p, "ptr is null with size:{}!", size);
    *ptr = p;
    return tfbe::runtime::success();
}

tfbe::runtime::RTErr_t cudaFree(void* ptr)
{
    free(ptr);
    return tfbe::runtime::success();
}

template <typename T>
tfbe::runtime::RTErr_t cudaMemcpy(void* dst, void* src, size_t size, T&& t)
{
    memcpy(dst, src, size);
    return tfbe::runtime::success();
}

tfbe::runtime::RTErr_t cudaSetDevice(...)
{
    return tfbe::runtime::success();
}

tfbe::runtime::RTErr_t cudaStreamSynchronize(void*)
{
    return tfbe::runtime::success();
}
#endif

namespace tfbe
{
namespace runtime
{
RTErr_t success()
{
    return RTErr_t::success();
}
RTErr_t failure()
{
    return static_cast<RTErr_t>(-1);
}

RTErr_t device_malloc(void** ptr, size_t size, DeviceInfo dev)
{
    cudaMalloc(ptr, size);
    CHECK(*ptr, "ptr is null with size:{}!", size);
    return success();
}

RTErr_t device_free(void* ptr)
{
    if (ptr)
    {
        cudaFree(ptr);
    }
    return success();
}

RTErr_t device_memcpy(void* dst, void* src, size_t size, DeviceInfo dst_dev, DeviceInfo src_dev)
{
    if (src_dev.isCPU() && dst_dev.isCPU())
    {
        memcpy(dst, src, size);
    }
    else if (src_dev.isGPU() && dst_dev.isCPU())
    {
        CHECK(isSuccess(cudaSetDevice(src_dev.devId())), "cudaSetDevice fail: device id:{}", src_dev.devId());
        CHECK(isSuccess(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)),
              "cudaMemcpyDeviceToHost fail dst:{}, src:{} size:{}", dst, src, size);
        cudaStreamSynchronize(nullptr);
    }
    else if (src_dev.isCPU() && dst_dev.isGPU())
    {
        CHECK(isSuccess(cudaSetDevice(dst_dev.devId())), "cudaSetDevice fail: device id:{}", src_dev.devId());
        CHECK(isSuccess(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)),
              "cudaMemcpyHostToDevice fail dst:{}, src:{} size:{}", dst, src, size);
        cudaStreamSynchronize(nullptr);
    }
    else if (src_dev.isGPU() && dst_dev.isGPU())
    {
        CHECK(isSuccess(cudaSetDevice(src_dev.devId())), "cudaSetDevice fail: device id:{}", src_dev.devId());
        CHECK(isSuccess(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)),
              "cudaMemcpyDeviceToDevice fail dst:{}, src:{} size:{}", dst, src, size);
        cudaStreamSynchronize(nullptr);
    }
    else
    {
        be_unreachable("has not implement!");
    }
    return 0;
}

} // namespace runtime
} // namespace tfbe
