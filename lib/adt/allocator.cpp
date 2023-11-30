
#include "type/allocator.h"

#include <stdlib.h>

#include "../../runtime/runtime.h"
#include "logger/logger.h"

namespace tfbe
{

//============================================================//
// base Allocator
//============================================================//

void* Allocator::allocate(size_t size)
{
    return malloc(size);
}

void Allocator::deallocate(void* ptr)
{
    free(ptr);
}

//============================================================//
// CpuAllocator
//============================================================//


void* CpuAllocator::allocate(size_t size, DeviceInfo deviceInfo)
{
    auto ptr = malloc(size);
    LOG(INFO) << "allocate Cpu tensor size:" << size << " pointer:" << ptr;
    return ptr;
}

void CpuAllocator::deallocate(void* ptr, DeviceInfo deviceInfo)
{
    LOG(INFO) << "deallocate Cpu tensor ptr:" << ptr;
    free(ptr);
}

//============================================================//
// GpuAllocator
//============================================================//

void* GpuAllocator::allocate(size_t size)
{
    be_unreachable("unimplement");
    return nullptr;
}

void GpuAllocator::deallocate(void* ptr)
{
    be_unreachable("unimplement");
}

void* GpuAllocator::allocate(size_t size, DeviceInfo deviceInfo)
{
    void* ptr = nullptr;
    auto ret = runtime::device_malloc(&ptr, size, deviceInfo);
    CHECK(runtime::isSuccess(ret), "error allocate gou memory code:{}", static_cast<int>(ret));
    LOG(INFO) << "allocate Gpu tensor size:" << size << " pointer:" << ptr;
    return ptr;
}

void GpuAllocator::deallocate(void* ptr, DeviceInfo deviceInfo)
{
    LOG(INFO) << "deallocate Gpu tensor ptr:" << ptr;
    runtime::device_free(ptr);
}

} // namespace tfbe