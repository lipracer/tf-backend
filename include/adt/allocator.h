
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../device.h"

namespace tfbe
{
class Allocator
{
public:
    virtual void* allocate(size_t size);
    virtual void deallocate(void* ptr);

    // some device has not mfu, we need the device info
    virtual void* allocate(size_t size, DeviceInfo deviceInfo) = 0;
    virtual void deallocate(void* ptr, DeviceInfo deviceInfo) = 0;
};

class CpuAllocator : public Allocator
{
public:
    void* allocate(size_t size, DeviceInfo deviceInfo) override;
    void deallocate(void* ptr, DeviceInfo deviceInfo) override;
};

class GpuAllocator : public Allocator
{
public:
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;

    void* allocate(size_t size, DeviceInfo deviceInfo) override;
    void deallocate(void* ptr, DeviceInfo deviceInfo) override;
};

} // namespace tfbe
