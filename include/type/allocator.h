
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace tfbe
{

class Allocator
{
public:
    virtual void* allocate(size_t size);
    virtual void deallocate(void* ptr);
};

class GpuAllocator : public Allocator
{
public:
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
};

} // namespace tfbe
