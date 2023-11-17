
#include "type/allocator.h"

#include <stdlib.h>

namespace tfbe
{

void* Allocator::allocate(size_t size)
{
    return malloc(size);
}

void Allocator::deallocate(void* ptr)
{
    free(ptr);
}

void* GpuAllocator::allocate(size_t size)
{
    return malloc(size);
}

void GpuAllocator::deallocate(void* ptr)
{
    free(ptr);
}

} // namespace tfbe