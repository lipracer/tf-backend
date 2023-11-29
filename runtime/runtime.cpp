
#include "runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logger/logger.h"
#include "macro.h"

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

RTErr_t device_malloc(size_t size, DeviceInfo dev, void** ptr)
{
    auto p = malloc(size);
    CHECK(ptr, "ptr is null!");
    *ptr = p;
    return success();
}
RTErr_t device_free(void* ptr)
{
    if (ptr)
    {
        free(ptr);
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
        memcpy(dst, src, size);
    }
    else if (src_dev.isCPU() && dst_dev.isGPU())
    {
        memcpy(dst, src, size);
    }
    else if (src_dev.isGPU() && dst_dev.isGPU())
    {
        memcpy(dst, src, size);
    }
    else
    {
        be_unreachable("has not implement!");
    }
    return 0;
}

} // namespace runtime
} // namespace tfbe
