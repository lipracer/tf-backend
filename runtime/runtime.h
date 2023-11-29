#pragma once

#include "device.h"
#include "type/base_types.h"

namespace tfbe
{
namespace runtime
{
class RTErr_t
{
public:
    RTErr_t() = default;
    RTErr_t(int v) : value_(v) {}

    operator int()
    {
        return value_;
    }

    bool operator==(const RTErr_t& other) const
    {
        return value_ == other.value_;
    }

    bool operator!=(const RTErr_t& other) const
    {
        return !(*this == other);
    }

    static RTErr_t success()
    {
        return RTErr_t();
    }

private:
    int value_ = 0;
};

inline bool isSuccess(const RTErr_t& code)
{
    return code == RTErr_t();
}

inline bool isFailure(const RTErr_t& code)
{
    return code != RTErr_t();
}

RTErr_t success();
RTErr_t failure();

RTErr_t device_malloc(size_t size, DeviceInfo dev, void** ptr);
RTErr_t device_free(void* ptr);

RTErr_t device_memcpy(void* dst, void* src, size_t size, DeviceInfo dst_dev, DeviceInfo src_dev);

} // namespace runtime
} // namespace tfbe
