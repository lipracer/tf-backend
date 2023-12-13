#pragma once

#include "adt/base_types.h"
#include "device.h"

namespace tfbe
{
namespace runtime
{
class RTErr_t
{
    using Value_t = int;
public:
    RTErr_t() = default;
    RTErr_t(int v) : value_(v) {}

    template <typename T>
    RTErr_t(T&& t) : value_(static_cast<Value_t>(t))
    {
    }

    operator Value_t()
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
    Value_t value_ = 0;
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

RTErr_t device_malloc(void** ptr, size_t size, DeviceInfo dev);
RTErr_t device_free(void* ptr);

RTErr_t device_memcpy(void* dst, void* src, size_t size, DeviceInfo dst_dev, DeviceInfo src_dev);

} // namespace runtime
} // namespace tfbe
