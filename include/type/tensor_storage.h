#pragma once
#include "../device.h"
#include "../macro.h"
#include "allocator.h"

namespace tfbe
{
class BE_EXPORT TensorStorage
{
public:
    TensorStorage() = default;
    explicit TensorStorage(DeviceInfo info) : device_info_(info) {}
    TensorStorage(DeviceInfo info, void* ptr, size_t size);
    explicit TensorStorage(void*);

    void* data() const;
    size_t numBytes() const;

    DeviceInfo getDeviceInfo() const;

private:
    void* storage_{nullptr};
    size_t size_{0};
    DeviceInfo device_info_;
};
} // namespace tfbe
