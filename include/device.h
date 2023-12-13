#pragma once

#include <iosfwd>
#include <string>

#include <stdint.h>

#include "macro.h"

namespace tfbe
{

enum class DeviceType
{
    Invalid = 0,
    CPU,
    GPU,
};

class BE_EXPORT DeviceInfo
{
public:
    using DevId_t = int;

    static DeviceInfo CPUDevice(int64_t device_id);
    static DeviceInfo GPUDevice(int64_t device_id);

    DeviceInfo()
    {
        BE_EXPECT_NQ(type_, DeviceType::Invalid);
    }
    DeviceInfo(DeviceType type, int64_t device_id);

    // remove this implicit constructor
    DeviceInfo(DeviceType type);

    bool operator==(const DeviceInfo& other) const
    {
        return type_ == other.type_ && device_id_ == other.device_id_;
    }

    DevId_t devId() const
    {
        return static_cast<DevId_t>(device_id_);
    }

    bool isCPU();
    bool isGPU();

    std::string to_string() const
    {
        std::string str;
        if (type_ == DeviceType::CPU)
        {
            str += "CPU";
        }
        else if (type_ == DeviceType::GPU)
        {
            str += "GPU";
        }
        else
        {
            BE_EXPECT(false, "unknown device!");
        }
        str += ":" + std::to_string(device_id_);
        return str;
    }

private:
    DeviceType type_{DeviceType::Invalid};
    int64_t device_id_{0};
};

struct BE_EXPORT DeviceGuard
{
    DeviceGuard() = delete;
    DeviceGuard(const DeviceInfo& info);
    ~DeviceGuard();
};

struct BE_EXPORT DeviceStream
{
    void* stream;
};

std::ostream& operator<<(std::ostream& os, const DeviceInfo& info);

class Allocator;

Allocator* getAllocator(DeviceInfo);

} // namespace tfbe