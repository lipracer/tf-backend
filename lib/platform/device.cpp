#include "device.h"

#include "logger/logger.h"
#include "type/allocator.h"

namespace tfbe
{

DeviceInfo::DeviceInfo(DeviceType type, int64_t device_id) : type_(type), device_id_(device_id)
{
    EXPECT_NQ(type_, DeviceType::Invalid);
}

DeviceInfo::DeviceInfo(DeviceType type) : DeviceInfo(type, 0)
{
    EXPECT_NQ(type_, DeviceType::Invalid);
}

DeviceInfo DeviceInfo::CPUDevice(int64_t device_id)
{
    return DeviceInfo(DeviceType::CPU, device_id);
}
DeviceInfo DeviceInfo::GPUDevice(int64_t device_id)
{
    return DeviceInfo(DeviceType::GPU, device_id);
}

bool DeviceInfo::isCPU()
{
    return type_ == DeviceType::CPU;
}
bool DeviceInfo::isGPU()
{
    return type_ == DeviceType::GPU;
}

std::ostream& operator<<(std::ostream& os, const DeviceInfo& info)
{
    os << "(" << info.to_string() << ")";
    return os;
}

Allocator* getAllocator(DeviceInfo info)
{
    if (info.isCPU())
    {
        return new Allocator();
    }
    else if (info.isGPU())
    {
        return new GpuAllocator();
    }
    else
    {
        EXPECT(false, "unknown device info!");
    }
}

} // namespace tfbe
