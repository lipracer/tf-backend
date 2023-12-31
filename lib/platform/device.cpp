#include "device.h"

#include "adt/allocator.h"
#include "logger/logger.h"

namespace tfbe
{

DeviceInfo::DeviceInfo(DeviceType type, int64_t device_id) : type_(type), device_id_(device_id)
{
    BE_EXPECT_NQ(type_, DeviceType::Invalid);
}

DeviceInfo::DeviceInfo(DeviceType type) : DeviceInfo(type, 0)
{
    BE_EXPECT_NQ(type_, DeviceType::Invalid);
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

DeviceGuard::DeviceGuard(const DeviceInfo& info)
{
    be_unreachable("unimplement!");
}

DeviceGuard::~DeviceGuard()
{
    be_unreachable("unimplement!");
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
        return new CpuAllocator();
    }
    else if (info.isGPU())
    {
        return new GpuAllocator();
    }
    else
    {
        BE_EXPECT(false, "unknown device info!");
    }
    return {};
}

} // namespace tfbe
