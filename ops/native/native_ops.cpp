#include "adt/tensor.h"
#include "logger/logger.h"

namespace tfbe
{
namespace autogen
{
Tensor AddV2(const Tensor& lhs, const Tensor& rhs)
{
    auto cpu_lhs = lhs.to(DeviceType::CPU);
    auto cpu_rhs = rhs.to(DeviceType::CPU);

    if (cpu_lhs.totalElements() > cpu_rhs.totalElements())
    {
        cpu_rhs = cpu_rhs.broadcast(cpu_lhs.shape());
    }
    else if (cpu_lhs.totalElements() < cpu_rhs.totalElements())
    {
        cpu_lhs = cpu_lhs.broadcast(cpu_rhs.shape());
    }
    Tensor result = cpu_lhs + cpu_rhs;

    LOG(INFO) << "lhs:" << cpu_lhs;
    LOG(INFO) << "rhs:" << cpu_rhs;
    LOG(INFO) << "result:" << result;

    result = result.to(DeviceType::GPU);
    return result;
}

Tensor AddV3(const Tensor& lhs, const Tensor& rhs, float alpha, const std::vector<int>& shape)
{
    return {};
}

} // namespace autogen
} // namespace tfbe