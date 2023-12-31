#include "native_ops.h"

#include "adt/tensor.h"
#include "device_ops/device_ops.h"
#include "logger/logger.h"

namespace tfbe
{
namespace autogen
{

Tensor Neg(AnyOpaque opaque, const Tensor& input)
{
    auto result = empty_tensor(input.getDeviceInfo(), input.shape(), input.elementType());
#ifdef USE_CUDA
    auto ret = ns_ops::neg<float>(opaque, input.data<float>(), result.data<float>(), result.totalElements());
    CHECK(ret == 0, "neg error:{}", ret);
#endif
    return result;
}

Tensor AddV2(AnyOpaque opaque, const Tensor& lhs, const Tensor& rhs)
{
    Tensor result;
    if (lhs.totalElements() >= rhs.totalElements())
    {
        result = empty_tensor(lhs.getDeviceInfo(), lhs.shape(), lhs.elementType());
    }
    else
    {
        result = empty_tensor(rhs.getDeviceInfo(), rhs.shape(), rhs.elementType());
    }
#ifdef USE_CUDA
    auto ret = ns_ops::broadcast_add<float>(opaque, lhs.data<float>(), rhs.data<float>(), result.data<float>(),
                                            lhs.shape().vec(), rhs.shape().vec());
    CHECK(ret == 0, "broadcast_add error:{}", ret);
#endif
    return result;
}

Tensor AddV3(AnyOpaque opaque, const Tensor& lhs, const Tensor& rhs, float alpha, const std::vector<int>& shape)
{
    return {};
}

} // namespace autogen
} // namespace tfbe