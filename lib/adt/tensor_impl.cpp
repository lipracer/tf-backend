#include "type/tensor_impl.h"

#include <ostream>

#include <string.h>

#include "adaptor.h"

namespace tfbe
{

size_t getTensorNumBytes(TensorImpl* impl)
{
    return impl->numBytes();
}

TensorImpl::TensorImpl(DeviceInfo device_info, ArrayRef<DimT> shape, ElementType ele_type)
    : shape_(shape), element_type_(ele_type), storage_(device_info)
{
    auto numBytes = TotalElements(shape) * ElementSize(ele_type);
    allocator_ = getAllocator(device_info);
    auto data = allocator_->allocate(numBytes);
    storage_ = TensorStorage(device_info, data, numBytes);
}

TensorImpl::~TensorImpl() {}

size_t TensorImpl::totalElements() const
{
    return ::std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<DimT>());
}

size_t TensorImpl::numBytes() const
{
    return storage_.numBytes();
}

ArrayRef<DimT> TensorImpl::shape() const
{
    return shape_;
}

ElementType TensorImpl::elementType() const
{
    return element_type_;
}

void TensorImpl::setElementType(ElementType type)
{
    element_type_ = type;
}

void* TensorImpl::data() const
{
    return storage_.data();
}

void TensorImpl::setAllocator(Allocator* allocator)
{
    allocator_ = allocator;
}

DeviceInfo TensorImpl::getDeviceInfo() const
{
    return storage_.getDeviceInfo();
}

std::ostream& operator<<(std::ostream& os, const TensorImpl& tensor)
{
    EXPECT_EQ(tensor.elementType(), ElementType::Float32_t);
    os << tensor.getDeviceInfo();
    os << "(opaque:" << tensor.data() << ")";
    os << "[";
    if (tensor.totalElements() >= 1)
    {
        os << *(tensor.data<float>() + 0);
    }
    for (size_t i = 1; i < tensor.totalElements(); ++i)
    {
        os << "," << *(tensor.data<float>() + i);
    }

    os << "]";
    return os;
}

} // namespace tfbe
