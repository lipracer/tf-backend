#include "adt/tensor_impl.h"

#include <ostream>

#include <string.h>

#include "logger/logger.h"

namespace tfbe
{
size_t ElementSize(ElementType ele_type)
{
    switch (ele_type)
    {
        case ElementType::Int8_t: return 1;
        case ElementType::Uint8_t: return 1;
        case ElementType::Int16_t: return 2;
        case ElementType::Uint16_t: return 2;
        case ElementType::Int32_t: return 4;
        case ElementType::Uint32_t: return 4;
        case ElementType::Int64_t: return 8;
        case ElementType::Uint64_t: return 8;
        case ElementType::Float16_t: return 2;
        case ElementType::Float32_t: return 4;
        case ElementType::Float64_t: return 8;
        case ElementType::Bfoat16_t: return 2;
        default: break;
    };
    be_unreachable("unknown type!");
    return 0;
}

size_t getTensorNumBytes(TensorImpl* impl)
{
    return impl->numBytes();
}

TensorImpl::TensorImpl(DeviceInfo device_info, ArrayRef<DimT> shape, ElementType ele_type)
    : shape_(shape.vec()), element_type_(ele_type), storage_(std::make_shared<TensorStorage>(device_info))
{
    auto numBytes = TotalElements(shape) * ElementSize(ele_type);
    allocator_ = getAllocator(device_info);
    auto data = allocator_->allocate(numBytes, device_info);
    // storage_ = std::make_shared<TensorStorage>(device_info, data, numBytes);
    storage_.reset(new TensorStorage(device_info, data, numBytes),
                   [=](TensorStorage* storage) { allocator_->deallocate(storage->data(), device_info); });
}

TensorImpl::~TensorImpl() {}

size_t TensorImpl::totalElements() const
{
    return ::std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<DimT>());
}

size_t TensorImpl::numBytes() const
{
    return storage_->numBytes();
}

ArrayRef<DimT> TensorImpl::shape() const
{
    return shape_;
}

DimT TensorImpl::rank() const
{
    return shape_.size();
}

ElementType TensorImpl::elementType() const
{
    return element_type_;
}

void* TensorImpl::data() const
{
    EXPECT(storage_->data(), "");
    return storage_->data();
}

std::shared_ptr<TensorStorage>& TensorImpl::getStorage()
{
    return storage_;
}

const std::shared_ptr<TensorStorage>& TensorImpl::getStorage() const
{
    return storage_;
}

void TensorImpl::setShape(ArrayRef<DimT> shape)
{
    shape_ = shape.vec();
}

void TensorImpl::setElementType(ElementType elementType)
{
    element_type_ = elementType;
}

void TensorImpl::setStorage(const std::shared_ptr<TensorStorage>& storage)
{
    storage_ = storage;
}

void TensorImpl::setAllocator(Allocator* allocator)
{
    allocator_ = allocator;
}

DeviceInfo TensorImpl::getDeviceInfo() const
{
    return storage_->getDeviceInfo();
}

TensorImpl* TensorImpl::reahspe(ArrayRef<DimT> new_shape)
{
    auto result = new TensorImpl();
    result->setShape(new_shape);
    result->setElementType(elementType());
    result->setStorage(storage_);
    result->setAllocator(allocator_);
    return result;
}

std::ostream& operator<<(std::ostream& os, const TensorImpl& tensor)
{
    EXPECT_EQ(tensor.elementType(), ElementType::Float32_t);
    os << tensor.getDeviceInfo();
    os << "(opaque:" << tensor.data() << ")";
    os << "(" << to_string(tensor.elementType()) << ")";
    os << tensor.shape();
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

std::ostream& operator<<(std::ostream& os, ArrayRef<DimT> shape)
{
    std::ostringstream oss;
    oss << "[";
    for (auto dim : shape)
    {
        oss << dim << ",";
    }
    auto str = oss.str();
    str.back() = ']';
    os << str;
    return os;
}

} // namespace tfbe
