#include "type/tensor.h"

#include <functional>
#include <numeric>
#include <ostream>

#include <string.h>

#include "adaptor.h"
#include "internal/support.h"
#include "logger/logger.h"
#include "type/tensor_impl.h"

namespace tfbe
{

TensorStorage::TensorStorage(DeviceInfo info, void* ptr, size_t size) : storage_(ptr), size_(size), device_info_(info)
{
}

size_t TensorStorage::numBytes() const
{
    return size_;
}

TensorStorage::TensorStorage(void* data) : storage_(data) {}

void* TensorStorage::data() const
{
    return storage_;
}

DeviceInfo TensorStorage::getDeviceInfo() const
{
    return device_info_;
}

Tensor::Tensor(IntrusivePtr<TensorImpl> impl) : impl_(impl) {}

Tensor::Tensor(DeviceInfo device_info, ArrayRef<DimT> shape, ElementType ele_type)
    : impl_(IntrusivePtrCtorTag_t(), device_info, shape, ele_type)
{
}

Tensor::~Tensor() {}

size_t Tensor::totalElements() const
{
    return impl_->totalElements();
}

size_t Tensor::numBytes() const
{
    return impl_->numBytes();
}

ArrayRef<DimT> Tensor::shape() const
{
    return impl_->shape();
}

ElementType Tensor::elementType() const
{
    return impl_->elementType();
}

void* Tensor::data() const
{
    return impl_->data();
}

IntrusivePtr<TensorImpl>& Tensor::getImpl()
{
    return impl_;
}

const IntrusivePtr<TensorImpl>& Tensor::getImpl() const
{
    return impl_;
}

Tensor Tensor::to(DeviceInfo device)
{
    if (getDeviceInfo() == device)
    {
        return Tensor(getImpl());
    }
    auto new_tensor = empty_tensor(device, shape(), elementType());
    // TODO memcpy to device
    memcpy(new_tensor.data(), data(), numBytes());
    return new_tensor;
}

void Tensor::setAllocator(Allocator* allocator)
{
    impl_->setAllocator(allocator);
}

DeviceInfo Tensor::getDeviceInfo() const
{
    return impl_->getDeviceInfo();
}

FloatTensor::FloatTensor(DeviceInfo device_info, ArrayRef<DimT> shape, size_t bpe, bool is_std)
    : Tensor(device_info, shape)
{
    EXPECT_EQ(is_std, true);
    if (bpe = 16)
    {
        impl_->setElementType(ElementType::Float16_t);
    }
    else if (bpe = 32)
    {
        impl_->setElementType(ElementType::Float32_t);
    }
    else if (bpe = 64)
    {
        impl_->setElementType(ElementType::Float64_t);
    }
    be_unreachable("unknown element type!");
}

class GC_Tensor
{
public:
    void record(TensorImpl* impl)
    {
        impls_.push_back(impl);
    }
    ~GC_Tensor()
    {
        for (auto impl : impls_)
        {
            auto ref_count = impl->fetch_count();
            if (ref_count != 0)
            {
                LOG(WARN) << "leaked tensor:"
                          << ""
                          << " with ref:" << ref_count;
            }
        }
    }
    std::vector<TensorImpl*> impls_;
};

static GlobalVariable<GC_Tensor> gGC_Tensor;

Tensor empty_tensor(DeviceInfo info, ArrayRef<DimT> shape, ElementType ele_type, bool init_zero)
{
    auto tensor = Tensor(info, shape, ele_type);
    // LOG(INFO) << "empty_tensor:" << tensor;
    return tensor;
}

Tensor Tensor::operator+(const Tensor& rhs)
{
    EXPECT_EQ(elementType(), ElementType::Float32_t);
    size_t size = std::max(totalElements(), rhs.totalElements());
    auto result = empty_tensor(DeviceType::GPU, shape(), elementType());
    for (size_t i = 0; i < size; ++i)
    {
        *(result.data<float>() + i) = *(data<float>() + i) + *(rhs.data<float>() + i);
    }
    return result;
}

Tensor Tensor::operator-()
{
    EXPECT_EQ(elementType(), ElementType::Float32_t);
    size_t size = totalElements();
    auto result = empty_tensor(DeviceType::GPU, shape(), elementType());
    for (size_t i = 0; i < size; ++i)
    {
        *(result.data<float>() + i) = -*(data<float>() + i);
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{
    os << *tensor.getImpl().get();
    return os;
}

} // namespace tfbe
