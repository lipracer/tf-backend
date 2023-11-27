#include "type/tensor.h"

#include <functional>
#include <numeric>
#include <ostream>

#include <string.h>

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

ArrayRef<DimT> Tensor::shape()
{
    return impl_->shape();
}

DimT Tensor::rank() const
{
    return impl_->rank();
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
    if (bpe == 16)
    {
        impl_->setElementType(ElementType::Float16_t);
    }
    else if (bpe == 32)
    {
        impl_->setElementType(ElementType::Float32_t);
    }
    else if (bpe == 64)
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
    EXPECT_EQ(totalElements(), rhs.totalElements());
    size_t size = std::max(totalElements(), rhs.totalElements());
    auto result = empty_tensor(getDeviceInfo(), shape(), elementType());
    for (size_t i = 0; i < size; ++i)
    {
        *(result.data<float>() + i) = *(data<float>() + i) + *(rhs.data<float>() + i);
    }
    return result;
}

Tensor Tensor::operator-()
{
    EXPECT_EQ(elementType(), ElementType::Float32_t);
    size_t lhsSize = totalElements();
    size_t rhsSize = totalElements();
    EXPECT_EQ(lhsSize, rhsSize);
    auto result = empty_tensor(getDeviceInfo(), shape(), elementType());
    for (size_t i = 0; i < lhsSize; ++i)
    {
        *(result.data<float>() + i) = -*(data<float>() + i);
    }
    return result;
}

namespace
{
void broadcastOneDim(ArrayRef<DimT> shape, DimT dim, size_t eleBytes, void* src_data, void* dst_data,
                     size_t src_size = -1, size_t dst_size = -1)
{
    auto outter_loop_count = std::accumulate(shape.begin(), shape.begin() + dim, 1, std::multiplies<>());
    auto batch_size = std::accumulate(shape.begin() + dim + 1, shape.end(), eleBytes, std::multiplies<>());
    for (DimT i = 0; i < outter_loop_count; ++i)
    {
        auto src_offset = batch_size * i;
        auto dst_batch = src_offset * shape[dim];
        for (DimT j = 0; j < shape[dim]; ++j)
        {
            auto dst_offset = dst_batch + j * batch_size;
            CHECK_LE(src_offset + batch_size, src_size);
            CHECK_LE(dst_offset + batch_size, dst_size);
            memcpy(reinterpret_cast<char*>(dst_data) + dst_offset, reinterpret_cast<char*>(src_data) + src_offset,
                   batch_size);
        }
    }
}
} // namespace

Tensor Tensor::broadcast(ArrayRef<DimT> shape)
{
    auto result = empty_tensor(getDeviceInfo(), shape, elementType());
    ShapeType<DimT> new_shape;
    new_shape.resize(shape.size(), 1);
    DimT j = 0;
    for (DimT i = 0; i < static_cast<DimT>(shape.size()); ++i)
    {
        if (shape[i] == this->shape()[j])
        {
            new_shape[i] = shape[i];
            ++j;
        }
    }
    CHECK_EQ(j, rank());
    // LOG(INFO) << "new shape:" << new_shape;
    for (DimT i = shape.size() - 1; i >= 0; --i)
    {
        if (new_shape[i] == 1)
        {
            broadcastOneDim(shape, i, ElementSize(elementType()), data(), result.data(), numBytes(), result.numBytes());
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{
    os << *tensor.getImpl().get();
    return os;
}

#define CASE_TYPE_TO_STR(ty) \
    case ty: return #ty;

std::string to_string(ElementType eleType)
{
    switch (eleType)
    {
        CASE_TYPE_TO_STR(ElementType::Int8_t);
        CASE_TYPE_TO_STR(ElementType::Uint8_t);
        CASE_TYPE_TO_STR(ElementType::Int16_t);
        CASE_TYPE_TO_STR(ElementType::Uint16_t);
        CASE_TYPE_TO_STR(ElementType::Int32_t);
        CASE_TYPE_TO_STR(ElementType::Uint32_t);
        CASE_TYPE_TO_STR(ElementType::Int64_t);
        CASE_TYPE_TO_STR(ElementType::Uint64_t);
        CASE_TYPE_TO_STR(ElementType::Float16_t);
        CASE_TYPE_TO_STR(ElementType::Float32_t);
        CASE_TYPE_TO_STR(ElementType::Float64_t);
        CASE_TYPE_TO_STR(ElementType::Bfoat16_t);
        default:
        {
        }
    };
    be_unreachable("unknown type!");
    return {};
}

} // namespace tfbe
