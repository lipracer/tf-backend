#include "adt/tensor.h"

#include <cmath>
#include <functional>
#include <numeric>
#include <ostream>

#include <string.h>

#include "../runtime/runtime.h"
#include "adt/tensor_impl.h"
#include "internal/support.h"
#include "logger/logger.h"

namespace tfbe
{
//============================================================//
// TensorStorage
//============================================================//

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

//============================================================//
// Tensor
//============================================================//

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

Tensor Tensor::to(DeviceInfo device) const
{
    auto new_tensor = empty_tensor(device, shape(), elementType());
    auto result = runtime::device_memcpy(new_tensor.data(), data(), numBytes(), device, getDeviceInfo());
    CHECK(runtime::isSuccess(result), "runtime error code {}!", static_cast<int>(result));
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

//============================================================//
// FloatTensor
//============================================================//

FloatTensor::FloatTensor(DeviceInfo device_info, ArrayRef<DimT> shape, size_t bpe, bool is_std)
    : Tensor(device_info, shape)
{
    BE_EXPECT_EQ(is_std, true);
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
    BE_EXPECT_EQ(elementType(), ElementType::Float32_t);
    BE_EXPECT_EQ(totalElements(), rhs.totalElements());
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
    BE_EXPECT_EQ(elementType(), ElementType::Float32_t);
    size_t lhsSize = totalElements();
    size_t rhsSize = totalElements();
    BE_EXPECT_EQ(lhsSize, rhsSize);
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

Tensor Tensor::reshape(ArrayRef<DimT> new_shape)
{
    return Tensor(IntrusivePtr<TensorImpl>(impl_->reahspe(new_shape)));
}

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
    for (DimT i = shape.size() - 1; i >= 0; --i)
    {
        if (new_shape[i] == 1)
        {
            broadcastOneDim(shape, i, ElementSize(elementType()), data(), result.data(), numBytes(), result.numBytes());
        }
    }
    return result;
}

bool Tensor::isFloating() const
{
    CHECK_EQ(static_cast<int>(ElementType::UInt64_t) + 1, static_cast<int>(ElementType::Float16_t),
             "need update this function!");
    return static_cast<int>(elementType()) > static_cast<int>(ElementType::UInt64_t);
}

bool Tensor::isIntegral() const
{
    return static_cast<int>(elementType()) <= static_cast<int>(ElementType::UInt64_t);
}

bool allCloseIntegral(const Tensor& lhs, const Tensor& rhs, double rtol = 1e-05, double atol = 1e-08,
                      bool equal_nan = false)
{
    return std::equal(reinterpret_cast<char*>(lhs.data()), reinterpret_cast<char*>(lhs.data()) + lhs.numBytes(),
                      reinterpret_cast<char*>(rhs.data()));
}

// absolute(a - b) <= (atol + rtol * absolute(b))
bool allCloseFloating(const Tensor& lhs, const Tensor& rhs, double rtol, double atol, bool equal_nan)
{
    BE_EXPECT_EQ(lhs.elementType(), ElementType::Float32_t);
    for (size_t i = 0; i < lhs.totalElements(); ++i)
    {
        auto lhs_value = *(lhs.data<float>() + i);
        auto rhs_value = *(rhs.data<float>() + i);
        if (std::isnan(lhs_value) && std::isnan(rhs_value) && !equal_nan)
        {
            return false;
        }
        else if (std::isnan(lhs_value) && std::isnan(rhs_value) && equal_nan)
        {
            continue;
        }
        else if (std::isnan(lhs_value) || std::isnan(rhs_value))
        {
            return false;
        }
        if (::abs(lhs_value - rhs_value) > atol + rtol * ::abs(rhs_value))
        {
            return false;
        }
    }
    return true;
}

bool allClose(const Tensor& lhs, const Tensor& rhs, double rtol, double atol, bool equal_nan)
{
    if (lhs.isIntegral())
    {
        return allCloseIntegral(lhs, rhs);
    }
    return allCloseFloating(lhs, rhs, rtol, atol, equal_nan);
}

template <>
Tensor makeCconstantTensor<float>(std::vector<float>& data)
{
    std::vector<DimT> shape = {static_cast<DimT>(data.size()), DimT(1)};
    auto result = empty_tensor(DeviceInfo(DeviceType::CPU), shape, ElementType::Float32_t);
    memcpy(result.data(), data.data(), result.numBytes());
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
        CASE_TYPE_TO_STR(ElementType::Bool_t);
        CASE_TYPE_TO_STR(ElementType::UInt8_t);
        CASE_TYPE_TO_STR(ElementType::Int16_t);
        CASE_TYPE_TO_STR(ElementType::UInt16_t);
        CASE_TYPE_TO_STR(ElementType::Int32_t);
        CASE_TYPE_TO_STR(ElementType::UInt32_t);
        CASE_TYPE_TO_STR(ElementType::Int64_t);
        CASE_TYPE_TO_STR(ElementType::UInt64_t);
        CASE_TYPE_TO_STR(ElementType::Float16_t);
        CASE_TYPE_TO_STR(ElementType::Float32_t);
        CASE_TYPE_TO_STR(ElementType::Float64_t);
        CASE_TYPE_TO_STR(ElementType::BFloat16_t);
        default:
        {
        }
    };
    be_unreachable("unknown type!");
    return {};
}

} // namespace tfbe
