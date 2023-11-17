#pragma once

#include <functional>
#include <numeric>
#include <vector>

#include "../macro.h"
#include "ref_counter_ptr.h"
#include "tensor_storage.h"

namespace tfbe
{

enum class ElementType
{
    Unknown = 0,
    Int8_t,
    Uint8_t,
    Int16_t,
    Uint16_t,
    Int32_t,
    Uint32_t,
    Int64_t,
    Uint64_t,
    Float16_t,
    Float32_t,
    Float64_t,
    Bfoat16_t,
};

using DimT = int64_t;

template <typename T>
using ShapeType = std::vector<T>;

template <typename T>
using ArrayRef = const std::vector<T>&;

inline size_t ElementSize(ElementType ele_type)
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
    };
    be_unreachable("unknown type!");
}

inline size_t TotalElements(ArrayRef<DimT> shape)
{
    return ::std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<DimT>());
}

class TensorImpl : public RefCounterImpl<TensorImpl>
{
public:
    TensorImpl() = default;
    TensorImpl(DeviceInfo device_info, ArrayRef<DimT> shape, ElementType ele_type = ElementType::Unknown);

    ~TensorImpl();

    size_t totalElements() const;
    ArrayRef<DimT> shape() const;

    ElementType elementType() const;
    void setElementType(ElementType type);

    size_t numBytes() const;

    void* data() const;

    template <typename T>
    T* data() const
    {
        return reinterpret_cast<T*>(data());
    }

    void setAllocator(Allocator* allocator);

    DeviceInfo getDeviceInfo() const;

protected:
    // TODO use TnesorImpl avoid the detail of implement
    TensorStorage storage_;
    Allocator* allocator_;
    ShapeType<DimT> shape_;
    ElementType element_type_{ElementType::Unknown};
};

std::ostream& operator<<(std::ostream& os, const TensorImpl& tensor);

} // namespace tfbe
