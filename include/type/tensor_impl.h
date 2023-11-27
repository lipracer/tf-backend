#pragma once

#include <functional>
#include <numeric>
#include <vector>

#include "../macro.h"
#include "ArrayRef.h"
#include "base_types.h"
#include "ref_counter_ptr.h"
#include "tensor_storage.h"

namespace tfbe
{

template <typename T>
using ShapeType = std::vector<T>;

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
        default: break;
    };
    be_unreachable("unknown type!");
    return 0;
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

    DimT rank() const;

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
    ShapeType<DimT> shape_;
    ElementType element_type_{ElementType::Unknown};
    TensorStorage storage_;
    Allocator* allocator_;
};

std::ostream& operator<<(std::ostream& os, const TensorImpl& tensor);
std::ostream& operator<<(std::ostream& os, ArrayRef<DimT> shape);

} // namespace tfbe
