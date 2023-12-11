#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "../macro.h"
#include "ArrayRef.h"
#include "base_types.h"
#include "ref_counter_ptr.h"
#include "tensor_storage.h"
#include "shape_type.h"

namespace tfbe
{
// template <typename T>
// using ShapeType = std::vector<T>;

size_t ElementSize(ElementType ele_type);

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

    size_t numBytes() const;

    DimT rank() const;

    void* data() const;

    template <typename T>
    T* data() const
    {
        return reinterpret_cast<T*>(data());
    }

    DeviceInfo getDeviceInfo() const;

    std::shared_ptr<TensorStorage>& getStorage();
    const std::shared_ptr<TensorStorage>& getStorage() const;

    void setShape(ArrayRef<DimT> shape);
    void setElementType(ElementType elementType);
    void setStorage(const std::shared_ptr<TensorStorage>& storage);
    void setAllocator(Allocator* allocator);

    TensorImpl* reahspe(ArrayRef<DimT> shape);

protected:
    // TODO use TnesorImpl avoid the detail of implement
    ShapeType<DimT> shape_;
    ElementType element_type_{ElementType::Unknown};
    std::shared_ptr<TensorStorage> storage_;
    Allocator* allocator_;

    friend class Tensor;
};

std::ostream& operator<<(std::ostream& os, const TensorImpl& tensor);
std::ostream& operator<<(std::ostream& os, ArrayRef<DimT> shape);

} // namespace tfbe
