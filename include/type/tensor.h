#pragma once

#include <iosfwd>
#include <vector>

#include <stdint.h>

#include "../device.h"
#include "../macro.h"
#include "allocator.h"
#include "ref_counter_ptr.h"
#include "tensor_impl.h"

namespace tfbe
{

class BE_EXPORT Tensor
{
public:
    Tensor() = default;
    Tensor(IntrusivePtr<TensorImpl> impl);
    Tensor(DeviceInfo device_info, ArrayRef<DimT> shape, ElementType ele_type = ElementType::Unknown);

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;

    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) = default;

    ~Tensor();
    // total element size
    size_t totalElements() const;
    // ArrayRef has the const qualify, this is proxy type
    ArrayRef<DimT> shape();

    ElementType elementType() const;

    size_t numBytes() const;

    DimT rank() const;

    template <typename T>
    T* data() const
    {
        return reinterpret_cast<T*>(data());
    }

    void* data() const;

    // must return references to avoid redundant copy operations, this wiil cause increase ref count
    IntrusivePtr<TensorImpl>& getImpl();
    const IntrusivePtr<TensorImpl>& getImpl() const;

    Tensor operator+(const Tensor& rhs);
    Tensor operator-();

    Tensor reshape(ArrayRef<DimT> new_shape);
    Tensor broadcast(ArrayRef<DimT> shape);

    Tensor to(DeviceInfo device);

    void setAllocator(Allocator* allocator);

    DeviceInfo getDeviceInfo() const;

    // help function
    bool isFloating() const;
    bool isIntegral() const;

protected:
    IntrusivePtr<TensorImpl> impl_;
};

class FloatTensor : public Tensor
{
    FloatTensor(DeviceInfo device_info, ArrayRef<DimT> shape, size_t bpe = 32, bool is_std = false);
};

// refer: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
bool allClose(const Tensor& lhs, const Tensor& rhs, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);

BE_EXPORT Tensor empty_tensor(DeviceInfo info, ArrayRef<DimT> shape, ElementType ele_type, bool init_zero = false);

BE_EXPORT std::ostream& operator<<(std::ostream&, const Tensor& tensor);

template <typename T>
Tensor const_tensor(std::vector<T>& data);

} // namespace tfbe
