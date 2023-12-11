#pragma once

#include <cassert>
#include <vector>

#include <stddef.h>
#include <stdint.h>

namespace tfbe
{
template <typename T>
class ArrayRef
{
public:
    using pointer = const T*;
    using const_pointer = const T*;
    using iteartor = pointer;
    using const_iteartor = pointer;

    ArrayRef() {}

    ArrayRef(const_pointer data, size_t size) : data_(data), size_(size) {}

    ArrayRef(const std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {}

    const T* data() const
    {
        return this->data_;
    }
    size_t size() const
    {
        return this->size_;
    }
    const_iteartor begin() const
    {
        return this->data_;
    }
    const_iteartor end() const
    {
        return this->data_ + this->size_;
    }
    const T& operator[](size_t index)
    {
        return *(data_ + index);
    }
    bool empty() const
    {
        return !this->data_ || !this->size_;
    }

    std::vector<T> vec() const
    {
        return std::vector<T>(begin(), end());
    }

    void pop_front(size_t size)
    {
        assert(size <= size_ && "pop front too mant element!");
        data_ = data_ + size;
    }
    void pop_back(size_t size)
    {
        assert(size <= size_ && "pop back too mant element!");
        size_ -= size;
    }

private:
    const_pointer data_ = nullptr;
    size_t size_ = 0;
};

template <typename T, template <typename N> typename C>
ArrayRef<T> makeArrayRef(const C<T>& c)
{
    return ArrayRef<T>(c.data(), c.size());
}

} // namespace tfbe
