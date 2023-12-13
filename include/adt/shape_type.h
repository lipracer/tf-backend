#pragma once

#include <iosfwd>
#include <string>
#include <utility>

#include <stddef.h>
#include <string.h>

#include "ArrayRef.h"

namespace tfbe
{
template <typename T, size_t S = 4, bool C = false>
class ShapeTypeBase;

template <typename T, size_t S>
class ShapeTypeBase<T, S, /*maybe we just need to special move and destroy functions*/ true>
{
    struct ShapeTypeStorage
    {
        union
        {
            T array[S];
            T* data;
        } s;
        size_t capacity;
    };

public:
    using pointer = T*;
    using const_pointer = const T*;

    using iterator = T*;
    using const_iterator = const T*;

    using reference = T&;
    using const_reference = const T&;

    using size_type = size_t;

    using SelfType = ShapeTypeBase<T, S, true>;

    ShapeTypeBase()
    {
        memset(&storage_, 0, sizeof(storage_));
        size_ = 0;
    }

    ShapeTypeBase(const std::initializer_list<T>& l) : ShapeTypeBase()
    {
        auto buffer = getSuitableStorage(l.size());
        std::copy(l.begin(), l.end(), buffer);
        size_ = l.size();
    }

    template <typename I>
    ShapeTypeBase(I b, I e) : ShapeTypeBase()
    {
        size_t size = std::distance(b, e);
        pointer buffer = getSuitableStorage(size);
        std::copy(b, e, buffer);
        size_ = size;
    }

    ShapeTypeBase(const SelfType& other)
    {
        *this = SelfType(other.begin(), other.end());
    }
    ShapeTypeBase(SelfType&& other)
    {
        *this = std::move(other);
    }
    ShapeTypeBase& operator=(const SelfType& other)
    {
        if (this != &other)
        {
            *this = SelfType(other.begin(), other.end());
        }
        return *this;
    }
    ShapeTypeBase& operator=(SelfType&& other)
    {
        if (this == &other)
        {
            return *this;
        }
#if __cplusplus >= 201402L
        size_ = std::exchange(other.size_, 0);
#else
        size_ = other.size_;
        other.size_ = 0;
#endif
        if (size_ <= S)
        {
            std::move(other.storage_.s.array, other.storage_.s.array + size_, storage_.s.array);
        }
        else
        {
#if __cplusplus >= 201402L
            storage_.s.data = std::exchange(other.storage_.s.data, nullptr);
#else
            storage_.s.data = other.storage_.s.data;
            other.storage_.s.data = nullptr;
#endif
        }
        return *this;
    }

    ~ShapeTypeBase()
    {
        release_heap_memory();
    }

    void push_back(const T& v)
    {
        write_back_value(v);
    }

    void push_back(T&& v)
    {
        write_back_value(std::move(v));
    }

    template <typename... Args>
    void emplace_back(Args&&... args)
    {
        write_back_value(std::forward<Args>(args)...);
    }

    size_t size() const
    {
        return size_;
    }

    bool empty() const
    {
        return !size();
    }

    iterator begin()
    {
        return getBeginPtr();
    }
    iterator end()
    {
        return getBeginPtr() + size_;
    }

    const_iterator begin() const
    {
        return getBeginPtr();
    }
    const_iterator end() const
    {
        return getBeginPtr() + size_;
    }

    pointer data()
    {
        return getBeginPtr();
    }

    const_pointer data() const
    {
        return getBeginPtr();
    }

    const_reference operator[](size_t size) const
    {
        assert(size < size_ && "over bound!");
        return *(begin() + size);
    }
    reference operator[](size_t size)
    {
        assert(size < size_ && "over bound!");
        return *(begin() + size);
    }

    size_type capacity() const
    {
        return storage_.capacity != 0 ? storage_.capacity : S;
    }

    void resize(size_t size, const_reference d = {})
    {
        if (size >= size_)
        {
            increase_storage(size, d);
        }
        else
        {
            decrease_storage(size);
        }
        size_ = size;
    }

    void clear()
    {
        size_ = 0;
    }

    void reserve(size_t size)
    {
        if (size <= S)
        {
            return;
        }
        auto ptr = allocStorage(capacity() << 1);
        storage_.s.data = ptr;
    }

    void increase_storage(size_t size, const_reference d = {})
    {
        auto ptr = getSuitableStorage(size);
        std::fill(ptr + size_, ptr + size, d);
    }
    void decrease_storage(size_t size)
    {
        if (!isStackStorage())
        {
            if (size <= S)
            {
                std::copy(storage_.s.data, storage_.s.data + size_, storage_.s.array);
            }
        }
    }

    template <typename... Args>
    void write_back_value(Args&&... args)
    {
        T* back_ptr = getSuitableStorage(size_ + 1);
        (void)new (back_ptr + size_) T(std::forward<Args>(args)...);
        ++size_;
    }

private:
    T* allocStorage(size_t size)
    {
        auto ptr = reinterpret_cast<T*>(malloc(size * sizeof(T)));
        storage_.capacity = size;
        return ptr;
    }

    pointer getSuitableStorage(size_t size)
    {
        if (size > S)
        {
            if (size <= capacity())
            {
                return storage_.s.data;
            }
            else
            {
                auto ptr = allocStorage(capacity() << 1);
                if (!isStackStorage())
                {
                    std::copy(storage_.s.data, storage_.s.data + size_, ptr);
                    free(storage_.s.data);
                }
                else
                {
                    std::copy(storage_.s.array, storage_.s.array + size_, ptr);
                }
                storage_.s.data = ptr;
                return ptr;
            }
        }
        else
        {
            return storage_.s.array;
        }
    }

    pointer getBeginPtr()
    {
        if (size_ <= S)
        {
            return storage_.s.array;
        }
        else
        {
            return storage_.s.data;
        }
    }

    const_pointer getBeginPtr() const
    {
        return const_cast<ShapeTypeBase*>(this)->getBeginPtr();
    }

    bool isStackStorage()
    {
        return size_ <= S;
    }

    void release_heap_memory()
    {
        if (size_ > S)
        {
            free(storage_.s.data);
        }
    }

private:
    ShapeTypeStorage storage_;
    size_t size_;
};

template <typename T>
class ShapeType : public ShapeTypeBase<T, 4, std::is_trivial<T>::value>
{
public:
    using ShapeTypeBase<T, 4, std::is_trivial<T>::value>::ShapeTypeBase;
};

template <typename T>
ArrayRef<T> makeArrayRef(const ShapeType<T>& s)
{
    return ArrayRef<T>(s.data(), s.size());
}

} // namespace tfbe
