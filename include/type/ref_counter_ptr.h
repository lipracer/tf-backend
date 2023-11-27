#pragma once
#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

namespace tfbe
{

class RefCounter
{
public:
    void increase()
    {
        ++counter_;
    }
    void decrease()
    {
        assert(counter_ > 0 && "decrease zero counter!");
        --counter_;
    }
    size_t fetch_count()
    {
        return counter_;
    }
    std::atomic<size_t> counter_{0};
};

template <typename DerivedT>
class RefCounterImpl
{
public:
    RefCounterImpl() : ref_counter_(new RefCounter()) {}

    ~RefCounterImpl()
    {
        delete ref_counter_;
        ref_counter_ = nullptr;
    }

    void increase_ref()
    {
        assert(ref_counter_ && "increase empty counter!");
        ref_counter_->increase();
    }

    void decrease_ref()
    {
        // Don't check this because !!this always true with release mode
        // if (!this)
        // {
        //     return;
        // }
        // if (!ref_counter_)
        //     return;
        assert(ref_counter_ && "decrease empty counter!");
        ref_counter_->decrease();

        if (fetch_count() == 0)
        {
            auto real_ptr = static_cast<DerivedT*>(this);
            // LOG(INFO) << "delete tensor:" << *real_ptr;
            delete real_ptr;
        }
    }

    size_t fetch_count()
    {
        return ref_counter_->fetch_count();
    }

private:
    RefCounter* ref_counter_{nullptr};
};

struct IntrusivePtrCtorTag_t
{
};

template <typename T>
class IntrusivePtr
{
public:
    template <typename... ArgsT>
    IntrusivePtr(IntrusivePtrCtorTag_t tag, ArgsT&&... args) : ptr_(new T(std::forward<ArgsT>(args)...))
    {
        safe_increase(ptr_);
    }

    IntrusivePtr() = default;

    IntrusivePtr(T* ptr) : ptr_(ptr)
    {
        safe_increase(ptr_);
    }

    IntrusivePtr(const IntrusivePtr& other)
    {
        *this = other;
    }
    IntrusivePtr(IntrusivePtr<T>&& other)
    {
        *this = std::move(other);
    }
    IntrusivePtr& operator=(const IntrusivePtr& other)
    {
        if (this == &other)
        {
            return *this;
        }
        if (ptr_ != other.ptr_)
        {
            safe_decrease(ptr_);
            ptr_ = other.ptr_;
        }
        else
        {
        }
        safe_increase(ptr_);
        return *this;
    }
    IntrusivePtr& operator=(IntrusivePtr&& other)
    {
        if (this == &other)
        {
            return *this;
        }
        if (ptr_ == other.ptr_)
        {
        }
        else
        {
            safe_decrease(ptr_);
            ptr_ = other.ptr_;
        }
        // TODO set ref count directly
        safe_increase(ptr_);
        return *this;
    }

    ~IntrusivePtr()
    {
        safe_decrease(ptr_);
        ptr_ = nullptr;
    }

    T* operator->()
    {
        return ptr_;
    }
    const T* operator->() const
    {
        return ptr_;
    }
    T& operator*()
    {
        return *ptr_;
    }
    const T& operator*() const
    {
        return *ptr_;
    }

    bool operator!() const
    {
        return !ptr_;
    }
    operator bool() const
    {
        return !!ptr_;
    }

    T* get()
    {
        return ptr_;
    }
    const T* get() const
    {
        return ptr_;
    }

    void safe_increase(T* ptr)
    {
        if (ptr)
        {
            ptr->increase_ref();
        }
    }
    void safe_decrease(T* ptr)
    {
        if (ptr)
        {
            ptr->decrease_ref();
        }
    }

private:
    T* ptr_{nullptr};
};

} // namespace tfbe
