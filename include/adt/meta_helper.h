#pragma once

namespace tfbe
{
class AnyOpaque
{
public:
    AnyOpaque(void* ptr) : ptr_(ptr) {}

    template <typename T>
    operator T()
    {
        return reinterpret_cast<T>(ptr_);
    }

private:
    void* ptr_;
};
} // namespace tfbe
