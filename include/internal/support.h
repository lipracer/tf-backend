#pragma once

#include <functional>

namespace tfbe
{

using InitializationFunc = std::function<void(void)>;

void registeInitialization(const InitializationFunc& initialization);
void registeRelease(const InitializationFunc& initialization);

template <typename T, typename... Args>
void registeGlobalVariable(T** t, Args... args)
{
    registeInitialization([=]() { *t = new T(std::forward<Args>(args)...); });
    registeRelease([=]() { delete *t; });
}

template <typename T>
class GlobalVariable
{
public:
    GlobalVariable()
    {
        registeGlobalVariable<T>(&ptr_);
    }
    T* operator->()
    {
        return ptr_;
    }
    T* ptr_{nullptr};
};

} // namespace tfbe
