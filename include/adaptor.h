#pragma once

// TODO: replace log.h
#include <iostream>

#include "type/tensor.h"

struct LogWrapper
{
    ~LogWrapper()
    {
        std::cout << std::endl;
    }
};

template <typename T>
inline const LogWrapper& operator<<(const LogWrapper& w, T&& t)
{
    std::cout << std::forward<T>(t);
    return w;
}

// #define LLOG(level) (LogWrapper())
#define LLOG(level)

namespace tfbe
{

TensorStorage* backend_alloc();
TensorStorage* backend_dealloc();

size_t getTensorByteSize(TensorImpl* impl);

} // namespace tfbe
