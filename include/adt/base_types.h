#pragma once

#include <string>

#include <stddef.h>
#include <stdint.h>

namespace tfbe
{
using DimT = int64_t;

enum class ElementType
{
    Unknown = 0,
    Bool_t,
    Int8_t,
    UInt8_t,
    Int16_t,
    UInt16_t,
    Int32_t,
    UInt32_t,
    Int64_t,
    UInt64_t,
    Float16_t,
    Float32_t,
    Float64_t,
    BFloat16_t,
};

std::string to_string(ElementType);

} // namespace tfbe
