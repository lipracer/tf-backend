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
    Int8_t,
    Uint8_t,
    Int16_t,
    Uint16_t,
    Int32_t,
    Uint32_t,
    Int64_t,
    Uint64_t,
    Float16_t,
    Float32_t,
    Float64_t,
    Bfoat16_t,
};

std::string to_string(ElementType);

} // namespace tfbe
