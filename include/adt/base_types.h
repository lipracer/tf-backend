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

template <ElementType>
struct MapEnumEleTypeToNative;

#define DEF_ELE_TYPE_MAP_CLASS(EleType, NativeType) \
    template <>                                     \
    struct MapEnumEleTypeToNative<EleType>          \
    {                                               \
        using type = NativeType;                    \
    };

DEF_ELE_TYPE_MAP_CLASS(ElementType::Bool_t, uint8_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::Int8_t, int8_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::UInt8_t, uint8_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::Int16_t, int16_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::UInt16_t, uint16_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::Int32_t, int32_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::UInt32_t, uint32_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::Int64_t, int64_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::UInt64_t, uint64_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::Float16_t, uint16_t)
DEF_ELE_TYPE_MAP_CLASS(ElementType::Float32_t, float)
DEF_ELE_TYPE_MAP_CLASS(ElementType::Float64_t, double)
DEF_ELE_TYPE_MAP_CLASS(ElementType::BFloat16_t, uint16_t)

#undef DEF_ELE_TYPE_MAP_CLASS

} // namespace tfbe
