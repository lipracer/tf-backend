#ifndef __TF1_X_PLUGIN_ADAPTOR__
#define __TF1_X_PLUGIN_ADAPTOR__

#include "tensorflow/compiler/jit/tf1.x_plugin/include/adt/tensor.h"
#include "tensorflow/core/framework/tensor.h"

namespace adaptor
{
// enum DataType : int {
//   DT_INVALID = 0,
//   DT_FLOAT = 1,
//   DT_DOUBLE = 2,
//   DT_INT32 = 3,
//   DT_UINT8 = 4,
//   DT_INT16 = 5,
//   DT_INT8 = 6,
//   DT_STRING = 7,
//   DT_COMPLEX64 = 8,
//   DT_INT64 = 9,
//   DT_BOOL = 10,
//   DT_QINT8 = 11,
//   DT_QUINT8 = 12,
//   DT_QINT32 = 13,
//   DT_BFLOAT16 = 14,
//   DT_QINT16 = 15,
//   DT_QUINT16 = 16,
//   DT_UINT16 = 17,
//   DT_COMPLEX128 = 18,
//   DT_HALF = 19,
//   DT_RESOURCE = 20,
//   DT_VARIANT = 21,
//   DT_UINT32 = 22,
//   DT_UINT64 = 23,
//   DT_FLOAT_REF = 101,
//   DT_DOUBLE_REF = 102,
//   DT_INT32_REF = 103,
//   DT_UINT8_REF = 104,
//   DT_INT16_REF = 105,
//   DT_INT8_REF = 106,
//   DT_STRING_REF = 107,
//   DT_COMPLEX64_REF = 108,
//   DT_INT64_REF = 109,
//   DT_BOOL_REF = 110,
//   DT_QINT8_REF = 111,
//   DT_QUINT8_REF = 112,
//   DT_QINT32_REF = 113,
//   DT_BFLOAT16_REF = 114,
//   DT_QINT16_REF = 115,
//   DT_QUINT16_REF = 116,
//   DT_UINT16_REF = 117,
//   DT_COMPLEX128_REF = 118,
//   DT_HALF_REF = 119,
//   DT_RESOURCE_REF = 120,
//   DT_VARIANT_REF = 121,
//   DT_UINT32_REF = 122,
//   DT_UINT64_REF = 123,
//   DataType_INT_MIN_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::min(),
//   DataType_INT_MAX_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::max()
// };

inline tfbe::ElementType cast_to_be(tensorflow::DataType dt)
{
    switch (dt)
    {
        case tensorflow::DT_FLOAT: return tfbe::ElementType::Float32_t;
        case tensorflow::DT_DOUBLE: return tfbe::ElementType::Float64_t;
        case tensorflow::DT_INT32: return tfbe::ElementType::Int32_t;
        case tensorflow::DT_UINT8: return tfbe::ElementType::UInt8_t;
        case tensorflow::DT_INT16: return tfbe::ElementType::Int16_t;
        case tensorflow::DT_INT8: return tfbe::ElementType::Int8_t;
        case tensorflow::DT_INT64: return tfbe::ElementType::Int64_t;
        case tensorflow::DT_BOOL: return tfbe::ElementType::Bool_t;
        case tensorflow::DT_BFLOAT16: return tfbe::ElementType::BFloat16_t;
        case tensorflow::DT_UINT16: return tfbe::ElementType::UInt16_t;
        case tensorflow::DT_HALF: return tfbe::ElementType::Float16_t;
        case tensorflow::DT_UINT32: return tfbe::ElementType::UInt32_t;
        case tensorflow::DT_UINT64: return tfbe::ElementType::UInt64_t;
        default: break;
    }
    // backend unsupport tensorflow type!
    CHECK(false);
}

inline tensorflow::DataType cast_to_tf(tfbe::ElementType dt)
{
    switch (dt)
    {
        case tfbe::ElementType::Float32_t: return tensorflow::DT_FLOAT;
        case tfbe::ElementType::Float64_t: return tensorflow::DT_DOUBLE;
        case tfbe::ElementType::Int32_t: return tensorflow::DT_INT32;
        case tfbe::ElementType::UInt8_t: return tensorflow::DT_UINT8;
        case tfbe::ElementType::Int16_t: return tensorflow::DT_INT16;
        case tfbe::ElementType::Int8_t: return tensorflow::DT_INT8;
        case tfbe::ElementType::Int64_t: return tensorflow::DT_INT64;
        case tfbe::ElementType::Bool_t: return tensorflow::DT_BOOL;
        case tfbe::ElementType::BFloat16_t: return tensorflow::DT_BFLOAT16;
        case tfbe::ElementType::UInt16_t: return tensorflow::DT_UINT16;
        case tfbe::ElementType::Float16_t: return tensorflow::DT_HALF;
        case tfbe::ElementType::UInt32_t: return tensorflow::DT_UINT32;
        case tfbe::ElementType::UInt64_t: return tensorflow::DT_UINT64;
        default: break;
    }
    // backend unsupport tensorflow type!
    CHECK(false);
}

inline tfbe::Tensor cast(const tensorflow::Tensor& tensor) {}

} // namespace adaptor

#endif