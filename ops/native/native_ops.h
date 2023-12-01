#pragma once

#include "type/tensor.h"

namespace tfbe
{
namespace autogen
{
Tensor AddV2(const Tensor& lhs, const Tensor& rhs);
}
} // namespace tfbe