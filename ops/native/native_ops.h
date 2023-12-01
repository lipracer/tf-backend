#pragma once

#include "type/tensor.h"

namespace tfbe
{
namespace autogen
{
Tensor AddVN(const Tensor& lhs, const Tensor& rhs);

Tensor AddV3(const Tensor& lhs, const Tensor& rhs, float alpha);
}
} // namespace tfbe