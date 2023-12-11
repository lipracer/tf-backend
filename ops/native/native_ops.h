#pragma once

#include "adt/tensor.h"

// !!!
// NOTE: we must named all of parameter, then use this name for reflection in the codegen process
// !!!
namespace tfbe
{
namespace autogen
{
Tensor AddVN(const Tensor& lhs, const Tensor& rhs);

Tensor AddV3(const Tensor& lhs, const Tensor& rhs, float alpha, const std::vector<int>& shape);
}
} // namespace tfbe