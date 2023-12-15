#pragma once

#include "adt/tensor.h"
#include "adt/meta_helper.h"

// !!!
// NOTE: we must named all of parameter, then use this name for reflection in the codegen process
// !!!
namespace tfbe
{
namespace autogen
{
Tensor AddV2(AnyOpaque opaque, const Tensor& lhs, const Tensor& rhs);

// dummy op for test codegen
// Tensor AddV3(const Tensor& lhs, const Tensor& rhs, float alpha, const std::vector<int>& shape);
}
} // namespace tfbe