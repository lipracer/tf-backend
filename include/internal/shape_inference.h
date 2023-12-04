#pragma once

#include "../adt/tensor_impl.h"

namespace tfbe
{
ShapeType<DimT> BroadcastInDim(ArrayRef<DimT> lhs, ArrayRef<DimT> rhs, ArrayRef<DimT> = {});

}