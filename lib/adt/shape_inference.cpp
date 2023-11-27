#include "internal/shape_inference.h"

namespace tfbe
{
ShapeType<DimT> BroadcastInDim(ArrayRef<DimT> lhs, ArrayRef<DimT> rhs, ArrayRef<DimT>)
{
    return {};
}

} // namespace tfbe