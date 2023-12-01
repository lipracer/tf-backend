

#include "op_registry.h"

#include "kernel_context.h"
#include "logger/logger.h"
#include "type/tensor.h"

class AddV2 : public tfbe::DeviceOpKernel<AddV2>
{
public:
    using DeviceOpKernel<AddV2>::DeviceOpKernel;
    void compute(tfbe::DeviceOpKernelContext* ctx) {}
};

REGISTER_KERNEL("AddV2", AddV2).Output("results : T").Input("lhs : Tensor").Input("rhs : Tensor");
