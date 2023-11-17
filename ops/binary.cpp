
#include "kernel_context.h"
#include "logger/logger.h"
#include "op_registry.h"
#include "type/tensor.h"

namespace tfbe
{
class AddV2 : public DeviceOpKernel<AddV2>
{
public:
    using DeviceOpKernel<AddV2>::DeviceOpKernel;
    void compute(DeviceOpKernelContext* ctx)
    {
        auto lhs = ctx->input(0);
        auto rhs = ctx->input(1);
        ctx->setOutput(2, lhs + rhs);
    }
};

class Neg : public DeviceOpKernel<Neg>
{
public:
    using DeviceOpKernel<Neg>::DeviceOpKernel;
    void compute(DeviceOpKernelContext* ctx)
    {
        auto lhs = ctx->input(0);
        ctx->setOutput(1, -lhs);
    }
};

} // namespace tfbe

// REGISTER_OP("AddV2BackendOp")
//     .Input("args0: T")
//     .Input("args1: T")
//     .Output("results: T")
//     .Attr("T: type")
//     // XLA random-number generation ops are stateful.
//     // TODO(phawkins): create stateful and non-stateful variants of _XlaRun.
//     .SetIsStateful()
//     .Doc(R"()");

// REGISTER_OP("NegBackendOp")
//     .Input("arg: T")
//     .Output("result: T")
//     .Attr("T: type")
//     // XLA random-number generation ops are stateful.
//     // TODO(phawkins): create stateful and non-stateful variants of _XlaRun.
//     .SetIsStateful()
//     .Doc(R"()");

REGISTER_KERNEL("AddV2", tfbe::AddV2).Input("args0: T").Input("args1: T").Output("results: T").Attr("T: type");
REGISTER_KERNEL("Neg", tfbe::Neg).Input("arg: T").Output("result: T").Attr("T: type");
