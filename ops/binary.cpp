
#include "adt/tensor.h"
#include "kernel_context.h"
#include "logger/logger.h"
#include "op_registry.h"

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

        auto cpu_lhs = lhs.to(DeviceType::CPU);
        auto cpu_rhs = rhs.to(DeviceType::CPU);

        if (cpu_lhs.totalElements() > cpu_rhs.totalElements())
        {
            cpu_rhs = cpu_rhs.broadcast(cpu_lhs.shape());
        }
        else if (cpu_lhs.totalElements() < cpu_rhs.totalElements())
        {
            cpu_lhs = cpu_lhs.broadcast(cpu_rhs.shape());
        }
        Tensor result = cpu_lhs + cpu_rhs;

        LOG(INFO) << "lhs:" << cpu_lhs;
        LOG(INFO) << "rhs:" << cpu_rhs;
        LOG(INFO) << "result:" << result;

        result = result.to(DeviceType::GPU);
        ctx->setOutput(2, result);
    }
};

class Neg : public DeviceOpKernel<Neg>
{
public:
    using DeviceOpKernel<Neg>::DeviceOpKernel;
    void compute(DeviceOpKernelContext* ctx)
    {
        auto input = ctx->input(0);
        auto cpu_input = input.to(DeviceType::CPU);
        auto result = -cpu_input;

        LOG(INFO) << "input:" << cpu_input;
        LOG(INFO) << "result:" << result;

        ctx->setOutput(1, result.to(DeviceType::GPU));
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
