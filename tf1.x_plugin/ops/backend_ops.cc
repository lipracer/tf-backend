
#include "tensorflow/compiler/jit/tf1.x_plugin/include/op_registry.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{

using shape_inference::InferenceContext;

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

REGISTER_OP("H2DOp")
    .Input("arg: T")
    .Output("result: T")
    .Attr("T: type")
    // .Attr("N: int >= 0")
    // XLA random-number generation ops are stateful.
    // TODO(phawkins): create stateful and non-stateful variants of _XlaRun.
    .SetIsStateful()
    .Doc(R"()");

REGISTER_OP("D2HOp")
    .Input("arg: T")
    .Output("result: T")
    .Attr("T: type")
    // .Attr("N: int >= 0")
    // XLA random-number generation ops are stateful.
    // TODO(phawkins): create stateful and non-stateful variants of _XlaRun.
    .SetIsStateful()
    .Doc(R"()");

class BackendResistry
{
public:
    BackendResistry()
    {
        for (const auto& it : tfbe::OpLibs::instance().libs())
        {
            string name = tfbe::OpNamePrefix + it->name();
            auto wrap = ::tensorflow::register_op::OpDefBuilderWrapper<true>(name.c_str());
            for (const auto& input : it->inputs())
            {
                wrap.Input(input);
            }
            for (const auto& output : it->outputs())
            {
                wrap.Output(output);
            }
            for (const auto& attr : it->attrs())
            {
                wrap.Attr(attr);
            }
            wrap.SetIsStateful().Doc(R"()");
            new ::tensorflow::register_op::OpDefBuilderReceiver(wrap);
        }
    }
};

static BackendResistry __registry;

} // namespace tensorflow
