
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
        auto libs = TfbeGetOpLibs();
        TfbeForeachOpLibs(
            libs, +[](TfbeOpDef_t op_def) {
                auto ref_name = TfbeGetOpDefName(op_def);
                string name = TfbeOpNamePrefix + ref_name.str();
                auto wrap = ::tensorflow::register_op::OpDefBuilderWrapper<true>(name.c_str());

                {
                    TfbeStringList* inputs = TfbeGetOpDefInputs(op_def);
                    for (size_t i = 0; i < inputs->size; ++i)
                    {
                        wrap.Input(inputs->strs[i].str());
                    }
                    FreeTfbeStringList(inputs);
                }
                {
                    TfbeStringList* outputs = TfbeGetOpDefOutputs(op_def);
                    for (size_t i = 0; i < outputs->size; ++i)
                    {
                        wrap.Output(outputs->strs[i].str());
                    }
                    FreeTfbeStringList(outputs);
                }
                {
                    TfbeStringList* attrs = TfbeGetOpDefAttributes(op_def);
                    for (size_t i = 0; i < attrs->size; ++i)
                    {
                        wrap.Attr(attrs->strs[i].str());
                    }
                    FreeTfbeStringList(attrs);
                }

                wrap.SetIsStateful().Doc(R"()");
                new ::tensorflow::register_op::OpDefBuilderReceiver(wrap);
            });
    }
};

static BackendResistry __registry;

} // namespace tensorflow
