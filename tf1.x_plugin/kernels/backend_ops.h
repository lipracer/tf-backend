#ifndef TENSORFLOW_KERNELS_BACKEND_OP_H_
#define TENSORFLOW_KERNELS_BACKEND_OP_H_

#include <unordered_map>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow
{

class H2DOp : public OpKernel
{
public:
    explicit H2DOp(OpKernelConstruction* ctx);
    ~H2DOp();
    void Compute(OpKernelContext* ctx) override;
};

class D2HOp : public OpKernel
{
public:
    explicit D2HOp(OpKernelConstruction* ctx);
    ~D2HOp();
    void Compute(OpKernelContext* ctx) override;
};

class BackendOp : public OpKernel
{
public:
    explicit BackendOp(OpKernelConstruction* ctx);
    ~BackendOp();
    void Compute(OpKernelContext* ctx) override;
};

} // namespace tensorflow
#endif // TENSORFLOW_KERNELS_BACKEND_OP_H_
