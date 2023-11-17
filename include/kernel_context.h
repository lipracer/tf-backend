#pragma once

#include <memory>
#include <vector>

#include "type/tensor.h"

namespace tfbe
{

struct OpCallStack
{
    Tensor* tensors;
    size_t size;
};

class CompilerContextImpl;

class CompilerContext
{
public:
    CompilerContext() = default;

private:
    CompilerContextImpl* impl_;
};

class DeviceOpKernelContextImpl;

class DeviceOpKernelContext
{
public:
    DeviceOpKernelContext();

    void pushStack(OpCallStack stack);

    void popStack();

    Tensor input(size_t idx);

    // template <typename T>
    // Tensor input(size_t idx);

    void setOutput(size_t idx, Tensor tensor);

private:
    std::shared_ptr<DeviceOpKernelContextImpl> impl_;
};

} // namespace tfbe