#pragma once

#include <memory>
#include <vector>

#include "type/tensor.h"
#include "device.h"

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

    template <typename T>
    T getAttr(const std::string& key)
    {
        return {};
    }

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

    DeviceStream getCurrentStream();

private:
    std::shared_ptr<DeviceOpKernelContextImpl> impl_;
};

} // namespace tfbe