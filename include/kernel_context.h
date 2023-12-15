#pragma once

#include <memory>
#include <vector>

#include "adt/meta_helper.h"
#include "adt/tensor.h"
#include "capi/backend_api.h"
#include "device.h"

namespace tfbe
{

class CompilerContextImpl;

class CompilerContext
{
public:
    CompilerContext() = default;
    CompilerContext(StringRef name);
    ~CompilerContext();

    template <typename T>
    T getAttr(const std::string& key) const
    {
        return {};
    }

    StringRef name() const;

private:
    CompilerContextImpl* impl_;
};

class DeviceOpKernelContextImpl;

class DeviceOpKernelContext
{
public:
    DeviceOpKernelContext();

    void pushStack(TfbeOpCallStack stack);

    void popStack();

    Tensor input(size_t idx);

    void setOutput(size_t idx, Tensor tensor);

    DeviceStream getCurrentStream();

    AnyOpaque getOpContext();

private:
    std::shared_ptr<DeviceOpKernelContextImpl> impl_;
};

} // namespace tfbe