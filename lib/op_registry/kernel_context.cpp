
#include "kernel_context.h"

#include "adt/tensor.h"

namespace tfbe
{
//============================================================//
// DeviceOpKernelContextImpl
//============================================================//
class DeviceOpKernelContextImpl
{
public:
    Tensor input(size_t idx);
    void setOutput(size_t idx, Tensor tensor);

    void pushStack(TfbeOpCallStack stack)
    {
        callStack_.push_back(stack);
    }

    void popStack()
    {
        callStack_.pop_back();
    }

private:
    std::vector<TfbeOpCallStack> callStack_;
};

Tensor DeviceOpKernelContextImpl::input(size_t idx)
{
    return callStack_.back().tensors[idx];
}

void DeviceOpKernelContextImpl::setOutput(size_t idx, Tensor tensor)
{
    callStack_.back().tensors[idx] = tensor;
}

//============================================================//
// DeviceOpKernelContext
//============================================================//

DeviceOpKernelContext::DeviceOpKernelContext()
{
    impl_ = std::make_shared<DeviceOpKernelContextImpl>();
}

void DeviceOpKernelContext::pushStack(TfbeOpCallStack stack)
{
    impl_->pushStack(stack);
}

void DeviceOpKernelContext::popStack()
{
    impl_->popStack();
}

Tensor DeviceOpKernelContext::input(size_t idx)
{
    return impl_->input(idx);
}

void DeviceOpKernelContext::setOutput(size_t idx, Tensor tensor)
{
    impl_->setOutput(idx, tensor);
}

DeviceStream DeviceOpKernelContext::getCurrentStream()
{
    return {};
}

} // namespace tfbe