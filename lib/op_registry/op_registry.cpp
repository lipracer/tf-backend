#include "op_registry.h"

#include <utility>
#include <string.h>

#include "logger/logger.h"

namespace tfbe
{

//============================================================//
// OpDefBuilder
//============================================================//
std::string& OpDef::name()
{
    return name_;
}
std::vector<std::string>& OpDef::inputs()
{
    return inputs_;
}
std::vector<std::string>& OpDef::outputs()
{
    return outputs_;
}
std::vector<std::string>& OpDef::attrs()
{
    return attrs_;
}

// TODO refine this, split to compile time and runtime
const BoxedFunc& OpDef::get_boxed_func() const
{
    if (!boxedFunc_)
    {
        boxedFunc_ = [&](OpCallStack stack) mutable -> void
        {
            auto cctx = new CompilerContext();
            auto kernel = this->kernel_creator_(cctx);
            auto rctx = new DeviceOpKernelContext();
            rctx->pushStack(stack);
            kernel->compute_impl(rctx);
            rctx->popStack();
        };
    }
    return boxedFunc_;
}

void OpDef::set_boxed_func(const BoxedFunc& func)
{
    boxedFunc_ = func;
}

OpDef::KernelCreatorT& OpDef::kernel_creator()
{
    return kernel_creator_;
}

//============================================================//
// OpDefBuilder
//============================================================//

OpDefBuilder OpDefBuilder::OpName(const std::string& str)
{
    op_def_->name() = str;
    OpLibs::instance().emplace_lib(op_def_);
    return *this;
}

OpDefBuilder OpDefBuilder::Input(const std::string& str)
{
    op_def_->inputs().push_back(str);
    return *this;
}

OpDefBuilder OpDefBuilder::Output(const std::string& str)
{
    op_def_->outputs().push_back(str);
    return *this;
}

OpDefBuilder OpDefBuilder::Attr(const std::string& str)
{
    op_def_->attrs().push_back(str);
    return *this;
}

OpDefBuilder OpDefBuilder::Device(const std::string& str)
{
    return *this;
}

OpDefBuilder OpDefBuilder::KernelCreator(const OpDef::KernelCreatorT& creator)
{
    op_def_->kernel_creator() = creator;
    return *this;
}

OpDefBuilder::OpDefBuilder()
{
    op_def_ = new OpDef();
}

//============================================================//
// OpLibs
//============================================================//

OpDef& OpLibs::emplace_lib(OpDef* opdef)
{
    libs_.emplace_back(std::move(opdef));
    map_.insert(std::make_pair(libs_.back()->name().c_str(), libs_.back().get()));
    return *libs_.back();
}

OpDef* OpLibs::lookup(const char* name)
{
    auto iter = map_.find(name);
    if (iter == map_.end())
    {
        return nullptr;
    }
    return iter->second;
}

OpLibs& OpLibs::instance()
{
    static OpLibs __instance;
    return __instance;
}

// don't use &&, the template args is real args
template <typename... Args>
struct ForwardHelper
{
    using OpFunc = void (*)(Args...);

    static void forward_parameter(OpFunc func, OpCallStack& stack)
    {
        __forward_parameter(func, stack, std::make_index_sequence<sizeof...(Args)>());
    }

    template <size_t... Idx>
    static void __forward_parameter(OpFunc func, OpCallStack& stack, std::index_sequence<Idx...> = {})
    {
        func(stack.tensors[Idx]...);
    }
};

//============================================================//
// Export functions
//============================================================//

void CallFrame(OpDef* op_def, OpCallStack stack)
{
    CHECK(op_def, "box func is null!");
    op_def->get_boxed_func()(stack);
}

OpDef* lookupOpDef(BEOpLibs_t libs, const char* name)
{
    if (!name)
    {
        return nullptr;
    }
    return reinterpret_cast<OpLibs*>(libs)->lookup(name);
}

BE_EXPORT BEOpLibs_t getOpLibs()
{
    return &OpLibs::instance();
}

} // namespace tfbe
