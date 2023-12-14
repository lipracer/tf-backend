#include "op_registry.h"

#include <utility>

#include <string.h>

#include "logger/logger.h"

namespace tfbe
{

//============================================================//
// OpDefBuilder
//============================================================//

struct OpDefStorage
{
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> attrs;
    int64_t priority = 0;
};

OpDef::OpDef() : storage_(new OpDefStorage()) {}
OpDef::~OpDef()
{
    delete storage_;
}

StringRef OpDef::name() const
{
    return storage_->name;
}

OpDef::string_range OpDef::inputs() const
{
    return storage_->inputs;
}

OpDef::string_range OpDef::outputs() const
{
    return storage_->outputs;
}
OpDef::string_range OpDef::attrs() const
{
    return storage_->attrs;
}

int64_t OpDef::priority() const
{
    return storage_->priority;
}

void OpDef::setName(StringRef name)
{
    storage_->name = name.str();
}
void OpDef::appendInputs(StringRef str)
{
    storage_->inputs.push_back(str.str());
}
void OpDef::appendOutputs(StringRef str)
{
    storage_->outputs.push_back(str.str());
}
void OpDef::appendAttrs(StringRef str)
{
    storage_->attrs.push_back(str.str());
}

void OpDef::setPriority(int64_t p)
{
    storage_->priority = p;
}

// TODO refine this, split to compile time and runtime
const BoxedFunc& OpDef::get_boxed_func() const
{
    if (!boxedFunc_)
    {
        boxedFunc_ = [&](TfbeOpCallStack stack) mutable -> void
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
    op_def_->setName(str);
    OpLibs::instance().emplace_lib(op_def_);
    return *this;
}

OpDefBuilder OpDefBuilder::Input(const std::string& str)
{
    op_def_->appendInputs(str);
    return *this;
}

OpDefBuilder OpDefBuilder::Output(const std::string& str)
{
    op_def_->appendOutputs(str);
    return *this;
}

OpDefBuilder OpDefBuilder::Attr(const std::string& str)
{
    op_def_->appendAttrs(str);
    return *this;
}

OpDefBuilder OpDefBuilder::Device(const std::string& str)
{
    return *this;
}

OpDefBuilder OpDefBuilder::Priority(int64_t p)
{
    op_def_->setPriority(p);
    return *this;
}

OpDefBuilder OpDefBuilder::KernelCreator(const OpDef::KernelCreatorT& creator)
{
    op_def_->kernel_creator() = creator;
    return *this;
}

OpDefBuilder::OpDefBuilder(StringRef file)
{
    LOG(INFO) << "registe op name:" << file;
    op_def_ = new OpDef();
}

//============================================================//
// OpLibs
//============================================================//

OpDef& OpLibs::emplace_lib(OpDef* opdef)
{
    std::lock_guard<std::mutex> lg(libs_mtx_);
    libs_.emplace_back(std::move(opdef));
    map_[libs_.back()->name().c_str()].insert(libs_.back().get());
    return *libs_.back();
}

OpDef* OpLibs::lookup(const char* name)
{
    std::lock_guard<std::mutex> lg(libs_mtx_);
    auto iter = map_.find(name);
    if (iter == map_.end())
    {
        return nullptr;
    }
    return *iter->second.begin();
}

OpLibs& OpLibs::instance()
{
    static OpLibs __instance;
    return __instance;
}

OpLibs::Iterator OpLibs::begin() const
{
    return OpLibs::Iterator(map_.begin());
}

OpLibs::Iterator OpLibs::end() const
{
    return OpLibs::Iterator(map_.end());
}

// don't use &&, the template args is real args
template <typename... Args>
struct ForwardHelper
{
    using OpFunc = void (*)(Args...);

    static void forward_parameter(OpFunc func, TfbeOpCallStack& stack)
    {
        __forward_parameter(func, stack, std::make_index_sequence<sizeof...(Args)>());
    }

    template <size_t... Idx>
    static void __forward_parameter(OpFunc func, TfbeOpCallStack& stack, std::index_sequence<Idx...> = {})
    {
        func(stack.tensors[Idx]...);
    }
};

std::ostream& operator<<(std::ostream& os, const OpDef& op_def)
{
    os << "{"
       << "name:" << op_def.name().c_str() << ", priority:" << op_def.priority() << "}";
    return os;
}

} // namespace tfbe
