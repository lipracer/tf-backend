
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernel_context.h"
#include "macro.h"
#include "type/tensor.h"

namespace tfbe
{

using BoxedFunc = std::function<void(OpCallStack)>;

class CompilerContext;
class DeviceOpKernelContext;

class DeviceOpKernelBase
{
public:
    using Base = DeviceOpKernelBase;
    // this func wiil called at compile time you need save the constant value to youer kernel
    DeviceOpKernelBase(CompilerContext* ctx) {}

    virtual void compute_impl(DeviceOpKernelContext* ctx) = 0;
};

template <typename DerivedT>
class DeviceOpKernel : public DeviceOpKernelBase
{
public:
    using Base::Base;
    void compute_impl(DeviceOpKernelContext* ctx) override
    {
        static_cast<DerivedT*>(this)->compute(ctx);
    }

    void compute(DeviceOpKernelContext* ctx)
    {
        EXPECT(false, "this function need derived class implement");
    }
};

constexpr const char OpNamePrefix[] = "BackendOp";

class BE_EXPORT OpDef
{
public:
    using KernelObjT = std::shared_ptr<DeviceOpKernelBase>;
    using KernelCreatorT = std::function<KernelObjT(CompilerContext*)>;

    std::string& name();
    std::vector<std::string>& inputs();
    std::vector<std::string>& outputs();
    std::vector<std::string>& attrs();

    KernelCreatorT& kernel_creator();

    const BoxedFunc& get_boxed_func() const;
    void set_boxed_func(const BoxedFunc&);

private:
    OpDef() = default;
    std::string name_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::vector<std::string> attrs_;

    mutable BoxedFunc boxedFunc_;

    KernelObjT kernel_obj_;
    KernelCreatorT kernel_creator_;

    friend class OpDefBuilder;
};

class BE_EXPORT OpDefBuilder
{
public:
    OpDefBuilder();

    OpDefBuilder OpName(const std::string& str);
    OpDefBuilder Input(const std::string& str);
    OpDefBuilder Output(const std::string& str);
    OpDefBuilder Attr(const std::string& str);
    OpDefBuilder Device(const std::string& str);

    template <typename... ArgsT, typename... ResultsT>
    OpDefBuilder Kernel(void (*func)(ArgsT..., ResultsT...))
    {
        return {};
    }

    OpDefBuilder KernelCreator(const OpDef::KernelCreatorT& creator);

private:
    OpDef* op_def_{nullptr};
};

class BE_EXPORT OpLibs
{
public:
    static OpLibs& instance();

    OpDef& emplace_lib(OpDef* opdef);
    // if can not find return null
    OpDef* lookup(const char* name);

    const std::vector<std::unique_ptr<OpDef>>& libs() const
    {
        return libs_;
    }

private:
    std::vector<std::unique_ptr<OpDef>> libs_;
    std::unordered_map<std::string, OpDef*> map_;
};

typedef void* BEOpLibs_t;

BE_EXPORT void CallFrame(OpDef* op_def, OpCallStack stack);

/* *
 * @brief
 * @name
 */
BE_EXPORT OpDef* lookupOpDef(BEOpLibs_t libs, const char* name);

BE_EXPORT CompilerContext* getCompilerContext();

BE_EXPORT BEOpLibs_t getOpLibs();

} // namespace tfbe

#define TFBE_MACRO_CONCAT_IMPL(a, b) a##b
#define TFBE_MACRO_CONCAT(a, b) TFBE_MACRO_CONCAT_IMPL(a, b)
#define TFBE_MACRO_UNIQUE_NAME(a) TFBE_MACRO_CONCAT(a, __COUNTER__)

#define REGISTER_KERNEL(name, kernel)                                                    \
    static ::tfbe::OpDefBuilder TFBE_MACRO_UNIQUE_NAME(TFBE_REGISTER_) =                 \
        ::tfbe::OpDefBuilder().OpName(name).KernelCreator([](tfbe::CompilerContext* ctx) \
                                                          { return std::make_shared<kernel>(ctx); })
