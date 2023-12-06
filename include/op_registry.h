
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "adt/ArrayRef.h"
#include "adt/StringRef.h"
#include "adt/iterator_range.h"
#include "adt/tensor.h"
#include "kernel_context.h"
#include "macro.h"

namespace tfbe
{
using BoxedFunc = std::function<void(TfbeOpCallStack)>;

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
        BE_EXPECT(false, "this function need derived class implement");
    }
};

class OpDefStorage;

class BE_EXPORT OpDef
{
public:
    using KernelObjT = std::shared_ptr<DeviceOpKernelBase>;
    using KernelCreatorT = std::function<KernelObjT(CompilerContext*)>;

    // using string_range = iterator_range<mapped_iterator<std::vector<std::string>::const_iterator, StringRef>>;
    using string_range = ArrayRef<std::string>;

    StringRef name();
    string_range inputs();
    string_range outputs();
    string_range attrs();

    void setName(StringRef name);
    void appendInputs(StringRef);
    void appendOutputs(StringRef);
    void appendAttrs(StringRef);

    KernelCreatorT& kernel_creator();

    const BoxedFunc& get_boxed_func() const;
    void set_boxed_func(const BoxedFunc&);

    ~OpDef();

private:
    OpDef();

    OpDef(const OpDef& other) = delete;
    OpDefStorage* storage_;

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

} // namespace tfbe

#define TFBE_MACRO_CONCAT_IMPL(a, b) a##b
#define TFBE_MACRO_CONCAT(a, b) TFBE_MACRO_CONCAT_IMPL(a, b)
#define TFBE_MACRO_UNIQUE_NAME(a) TFBE_MACRO_CONCAT(a, __COUNTER__)

#define REGISTER_KERNEL(name, kernel)                                                     \
    [[maybe_unused]] static ::tfbe::OpDefBuilder TFBE_MACRO_UNIQUE_NAME(TFBE_REGISTER_) = \
        ::tfbe::OpDefBuilder().OpName(name).KernelCreator(                                \
            [](tfbe::CompilerContext* ctx) { return std::make_shared<kernel>(ctx); })
