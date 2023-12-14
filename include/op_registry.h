
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <set>
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
    DeviceOpKernelBase(const CompilerContext* ctx) {}

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
    using KernelCreatorT = std::function<KernelObjT(const CompilerContext*)>;

    // using string_range = iterator_range<mapped_iterator<std::vector<std::string>::const_iterator, StringRef>>;
    using string_range = ArrayRef<std::string>;

    StringRef name() const;
    string_range inputs() const;
    string_range outputs() const;
    string_range attrs() const;
    int64_t priority() const;

    void setName(StringRef name);
    void appendInputs(StringRef);
    void appendOutputs(StringRef);
    void appendAttrs(StringRef);
    void setPriority(int64_t p);

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
    friend std::ostream& operator<<(std::ostream& os, const OpDef& op_def);
};

class BE_EXPORT OpDefBuilder
{
public:
    OpDefBuilder(StringRef file);

    OpDefBuilder OpName(const std::string& str);
    OpDefBuilder Input(const std::string& str);
    OpDefBuilder Output(const std::string& str);
    OpDefBuilder Attr(const std::string& str);
    OpDefBuilder Device(const std::string& str);
    OpDefBuilder Priority(int64_t p);

    // template <typename... ArgsT, typename... ResultsT>
    // OpDefBuilder Kernel(void (*func)(ArgsT..., ResultsT...))
    // {
    //     return {};
    // }

    OpDefBuilder KernelCreator(const OpDef::KernelCreatorT& creator);

private:
    OpDef* op_def_{nullptr};
};

class BE_EXPORT OpLibs
{
    struct OpDefCompare
    {
        bool operator()(const OpDef* lhs, const OpDef* rhs) const
        {
            return lhs->priority() >= rhs->priority();
        }
    };

public:
    static OpLibs& instance();

    using OIterator = std::unordered_map<std::string, std::set<OpDef*, class OpDefCompare>>::const_iterator;

    struct Iterator
    {
        Iterator() = default;
        Iterator(OIterator iterator) : iterator_(iterator) {}
        Iterator operator++()
        {
            return Iterator(++iterator_);
        }
        Iterator operator++(int)
        {
            return Iterator(iterator_++);
        }

        OpDef& operator*()
        {
            return *operator->();
        }
        const OpDef& operator*() const
        {
            return *operator->();
        }
        OpDef* operator->()
        {
            return *iterator_->second.begin();
        }

        const OpDef* operator->() const
        {
            return *iterator_->second.begin();
        }

        bool operator==(const Iterator& other) const
        {
            return iterator_ == other.iterator_;
        }

        bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

    private:
        OIterator iterator_;
    };

    OpDef& emplace_lib(OpDef* opdef);
    // if can not find return null
    OpDef* lookup(const char* name);

    Iterator begin() const;

    Iterator end() const;

private:
    OpLibs() = default;
    std::vector<std::unique_ptr<OpDef>> libs_;
    std::unordered_map<std::string, std::set<OpDef*, OpDefCompare>> map_;
    std::mutex libs_mtx_;
};

} // namespace tfbe

#define TFBE_MACRO_CONCAT_IMPL(a, b) a##b
#define TFBE_MACRO_CONCAT(a, b) TFBE_MACRO_CONCAT_IMPL(a, b)
#define TFBE_MACRO_UNIQUE_NAME(a) TFBE_MACRO_CONCAT(a, __COUNTER__)

#define REGISTER_KERNEL(name, kernel)                                                                  \
    [[maybe_unused]] static ::tfbe::OpDefBuilder TFBE_MACRO_UNIQUE_NAME(TFBE_REGISTER_) =              \
        ::tfbe::OpDefBuilder(__FILE__).OpName(name).KernelCreator([](const tfbe::CompilerContext* ctx) \
                                                                  { return std::make_shared<kernel>(ctx); })
