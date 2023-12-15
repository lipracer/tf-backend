#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger/logger.h"
#include "op_registry.h"

using namespace tfbe;

void dummy_op(int lhs, float rhs, float& ret)
{
    ret = lhs + rhs;
    return;
}

class AddV2 : public DeviceOpKernel<AddV2>
{
public:
    using DeviceOpKernel<AddV2>::DeviceOpKernel;
    void compute(DeviceOpKernelContext* ctx)
    {
        auto lhs = ctx->input(0);
        auto rhs = ctx->input(1);
        ctx->setOutput(2, lhs + rhs);
    }
};

TEST(OpRegistryTest, register_op)
{
    REGISTER_KERNEL("AddV2", AddV2).Input("T").Output("T").Attr("N > 0");
    auto op_def = TfbeLookupOpDef(TfbeGetOpLibs(), "AddV2");
    EXPECT_EQ(reinterpret_cast<tfbe::OpDef*>(op_def.data)->name(), "AddV2");
}

TEST(OpRegistryTest, nativa_register_op)
{
    for (auto& op_def : tfbe::OpLibs::instance())
    {
        LOG(INFO) << op_def;
    }
    auto op_def = TfbeLookupOpDef(TfbeGetOpLibs(), "AddV2");
    LOG(INFO) << "first priority:" << *reinterpret_cast<tfbe::OpDef*>(op_def.data);
    EXPECT_EQ(reinterpret_cast<tfbe::OpDef*>(op_def.data)->name(), "AddV2");
}