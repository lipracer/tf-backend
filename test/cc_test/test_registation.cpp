#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
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

// TEST(OpRegistryTest, forward_op)
// {
//     using forward_helper = ForwardHelper<int, float, float&>;
//     int lhs = 1;
//     float rhs = 2.0f;
//     float ret = 0.0f;
//     tfbe::Any alhs(lhs);
//     tfbe::Any arhs(rhs);
//     tfbe::Any aret(ret, Any::by_reference_tag());
//     // std::vector<tfbe::Any> stack = {std::move(alhs), std::move(arhs), std::move(aret)};
//     std::vector<tfbe::Any> stack;
//     stack.emplace_back(std::move(alhs));
//     stack.emplace_back(std::move(arhs));
//     stack.emplace_back(std::move(aret));
//     forward_helper::forward_parameter(&dummy_op, stack);
//     EXPECT_FLOAT_EQ(lhs + rhs, ret);
// }

TEST(OpRegistryTest, register_op)
{
    REGISTER_KERNEL("AddV2", AddV2).Input("T").Output("T").Attr("N > 0");
    auto op_def = lookupOpDef(getOpLibs(), "BackendOpAddV2");
    EXPECT_EQ(op_def->name(), "AddV2");
}