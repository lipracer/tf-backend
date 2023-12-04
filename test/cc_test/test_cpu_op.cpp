#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "adt/tensor.h"
#include "gtest/gtest.h"
#include "logger/logger.h"

using namespace tfbe;

TEST(TensorOperator, broadcast)
{
    std::vector<DimT> shape = {3, 1};
    std::vector<DimT> new_shape = {3, 2};
    auto tensor = tfbe::empty_tensor(DeviceType::CPU, shape, ElementType::Float32_t);
    tensor.data<float>()[0] = 1.0f;
    tensor.data<float>()[1] = 2.0f;
    tensor.data<float>()[2] = 3.0f;
    auto new_tensor = tensor.broadcast(new_shape);
    LOG(WARN) << tensor;
    LOG(WARN) << new_tensor;

    std::vector<float> golden_values = {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f};
    auto golden_tensor = makeCconstantTensor(golden_values);
    EXPECT_TRUE(allClose(new_tensor, golden_tensor));
}

TEST(TensorOperator, reshape)
{
    std::vector<DimT> shape = {100, 10};
    std::vector<DimT> new_shape = {1000, 1};
    Tensor tensor;
    Tensor new_tensor;
    {
        tensor = tfbe::empty_tensor(DeviceType::CPU, shape, ElementType::Float32_t);
        new_tensor = tensor.reshape(new_shape);
        EXPECT_TRUE(allClose(new_tensor, tensor));
        EXPECT_EQ(tensor.getImpl()->fetch_count(), 1);
        EXPECT_EQ(new_tensor.getImpl()->fetch_count(), 1);
        EXPECT_EQ(tensor.getImpl()->getStorage().use_count(), 2);
    }
}

TEST(TensorOperator, broadcast_mnist)
{
    std::vector<DimT> shape = {10};
    std::vector<DimT> new_shape = {100, 10};
    auto tensor = tfbe::empty_tensor(DeviceType::CPU, shape, ElementType::Float32_t);
    std::iota(tensor.data<float>(), tensor.data<float>() + tensor.totalElements(), 1);
    auto new_tensor = tensor.broadcast(new_shape);
    LOG(WARN) << tensor;
    LOG(WARN) << new_tensor;

    std::vector<float> golden_values(100 * 10);
    for (size_t i = 0; i < 100; ++i)
    {
        std::iota(golden_values.begin() + 10 * i, golden_values.begin() + 10 + 10 * i, 1);
    }
    auto golden_tensor = makeCconstantTensor(golden_values);
    EXPECT_TRUE(allClose(new_tensor, golden_tensor));
}
