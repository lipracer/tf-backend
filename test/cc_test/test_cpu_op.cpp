#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger/logger.h"
#include "type/tensor.h"

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
}
