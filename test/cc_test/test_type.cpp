#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "type/tensor.h"

using namespace tfbe;

TEST(TensorImplTest, ref_counter)
{
    std::vector<DimT> shape = {3, 2};
    auto a = tfbe::empty_tensor(DeviceType::CPU, shape, ElementType::Float32_t);
    EXPECT_EQ(a.getImpl()->fetch_count(), 1);
    {
        auto b = a;
        EXPECT_EQ(a.getImpl()->fetch_count(), 2);
        EXPECT_EQ(b.getImpl()->fetch_count(), 2);

        IntrusivePtr<TensorImpl> impl(a.getImpl());

        EXPECT_EQ(a.getImpl()->fetch_count(), 3);
        EXPECT_EQ(b.getImpl()->fetch_count(), 3);

        EXPECT_EQ(impl->fetch_count(), 3);
    }
    EXPECT_EQ(a.getImpl()->fetch_count(), 1);
}

static Tensor DummyFunc(Tensor tensor, size_t& counter)
{
    counter = tensor.getImpl()->fetch_count();
    return tensor;
}

TEST(TensorImplTest, ref_counter_copy)
{
    std::vector<DimT> shape = {3, 2};
    auto tensor = tfbe::empty_tensor(DeviceType::CPU, shape, ElementType::Float32_t);
    EXPECT_EQ(tensor.getImpl()->fetch_count(), 1);
    size_t counter = 0;
    std::cout << "ref count:" << tensor.getImpl()->fetch_count() << std::endl;
    {
        auto tensor1 = DummyFunc(tensor, counter);
        std::cout << "ref count:" << tensor.getImpl()->fetch_count() << std::endl;
        EXPECT_EQ(counter, 2);
        std::cout << "ref count:" << tensor.getImpl()->fetch_count() << std::endl;
        EXPECT_EQ(tensor.getImpl()->fetch_count(), 2);
    }
    EXPECT_EQ(tensor.getImpl()->fetch_count(), 1);
}
