#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "adt/tensor.h"
#include "gtest/gtest.h"

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

#define print_shape(shape)                                \
    std::cout << __LINE__ << " shape:[";                  \
    for (size_t i = 0; i < shape.size(); ++i)             \
    {                                                     \
        std::cout << &shape[i] << " " << shape[i] << " "; \
    }                                                     \
    std::cout << "]" << std::endl;

#define CHECK_SHAPE(lhs, rhs)                   \
    do                                          \
    {                                           \
        for (size_t i = 0; i < lhs.size(); ++i) \
        {                                       \
            EXPECT_EQ(lhs[i], rhs[i]);          \
        }                                       \
    } while (0)

TEST(ShapeType, shape)
{
    ShapeType<DimT> shape({1, 2, 3, 4});
    std::vector<DimT> vec_shape({1, 2, 3, 4});
    print_shape(shape);

    for (size_t i = 0; i < shape.size(); ++i)
    {
        EXPECT_EQ(shape[i], vec_shape[i]);
    }

    vec_shape = {2, 3, 4, 5, 6, 7};
    auto tmp_shape = ShapeType<DimT>(vec_shape.begin(), vec_shape.end());
    shape = std::move(tmp_shape);
    print_shape(shape);

    EXPECT_EQ(shape.size(), 6);

    ShapeType<DimT> shape1 = shape;
    print_shape(shape);
    print_shape(shape1);

    shape.resize(2, 1);
    vec_shape.resize(2, 1);
    print_shape(shape);
    EXPECT_EQ(shape.size(), 2);
    CHECK_SHAPE(shape, vec_shape);

    shape1 = shape;
    print_shape(shape);
    print_shape(shape1);
    EXPECT_EQ(shape.size(), 2);
    EXPECT_EQ(shape1.size(), 2);
    CHECK_SHAPE(shape, vec_shape);
    CHECK_SHAPE(shape1, vec_shape);

    shape1 = std::move(shape);
    print_shape(shape);
    print_shape(shape1);
    EXPECT_EQ(shape.size(), 0);
    CHECK_SHAPE(shape, vec_shape);
    CHECK_SHAPE(shape1, vec_shape);

    ShapeType<DimT> push_back_s;
    std::vector<DimT> vpush_back_s;
    for (size_t i = 0; i < 100; ++i)
    {
        push_back_s.push_back(i + 1);
        vpush_back_s.push_back(i + 1);
    }
    CHECK_SHAPE(push_back_s, vpush_back_s);

    ShapeType<DimT> emplace_back_s;
    std::vector<DimT> vemplace_back_s;
    for (size_t i = 0; i < 100; ++i)
    {
        emplace_back_s.emplace_back(i + 1);
        vemplace_back_s.emplace_back(i + 1);
    }
    CHECK_SHAPE(emplace_back_s, vemplace_back_s);

    EXPECT_EQ(push_back_s.capacity(), 128);
    EXPECT_EQ(emplace_back_s.capacity(), 128);
}
