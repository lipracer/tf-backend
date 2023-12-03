
#include "backend_ops.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>

#include "tensorflow/compiler/jit/tf1.x_plugin/include/adaptor.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/dump_graph.h"

// #include "tensorflow/compiler/jit/tf1.x_plugin/include/adaptor.h"
#include "tensorflow/compiler/jit/tf1.x_plugin/include/op_registry.h"

namespace tensorflow
{

#define LOCK_SCOPE         \
    static std::mutex mtx; \
    std::lock_guard<std::mutex> _(mtx);
#define LOCK_SCOPE

class BackendAllocator : public Allocator
{
public:
    // Align to 64 byte boundary.
    static constexpr size_t kAllocatorAlignment = 64;

    ~BackendAllocator() override;

    // Return a string identifying this allocator
    string Name() override;

    // Return an uninitialized block of memory that is "num_bytes" bytes
    // in size.  The returned pointer is guaranteed to be aligned to a
    // multiple of "alignment" bytes.
    // REQUIRES: "alignment" is a power of 2.
    void* AllocateRaw(size_t alignment, size_t num_bytes) override;

    // Deallocate a block of memory pointer to by "ptr"
    // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
    void DeallocateRaw(void* ptr) override;

    // Returns the user-requested size of the data allocated at
    // 'ptr'.  Note that the actual buffer allocated might be larger
    // than requested, but this function returns the size requested by
    // the user.
    //
    // REQUIRES: TracksAllocationSizes() is true.
    //
    // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
    // allocated by this allocator.
    size_t RequestedSize(const void* ptr) const override;

    void setCurrentTensor(tfbe::TensorImpl* impl)
    {
        mtx_.lock();
        impl_ = impl;
    }

private:
    tfbe::TensorImpl* impl_{nullptr};
    std::recursive_mutex mtx_;
};

BackendAllocator::~BackendAllocator() {}

string BackendAllocator::Name()
{
    return "BackendAllocator";
}

void* BackendAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    auto ret = impl_;
    mtx_.unlock();
    return ret;
}

void BackendAllocator::DeallocateRaw(void* ptr)
{
    if (!ptr)
    {
        return;
    }
    auto impl = reinterpret_cast<tfbe::TensorImpl*>(ptr);
    impl->decrease_ref();
}

size_t BackendAllocator::RequestedSize(const void* ptr) const
{
    return reinterpret_cast<const tfbe::TensorImpl*>(ptr)->numBytes();
}

static BackendAllocator backendAllocator;

tfbe::ShapeType<tfbe::DimT> makeShape(const Tensor& tensor)
{
    tfbe::ShapeType<tfbe::DimT> shape;
    for (size_t i = 0; i < tensor.dims(); ++i)
    {
        shape.push_back(tensor.dim_size(i));
    }
    return shape;
}

tfbe::Tensor makeBeTensor(const Tensor& tensor)
{
    tfbe::ShapeType<tfbe::DimT> shape = makeShape(tensor);
    tfbe::Tensor deviceTensor = tfbe::empty_tensor(tfbe::DeviceType::CPU, shape, tfbe::ElementType::Float32_t, true);
    memcpy(const_cast<void*>(deviceTensor.data()), const_cast<char*>(tensor.tensor_data().data()), tensor.TotalBytes());
    return deviceTensor;
}

tfbe::Tensor TfTensorToBeTensor(const Tensor& tensor)
{
    LOCK_SCOPE
    auto impl = reinterpret_cast<tfbe::TensorImpl*>(const_cast<char*>(tensor.tensor_data().data()));
    return tfbe::Tensor(impl);
}

// enum DataType : int {
//   DT_INVALID = 0,
//   DT_FLOAT = 1,
//   DT_DOUBLE = 2,
//   DT_INT32 = 3,
//   DT_UINT8 = 4,
//   DT_INT16 = 5,
//   DT_INT8 = 6,
//   DT_STRING = 7,
//   DT_COMPLEX64 = 8,
//   DT_INT64 = 9,
//   DT_BOOL = 10,
//   DT_QINT8 = 11,
//   DT_QUINT8 = 12,
//   DT_QINT32 = 13,
//   DT_BFLOAT16 = 14,
//   DT_QINT16 = 15,
//   DT_QUINT16 = 16,
//   DT_UINT16 = 17,
//   DT_COMPLEX128 = 18,
//   DT_HALF = 19,
//   DT_RESOURCE = 20,
//   DT_VARIANT = 21,
//   DT_UINT32 = 22,
//   DT_UINT64 = 23,
//   DT_FLOAT_REF = 101,
//   DT_DOUBLE_REF = 102,
//   DT_INT32_REF = 103,
//   DT_UINT8_REF = 104,
//   DT_INT16_REF = 105,
//   DT_INT8_REF = 106,
//   DT_STRING_REF = 107,
//   DT_COMPLEX64_REF = 108,
//   DT_INT64_REF = 109,
//   DT_BOOL_REF = 110,
//   DT_QINT8_REF = 111,
//   DT_QUINT8_REF = 112,
//   DT_QINT32_REF = 113,
//   DT_BFLOAT16_REF = 114,
//   DT_QINT16_REF = 115,
//   DT_QUINT16_REF = 116,
//   DT_UINT16_REF = 117,
//   DT_COMPLEX128_REF = 118,
//   DT_HALF_REF = 119,
//   DT_RESOURCE_REF = 120,
//   DT_VARIANT_REF = 121,
//   DT_UINT32_REF = 122,
//   DT_UINT64_REF = 123,
//   DataType_INT_MIN_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::min(),
//   DataType_INT_MAX_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::max()
// };

Tensor BeTensorToTfTensor(tfbe::Tensor tensor)
{
    backendAllocator.setCurrentTensor(tensor.getImpl().get());
    std::vector<int64> vShape(std::begin(tensor.shape()), std::end(tensor.shape()));
    TensorShape* shape = new TensorShape(vShape);
    return Tensor(&backendAllocator, DT_FLOAT, *shape);
}

H2DOp::H2DOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

H2DOp::~H2DOp() {}

void H2DOp::Compute(OpKernelContext* ctx)
{
    LOCK_SCOPE;
    // LOG(INFO) << "H2DOp::Compute op name:" << ctx->op_kernel().name();
    auto deviceTensor = makeBeTensor(ctx->input(0));
    // cpu tensor we release immediately avoid leak
    // deviceTensor.getImpl()->increase_ref();

    deviceTensor = deviceTensor.to(tfbe::DeviceType::GPU);
    // LOG(INFO) << "h2d result:" << deviceTensor;
    // input need always alive
    Tensor tfDeviceTensor = BeTensorToTfTensor(deviceTensor);
    // GPU tensor need always alive
    deviceTensor.getImpl()->increase_ref();
    ctx->set_output(0, tfDeviceTensor);
}

D2HOp::D2HOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

D2HOp::~D2HOp() {}

void D2HOp::Compute(OpKernelContext* ctx)
{
    LOCK_SCOPE;
    // LOG(INFO) << "D2HOp::Compute op name:" << ctx->op_kernel().name();
    {
        // need check input is tf or be
        auto deviceTensor = TfTensorToBeTensor(ctx->input(0));
        deviceTensor = deviceTensor.to(tfbe::DeviceType::CPU);
        Tensor* output = nullptr;
        ctx->allocate_output(0, ctx->input(0).shape(), &output);
        memcpy(const_cast<char*>(output->tensor_data().data()), deviceTensor.data<float>(), deviceTensor.numBytes());
    }
}

BackendOp::BackendOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

BackendOp::~BackendOp() {}

void BackendOp::Compute(OpKernelContext* ctx)
{
    LOCK_SCOPE;
    // LOG(INFO) << "BackendOp::Compute op name:" << ctx->op_kernel().name()
    //           << " kernel name:" << ctx->op_kernel().type_string().c_str();
    auto be_name = ctx->op_kernel().type_string().c_str();
    if (strstr(be_name, tfbe::OpNamePrefix))
    {
        be_name = be_name + strlen(tfbe::OpNamePrefix);
    }
    auto op_def = tfbe::lookupOpDef(tfbe::getOpLibs(), be_name);
    size_t argsSize = ctx->num_inputs() + ctx->num_outputs();
    tfbe::OpCallStack stack;
    stack.tensors = new tfbe::Tensor[argsSize];
    stack.size = argsSize;
    for (size_t i = 0; i < ctx->num_inputs(); ++i)
    {
        auto tensor = TfTensorToBeTensor(ctx->input(i));
        stack.tensors[i] = tensor;
    }

    tfbe::CallFrame(op_def, stack);

    // we need keep the output alive
    // TODO use alive analysis reduce the memory peak
    for (size_t i = 0; i < ctx->num_outputs(); ++i)
    {
        stack.tensors[ctx->num_inputs() + i].getImpl()->increase_ref();
    }

    for (size_t i = 0; i < ctx->num_outputs(); ++i)
    {
        ctx->set_output(i, BeTensorToTfTensor(stack.tensors[ctx->num_inputs() + i]));
    }
    delete[] stack.tensors;
}

// REGISTER_KERNEL_BUILDER(Name("BackendOp").Device(DEVICE_GPU), BackendOp);
// REGISTER_KERNEL_BUILDER(Name("AddV2BackendOp").Device(DEVICE_CPU), BackendOp);
// REGISTER_KERNEL_BUILDER(Name("NegBackendOp").Device(DEVICE_CPU), BackendOp);
REGISTER_KERNEL_BUILDER(Name("H2DOp").Device(DEVICE_CPU), H2DOp);
REGISTER_KERNEL_BUILDER(Name("D2HOp").Device(DEVICE_CPU), D2HOp);

class BackendOpImplResistry
{
public:
    BackendOpImplResistry()
    {
        for (const auto& it : tfbe::OpLibs::instance().libs())
        {
            string name = tfbe::OpNamePrefix + it->name();
            auto& builder = ::tensorflow::KernelDefBuilder(name.c_str()).Device(DEVICE_CPU);
            new ::tensorflow::kernel_factory::OpKernelRegistrar(
                register_kernel::Name(name.c_str()).Device(DEVICE_CPU).Build(), "BackendOp",
                [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel*
                { return new BackendOp(context); });
        }
    }
};

static BackendOpImplResistry __registry;

} // namespace tensorflow
