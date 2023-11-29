

#include <atomic>
#include <mutex>
#include <vector>

#include "adaptor.h"
#include "internal/support.h"
#include "logger/logger.h"

namespace tfbe
{

namespace
{
std::mutex gGlobalFuncMtx;
using FunctionList = std::vector<InitializationFunc>;
FunctionList* InitializationFs()
{
    std::once_flag once_flag;
    FunctionList* funcs = nullptr;
    std::call_once(once_flag, [&funcs]() { funcs = new FunctionList(); });
    return funcs;
};

FunctionList* ReleaseFs()
{
    std::once_flag once_flag;
    FunctionList* funcs = nullptr;
    std::call_once(once_flag, [&funcs]() { funcs = new FunctionList(); });
    return funcs;
};

} // namespace

void registeInitialization(const InitializationFunc& initialization)
{
    std::lock_guard<std::mutex> lg(gGlobalFuncMtx);
    InitializationFs()->push_back(initialization);
}

void registeRelease(const InitializationFunc& initialization)
{
    ReleaseFs()->push_back(initialization);
}

static void globale_constructor() __attribute__((constructor));
static void global_destructor() __attribute__((destructor));

void globale_constructor()
{
    LOG(INFO) << "load tf backend";
    (void*)InitializationFs();
}

void global_destructor()
{
#if 0
    auto& funcs = *ReleaseFs();
    for (auto& func : funcs)
    {
        func();
    }
#endif
    LOG(INFO) << "unload tf backend";
}

extern "C"
{

}
} // namespace tfbe
