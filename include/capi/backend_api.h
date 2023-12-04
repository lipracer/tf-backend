#pragma once
#include <type_traits>

#include "../adt/StringRef.h"
#include "../adt/tensor.h"
#include "../macro.h"
#include "stddef.h"

constexpr const char TfbeOpNamePrefix[] = "BackendOp";

static_assert(std::is_standard_layout<tfbe::Tensor>::value, "export Tensor must standard layout!");
static_assert(std::is_standard_layout<tfbe::StringRef>::value, "export StringRef must standard layout!");

#ifdef __cplusplus
extern "C"
{
#endif

    struct TfbeOpDef_t
    {
        void* data;
    };

    struct TfbeOpLibs_t
    {
        void* data;
    };

    struct TfbeOpCallStack
    {
        tfbe::Tensor* tensors;
        size_t size;
    };

    struct TfbeStringList
    {
        size_t size;
        tfbe::StringRef strs[0];
    };

    /* *
     * @brief
     * @name
     */
    BE_EXPORT TfbeOpDef_t TfbeLookupOpDef(TfbeOpLibs_t libs, const char* name);

    // BE_EXPORT CompilerContext* getCompilerContext();

    BE_EXPORT TfbeOpLibs_t TfbeGetOpLibs();

    BE_EXPORT void TfbeCallFrame(TfbeOpDef_t op_def, TfbeOpCallStack stack);

    typedef void (*ForeachOpdefCB)(TfbeOpDef_t);
    BE_EXPORT void TfbeForeachOpLibs(TfbeOpLibs_t libs, ForeachOpdefCB cb);

    BE_EXPORT tfbe::StringRef TfbeGetOpDefName(TfbeOpDef_t op_def);

    typedef const char* ConstCharPtr;

    BE_EXPORT TfbeStringList* TfbeGetOpDefInputs(TfbeOpDef_t op_def);

    BE_EXPORT TfbeStringList* TfbeGetOpDefOutputs(TfbeOpDef_t op_def);

    BE_EXPORT TfbeStringList* TfbeGetOpDefAttributes(TfbeOpDef_t op_def);

    BE_EXPORT void FreeTfbeStringList(TfbeStringList* list);

#ifdef __cplusplus
}
#endif
