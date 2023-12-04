
#include "capi/backend_api.h"
#include "op_registry.h"

//============================================================//
// Export functions
//============================================================//

void TfbeCallFrame(TfbeOpDef_t op, TfbeOpCallStack stack)
{
    auto op_def = reinterpret_cast<tfbe::OpDef*>(op.data);
    op_def->get_boxed_func()(stack);
}

TfbeOpDef_t TfbeLookupOpDef(TfbeOpLibs_t libs, const char* name)
{
    TfbeOpDef_t result = TfbeOpDef_t{.data = nullptr};
    if (!name)
    {
        return result;
    }
    result.data = reinterpret_cast<tfbe::OpLibs*>(libs.data)->lookup(name);
    return result;
}

TfbeOpLibs_t TfbeGetOpLibs()
{
    TfbeOpLibs_t result;
    result.data = &tfbe::OpLibs::instance();
    return result;
}

void TfbeForeachOpLibs(TfbeOpLibs_t libs, ForeachOpdefCB cb)
{
    for (const auto& lib : reinterpret_cast<tfbe::OpLibs*>(libs.data)->libs())
    {
        TfbeOpDef_t def;
        def.data = const_cast<tfbe::OpDef*>(lib.get());
        cb(def);
    }
}

tfbe::StringRef TfbeGetOpDefName(TfbeOpDef_t op_def)
{
    return reinterpret_cast<tfbe::OpDef*>(op_def.data)->name();
}

TfbeStringList* TfbeGetOpDefInputs(TfbeOpDef_t op_def)
{
    TfbeStringList* result = nullptr;
    auto inputs = reinterpret_cast<tfbe::OpDef*>(op_def.data)->inputs();
    result = reinterpret_cast<TfbeStringList*>(malloc(sizeof(result->size) + inputs.size() * sizeof(tfbe::StringRef)));
    result->size = inputs.size();
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        result->strs[i] = inputs[i].c_str();
    }
    return result;
}

TfbeStringList* TfbeGetOpDefOutputs(TfbeOpDef_t op_def)
{
    TfbeStringList* result = nullptr;
    auto outputs = reinterpret_cast<tfbe::OpDef*>(op_def.data)->outputs();
    result = reinterpret_cast<TfbeStringList*>(malloc(sizeof(result->size) + outputs.size() * sizeof(tfbe::StringRef)));
    result->size = outputs.size();
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        result->strs[i] = outputs[i].c_str();
    }
    return result;
}

TfbeStringList* TfbeGetOpDefAttributes(TfbeOpDef_t op_def)
{
    TfbeStringList* result = nullptr;
    auto attrs = reinterpret_cast<tfbe::OpDef*>(op_def.data)->attrs();
    result = reinterpret_cast<TfbeStringList*>(malloc(sizeof(result->size) + attrs.size() * sizeof(tfbe::StringRef)));
    result->size = attrs.size();
    for (size_t i = 0; i < attrs.size(); ++i)
    {
        result->strs[i] = attrs[i].c_str();
    }
    return result;
}

void FreeTfbeStringList(TfbeStringList* list)
{
    free(list);
}
