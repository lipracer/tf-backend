option(ENABLE_REG_CODEGEN "Enable codegen the op registry" OFF)

if(ENABLE_REG_CODEGEN)
    message("Enable codegen the op registry")
else()
    message("Disabled codegen the op registry")
endif()