option(ENABLE_REG_CODEGEN "Enable codegen the op registry" OFF)
option(ENABLE_BUILD_PLUGIN "rebuild plugin and update resource" OFF)

if(ENABLE_REG_CODEGEN)
    message("Enable codegen the op registry.")
else()
    message("Disabled codegen the op registry.")
endif()

if(ENABLE_BUILD_PLUGIN)
    message("Enable build plugin.")
else()
    message("Disabled build plugin.")
endif()