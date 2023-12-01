set(SHELL_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/../script/build_llvm.sh")
# Call the shell script using execute_process
execute_process(
    COMMAND bash ${SHELL_SCRIPT} ${CMAKE_CURRENT_LIST_DIR}
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    RESULT_VARIABLE SCRIPT_RESULT
    OUTPUT_VARIABLE SCRIPT_OUTPUT
    ERROR_VARIABLE SCRIPT_ERROR
)

# for debug
# Display the output and error messages
# message("Script result: ${SCRIPT_RESULT}")
# message("Script output: ${SCRIPT_OUTPUT}")
# message("Script error: ${SCRIPT_ERROR}")
