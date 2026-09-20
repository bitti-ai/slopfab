# Keep executable names stable while exposing independently selectable coverage.
# A suite with no usable hardware/fixtures returns 77, not a false green pass.
function(slopfab_test_suite name target)
  foreach(category IN ITEMS synthetic ${ARGN})
    if(category STREQUAL "benchmark" AND NOT SLOPFAB_TEST_BENCHMARKS)
      continue()
    endif()
    if(category STREQUAL "synthetic")
      set(test_name "${name}")
    else()
      set(test_name "${name}_${category}")
    endif()
    add_test(NAME "${test_name}" COMMAND "${target}")
    set_tests_properties("${test_name}" PROPERTIES
      ENVIRONMENT "SLOPFAB_TEST_CATEGORY=${category}"
      LABELS "${category};${name}"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK slopfab_gpu
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:${target}>")
    if(category STREQUAL "benchmark")
      set_property(TEST "${test_name}" APPEND PROPERTY
        ENVIRONMENT "SLOPFAB_RUN_BENCHMARKS=1")
    endif()
  endforeach()
endfunction()
