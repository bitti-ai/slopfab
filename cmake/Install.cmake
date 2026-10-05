# --- install ----------------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS slopfab RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Both the executable and shared library can find optional runtime libraries in
# an extracted package. Do not embed the build machine's library search paths.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(RELATIVE_PATH SLOPFAB_INSTALL_BIN_TO_LIB
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
  set_property(TARGET slopfab PROPERTY INSTALL_RPATH
    "$ORIGIN/${SLOPFAB_INSTALL_BIN_TO_LIB}")
endif()

# The shared library installs one header rather than the whole include tree: capi.h is the
# supported surface and the C++ headers beside it are internal implementation.
if(SLOPFAB_BUILD_C_API)
  install(TARGETS slopfab_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES include/slopfab/capi.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slopfab)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(TARGET slopfab_c PROPERTY INSTALL_RPATH "$ORIGIN")
  endif()

  # CUDA and cuBLAS DLLs are intentionally not installed. slopfab.dll discovers
  # the selected installed toolkit itself; consumers do not modify PATH.
endif()

install(FILES README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY docs/ DESTINATION ${CMAKE_INSTALL_DOCDIR}/docs
  FILES_MATCHING PATTERN "*.md")
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/slopfab)
foreach(dependency IN ITEMS sageattention motioncache vulkan seedvr2)
  if(dependency STREQUAL "vulkan")
    set(license "third_party/${dependency}/LICENSE.md")
  else()
    set(license "third_party/${dependency}/LICENSE")
  endif()
  install(FILES "${license}" DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/slopfab
    RENAME "${dependency}-LICENSE.txt")
endforeach()
install(FILES third_party/seedvr2/NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/slopfab
  RENAME seedvr2-NOTICE.txt)
if(SLOPFAB_WITH_FFMPEG)
  install(FILES external/ffmpeg/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/slopfab RENAME ffmpeg-LICENSE.txt)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(SLOPFAB_ENABLE_CUDA AND SLOPFAB_ENABLE_VULKAN)
    set(SLOPFAB_PACKAGE_BACKEND cuda-vulkan)
  elseif(SLOPFAB_ENABLE_CUDA)
    set(SLOPFAB_PACKAGE_BACKEND cuda)
  elseif(SLOPFAB_ENABLE_VULKAN)
    set(SLOPFAB_PACKAGE_BACKEND vulkan)
  else()
    set(SLOPFAB_PACKAGE_BACKEND cpu)
  endif()
  set(CPACK_GENERATOR TGZ)
  set(CPACK_PACKAGE_NAME slopfab)
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native video and audio generation")
  set(CPACK_PACKAGE_FILE_NAME
    "slopfab-${PROJECT_VERSION}-linux-${CMAKE_SYSTEM_PROCESSOR}-${SLOPFAB_PACKAGE_BACKEND}")
  set(CPACK_PACKAGING_INSTALL_PREFIX "/")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  include(CPack)
endif()
