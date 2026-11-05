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
install(DIRECTORY examples/workflows examples/mods examples/motions DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/slopfab)
foreach(dependency IN ITEMS sageattention motioncache vulkan seedvr2 quickjs thorvg whisper)
  if(dependency STREQUAL "vulkan")
    set(license "third_party/${dependency}/LICENSE.md")
  else()
    set(license "third_party/${dependency}/LICENSE")
  endif()
  install(FILES "${license}" DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/slopfab
    RENAME "${dependency}-LICENSE.txt")
endforeach()
install(FILES third_party/thorvg/LODEPNG-LICENSE.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/slopfab)
install(FILES third_party/seedvr2/NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/slopfab
  RENAME seedvr2-NOTICE.txt)


# Release archives are assembled from both platform builds by tools/package_bundle.py.
# cmake --install remains available for native development installations; do not
# generate a single-platform CPack release that cannot provision the other OS.
