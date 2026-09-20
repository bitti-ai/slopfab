# --- install ----------------------------------------------------------------

# The DLL installs one header rather than the whole include tree: capi.h is the
# supported surface and the C++ headers beside it are internal implementation.
if(SLOPFAB_BUILD_C_API)
  include(GNUInstallDirs)
  install(TARGETS slopfab_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES include/slopfab/capi.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slopfab)

  # CUDA and cuBLAS DLLs are intentionally not installed. slopfab.dll discovers
  # the selected installed toolkit itself; consumers do not modify PATH.
endif()
