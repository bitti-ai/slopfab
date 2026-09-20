# --- c api ------------------------------------------------------------------

# slopfab_c builds slopfab.dll, the stable C ABI with a hand-written export set, for
# Rust, C#, Python and anything else with an FFI. See include/slopfab/capi.h.
#
# Dependent on CUDA rather than merely checked against it, because the C API
# exists to run generations and `run_generate` is on the CUDA side. Making it a
# plain option defaulting to ON meant that a machine with no toolkit — the case
# the lazy-CUDA design at the top of this file exists to support — stopped
# configuring at all, since the default would then contradict a hard error two
# hundred lines earlier. This forces it off in that situation instead, and says
# so, and an explicit -DSLOPFAB_BUILD_C_API=ON cannot resurrect it.
include(CMakeDependentOption)
cmake_dependent_option(SLOPFAB_BUILD_C_API
  "Build slopfab_c, the stable C ABI shared library" ON
  "SLOPFAB_ENABLE_CUDA" OFF)

if(NOT SLOPFAB_BUILD_C_API AND NOT SLOPFAB_ENABLE_CUDA)
  message(STATUS "slopfab: no CUDA, so slopfab_c (the C ABI DLL) is not built")
endif()

if(SLOPFAB_BUILD_C_API)
  # The internal core and CUDA libraries are static and link into this target.
  # Consequently slopfab.dll is the only project DLL, and its exports are exactly
  # the C entry points — no unstable C++ ABI and no companion slopfab DLLs.
  #
  # Static cudart is embedded (see above), while the shared in-process loader
  # resolves CUDA 13/12 cuBLAS by absolute toolkit path on first use. Thus the
  # DLL and CLI have identical selection/error behavior and neither has a
  # fixed cublas64_<major>.dll import.
  add_library(slopfab_c SHARED src/capi/capi.cpp)
  # slopfab_cuda carries slopfab_core with it as a PUBLIC dependency, so naming
  # the one target is naming both.
  target_link_libraries(slopfab_c PRIVATE slopfab_cuda)

  # PRIVATE: consumers of the DLL include capi.h and nothing else, and the
  # install rule below places it for them. Anything in this build tree that
  # needs the header asks for the include directory itself.
  target_include_directories(slopfab_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Switches capi.h from dllimport to dllexport. It is the only definition the
  # C API needs: the version it reports is stated in the header next to the
  # constants a consumer compiles against, so the two cannot drift.
  target_compile_definitions(slopfab_c PRIVATE SLOPFAB_C_BUILD=1)

  # The static libraries end up inside a shared object here, which on ELF
  # targets requires every object in them to be position independent. Harmless
  # and ignored on Windows.
  set_property(TARGET slopfab_core PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET slopfab_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)

  # The export set is the point of this target, so it is stated rather than
  # inherited. WINDOWS_EXPORT_ALL_SYMBOLS is off explicitly because a build
  # that sets CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS globally would otherwise dump
  # the whole C++ surface into this DLL's export table alongside the C one, and
  # the resulting module would still work — silently defeating the boundary
  # instead of failing.
  #
  # The version is the *C ABI's*, not the project's. They are different numbers
  # answering different questions — capi.h declares its own version and tells
  # a binding to refuse a different major, while PROJECT_VERSION is 0.1.0 and
  # moves whenever anything in the project does. Using the latter here would
  # produce a libslopfab.so.0 whose soname says 0 for an ABI that says 1, and
  # would bump it for releases that did not touch the C interface at all.
  set(SLOPFAB_CAPI_VERSION_MAJOR 1)
  set(SLOPFAB_CAPI_VERSION_MINOR 11)
  set(SLOPFAB_CAPI_VERSION_PATCH 0)
  set(SLOPFAB_CAPI_VERSION
    "${SLOPFAB_CAPI_VERSION_MAJOR}.${SLOPFAB_CAPI_VERSION_MINOR}.${SLOPFAB_CAPI_VERSION_PATCH}")

  set_target_properties(slopfab_c PROPERTIES
    OUTPUT_NAME slopfab
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${SLOPFAB_CAPI_VERSION}
    SOVERSION ${SLOPFAB_CAPI_VERSION_MAJOR})
  if(WIN32)
    # The DLL and CLI intentionally share the runtime basename, but an MSVC
    # executable with exported symbols can also emit slopfab.lib. Give the DLL
    # import archive its own name so parallel aggregate builds cannot overwrite
    # the library C-API tests are about to link.
    set_target_properties(slopfab_c PROPERTIES ARCHIVE_OUTPUT_NAME slopfab_c)
  endif()

  # The visibility presets above are *compile* properties, so they only cover
  # capi.cpp. slopfab_core and slopfab_cuda are compiled with default visibility,
  # and on ELF their entire C++ symbol table would otherwise be re-exported
  # from libslopfab.so — precisely what WINDOWS_EXPORT_ALL_SYMBOLS OFF
  # prevents on the other platform. --exclude-libs makes the two agree.
  if(NOT WIN32 AND NOT APPLE)
    target_link_options(slopfab_c PRIVATE "LINKER:--exclude-libs,ALL")
  endif()

  # capi.h states the same three numbers for the compiler; if they ever
  # disagree, the soname and the number a binding checks would describe
  # different interfaces. Cheaper to catch here than in a consumer.
  #
  # Matched per name rather than by position. Reading the three numbers in file
  # order and joining them would compare 0.1.0 against 1.0.0 — and abort with a
  # message blaming the version rather than the reordering — if someone ever
  # rearranged three `#define`s that C is perfectly happy to see in any order.
  foreach(SLOPFAB_CAPI_PART MAJOR MINOR PATCH)
    unset(SLOPFAB_CAPI_HEADER_${SLOPFAB_CAPI_PART})
    file(STRINGS include/slopfab/capi.h SLOPFAB_CAPI_HEADER_LINE
      REGEX "^#define[ \t]+SLOPFAB_CAPI_VERSION_${SLOPFAB_CAPI_PART}[ \t]+[0-9]+")
    if(NOT SLOPFAB_CAPI_HEADER_LINE)
      message(FATAL_ERROR
        "slopfab: capi.h has no SLOPFAB_CAPI_VERSION_${SLOPFAB_CAPI_PART} to check against")
    endif()
    string(REGEX MATCH "[0-9]+$" SLOPFAB_CAPI_HEADER_${SLOPFAB_CAPI_PART}
      "${SLOPFAB_CAPI_HEADER_LINE}")
  endforeach()
  set(SLOPFAB_CAPI_HEADER_VERSION
    "${SLOPFAB_CAPI_HEADER_MAJOR}.${SLOPFAB_CAPI_HEADER_MINOR}.${SLOPFAB_CAPI_HEADER_PATCH}")
  if(NOT SLOPFAB_CAPI_HEADER_VERSION STREQUAL SLOPFAB_CAPI_VERSION)
    message(FATAL_ERROR
      "slopfab: capi.h declares ABI ${SLOPFAB_CAPI_HEADER_VERSION} but CMakeLists says "
      "${SLOPFAB_CAPI_VERSION}; update both.")
  endif()
  # `file(STRINGS)` creates no dependency of its own, so without this the check
  # would go stale at exactly the moment it exists for: editing the header's
  # version macros would not re-run CMake, and the soname would keep describing
  # the previous ABI.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/slopfab/capi.h")

  # `Tokenizer::load_embedded` looks the resource up in the module holding its
  # own code. That code is linked into slopfab.dll, so the 7 MB tokenizer has to
  # be here too or a C caller that leaves the tokenizer path empty gets
  # "embedded resource is missing".
  if(WIN32)
    target_sources(slopfab_c PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/slopfab_tokenizer.rc")
  endif()

  if(MSVC)
    target_compile_options(slopfab_c PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(slopfab_c PRIVATE -Wall -Wextra)
  endif()
endif()

