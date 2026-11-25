/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/Config.h>

///--------------------------------------
/// MARK: - String Macros

// Concatenation
#define IGL_CONCAT(a, b) _IGL_CONCAT_ARGS_WRAPPER(a, b)
#define _IGL_CONCAT_ARGS_WRAPPER(a, b) _IGL_CONCAT_ARGS(a, b)
#define _IGL_CONCAT_ARGS(a, b) a##b

#define IGL_CONCAT3(a, b, c) IGL_CONCAT(IGL_CONCAT(a, b), c)

// Convert Macro to String Literal
#define IGL_TO_STRING(a) _IGL_TO_STRING_WRAPPER(a)
#define _IGL_TO_STRING_WRAPPER(a) #a

///--------------------------------------
/// MARK: - Macro Helpers

#define _IGL_EXPAND(x) x
#define _IGL_ECHO(...) __VA_ARGS__
#define _IGL_CALL(macro, args) macro args

// Returns the 64th argument.
// clang-format off
#define _IGL_ARG_64(_,\
   _64,_63,_62,_61,_60,_59,_58,_57,_56,_55,_54,_53,_52,_51,_50,_49, \
   _48,_47,_46,_45,_44,_43,_42,_41,_40,_39,_38,_37,_36,_35,_34,_33, \
   _32,_31,_30,_29,_28,_27,_26,_25,_24,_23,_22,_21,_20,_19,_18,_17, \
   _16,_15,_14,_13,_12,_11,_10,_9,_8,_7,_6,_5,_4,_3,_2,ARG,...) ARG
// clang-format on

// Returns 0 if __VA_ARGS__ has a comma, 1 otherwise
// Does NOT handle the case when __VA_ARGS__ is empty
// Handles up to 64 arguments
// clang-format off
#define _IGL_HAS_COMMA(...) _IGL_EXPAND(_IGL_ARG_64(__VA_ARGS__, \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0))
// clang-format on

///--------------------------------------
/// MARK: - Linkage

#if defined(__cplusplus)
#define IGL_EXTERN extern "C"
#define IGL_EXTERN_BEGIN extern "C" {
#define IGL_EXTERN_END } // extern "C"
#else
#define IGL_EXTERN extern
#define IGL_EXTERN_BEGIN
#define IGL_EXTERN_END
#endif

// Visibility
#if defined(_MSC_VER)
#define _IGL_PUBLIC_ATTRIBUTE __declspec(dllexport)
#elif defined(__clang__) || defined(__GNUC__)
#define _IGL_PUBLIC_ATTRIBUTE __attribute__((visibility("default")))
#else
#define _IGL_PUBLIC_ATTRIBUTE
#endif

#define IGL_EXPORT _IGL_PUBLIC_ATTRIBUTE

// C API
#define IGL_API IGL_EXTERN _IGL_PUBLIC_ATTRIBUTE

///--------------------------------------
/// MARK: - Restrict

#if defined(__GNUC__) || defined(__clang__)
#define IGL_RESTRICT __restrict__
#elif defined(_MSC_VER) && (_MSC_VER >= 1900) // VS2015
#define IGL_RESTRICT __restrict
#else
#define IGL_RESTRICT
#endif

///--------------------------------------
/// MARK: - Inline

#if !defined(IGL_INLINE)
#define IGL_INLINE inline
#endif

///--------------------------------------
/// MARK: - Function Signature

#if defined(__GNUC__) || defined(__clang__)
#define IGL_FUNCTION __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define IGL_FUNCTION __FUNCSIG__
#elif defined(__cplusplus) && (__cplusplus >= 201103L) // C++11
#define IGL_FUNCTION __func__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) // C99
#define IGL_FUNCTION __func__
#else
#define IGL_FUNCTION "<function> " __FILE__ "(" IGL_MACRO_TO_STRING(__LINE__) ")"
#endif

///--------------------------------------
/// MARK: - Newline

#if IGL_PLATFORM_WINDOWS
#define IGL_NEWLINE "\r\n"
#else
#define IGL_NEWLINE "\n"
#endif

///--------------------------------------
/// MARK: Visual Studio compatibility

// not all control paths return a value
#if defined(_MSC_VER) || defined(__GNUC__)
#define IGL_UNREACHABLE_RETURN(value) \
  IGL_DEBUG_ASSERT_NOT_REACHED();     \
  return value;
#else
#define IGL_UNREACHABLE_RETURN(value)
#endif // _MSC_VER

///--------------------------------------
/// MARK: Integrated profiling

/// Do not use Tracy for GPU profiling if IGL_WITH_TRACY isn't defined
#if defined(IGL_WITH_TRACY_GPU) && !defined(IGL_WITH_TRACY)
#undef IGL_WITH_TRACY_GPU
#endif

#if defined(IGL_WITH_TRACY) && defined(__cplusplus)
#include "tracy/Tracy.hpp"
// predefined 0xRGB colors for "heavy" point-of-interest operations
#define IGL_PROFILER_COLOR_WAIT 0xff0000
#define IGL_PROFILER_COLOR_SUBMIT 0x0000ff
#define IGL_PROFILER_COLOR_PRESENT 0x00ff00
#define IGL_PROFILER_COLOR_CREATE 0xff6600
#define IGL_PROFILER_COLOR_DESTROY 0xffa500
#define IGL_PROFILER_COLOR_TRANSITION 0xffffff
#define IGL_PROFILER_COLOR_UPDATE 0xffa500
#define IGL_PROFILER_COLOR_DRAW 0x00ff00

// GPU profiling macros
#if defined(IGL_WITH_TRACY_GPU)
#define IGL_PROFILER_ZONE_GPU_OGL(name) TracyGpuZone(name);
#define IGL_PROFILER_ZONE_GPU_COLOR_OGL(name, color) TracyGpuZoneC(name, color);

#define IGL_PROFILER_ZONE_GPU_VK(name, profilingContext, cmdBuffer) \
  TracyVkZone(profilingContext, cmdBuffer, name);
#define IGL_PROFILER_ZONE_GPU_COLOR_VK(name, profilingContext, cmdBuffer, color) \
  TracyVkZoneC(profilingContext, cmdBuffer, name, color);

#define IGL_PROFILER_ZONE_TRANSIENT_GPU_OGL(varname, name) \
  {                                                        \
    TracyGpuZoneTransientS(varname, name, 0, true);
#define IGL_PROFILER_ZONE_TRANSIENT_GPU_VK(profilingContext, varname, cmdBuffer, name) \
  {                                                                                    \
    TracyVkZoneTransientS(profilingContext, varname, cmdBuffer, name, 0, true);
#define IGL_PROFILER_ZONE_GPU_END() }
#else // IGL_WITH_TRACY_GPU

#define IGL_PROFILER_ZONE_GPU_OGL(name)
#define IGL_PROFILER_ZONE_GPU_COLOR_OGL(name, color)

#define IGL_PROFILER_ZONE_GPU_VK(name, profilingContext, cmdBuffer)
#define IGL_PROFILER_ZONE_GPU_COLOR_VK(name, profilingContext, cmdBuffer, color)

#define IGL_PROFILER_ZONE_TRANSIENT_GPU_OGL(varname, name)
#define IGL_PROFILER_ZONE_TRANSIENT_GPU_VK(profilingContext, varname, cmdBuffer, name)
#define IGL_PROFILER_ZONE_GPU_END()

#endif // IGL_WITH_TRACY_GPU

#define IGL_PROFILER_FUNCTION() ZoneScoped
#define IGL_PROFILER_FUNCTION_COLOR(color) ZoneScopedC(color)
#define IGL_PROFILER_ZONE(name, color) \
  {                                    \
    ZoneScopedC(color);                \
    ZoneName(name, strlen(name))
#define IGL_PROFILER_ZONE_END() }
#define IGL_PROFILER_THREAD(name) tracy::SetThreadName(name)
#define IGL_PROFILER_FRAME(name) FrameMarkNamed(name)

#elif defined(IGL_WITH_PERFETTO) && defined(__cplusplus)
// Perfetto backend for standalone Android capture (barebone perfetto CLI).
// On Android: uses ATrace NDK — no category registration needed in Macros.h itself,
//   events appear in Perfetto via the ftrace/atrace data source automatically.
// On non-Android (Linux/Mac/Windows): no-ops so cross-platform builds are unaffected.
// TUs needing Perfetto SDK counter/instant tracks include IglPerfetto.h directly.

#define IGL_PROFILER_COLOR_WAIT 0xff0000
#define IGL_PROFILER_COLOR_SUBMIT 0x0000ff
#define IGL_PROFILER_COLOR_PRESENT 0x00ff00
#define IGL_PROFILER_COLOR_CREATE 0xff6600
#define IGL_PROFILER_COLOR_DESTROY 0xffa500
#define IGL_PROFILER_COLOR_TRANSITION 0xffffff
#define IGL_PROFILER_COLOR_UPDATE 0xffa500
#define IGL_PROFILER_COLOR_DRAW 0x00ff00

#define IGL_PROFILER_ZONE_GPU_OGL(name)
#define IGL_PROFILER_ZONE_GPU_COLOR_OGL(name, color)
#define IGL_PROFILER_ZONE_GPU_VK(name, profilingContext, cmdBuffer)
#define IGL_PROFILER_ZONE_GPU_COLOR_VK(name, profilingContext, cmdBuffer, color)
#define IGL_PROFILER_ZONE_TRANSIENT_GPU_OGL(varname, name)
#define IGL_PROFILER_ZONE_TRANSIENT_GPU_VK(profilingContext, varname, cmdBuffer, name)
#define IGL_PROFILER_ZONE_GPU_END()

#if defined(__ANDROID__)
#include <android/trace.h>

// Token-paste helpers for unique RAII variable names
#define IGL_DETAIL_CONCAT_(a, b) a##b
#define IGL_DETAIL_CONCAT(a, b) IGL_DETAIL_CONCAT_(a, b)

// RAII ATrace scope guard — inline, no extra dependencies
namespace igl::detail {
struct ATraceGuard {
  explicit ATraceGuard(const char* name) noexcept {
    ATrace_beginSection(name);
  }
  ~ATraceGuard() noexcept {
    ATrace_endSection();
  }
  ATraceGuard(const ATraceGuard&) = delete;
  ATraceGuard& operator=(const ATraceGuard&) = delete;
};
} // namespace igl::detail

#define IGL_PROFILER_FUNCTION() \
  ::igl::detail::ATraceGuard IGL_DETAIL_CONCAT(igl_atrace_, __LINE__)(__PRETTY_FUNCTION__)
#define IGL_PROFILER_FUNCTION_COLOR(color) IGL_PROFILER_FUNCTION()
#define IGL_PROFILER_ZONE(name, color) \
  {                                    \
    ::igl::detail::ATraceGuard IGL_DETAIL_CONCAT(igl_atrace_zone_, __LINE__)((name));
#define IGL_PROFILER_ZONE_END() }
#define IGL_PROFILER_THREAD(name)
#define IGL_PROFILER_FRAME(name) \
  do {                           \
    ATrace_beginSection(name);   \
    ATrace_endSection();         \
  } while (0)

#else // !__ANDROID__ — no-ops for Linux/Mac/Windows builds

#define IGL_PROFILER_FUNCTION()
#define IGL_PROFILER_FUNCTION_COLOR(color)
#define IGL_PROFILER_ZONE(name, color) {
#define IGL_PROFILER_ZONE_END() }
#define IGL_PROFILER_THREAD(name)
#define IGL_PROFILER_FRAME(name)

#endif // __ANDROID__
#else
#define IGL_PROFILER_ZONE_GPU_OGL(name)
#define IGL_PROFILER_ZONE_GPU_COLOR_OGL(name, color)

#define IGL_PROFILER_ZONE_GPU_VK(name, profilingContext, cmdBuffer)
#define IGL_PROFILER_ZONE_GPU_COLOR_VK(name, profilingContext, cmdBuffer, color)

#define IGL_PROFILER_ZONE_TRANSIENT_GPU_OGL(varname, name)
#define IGL_PROFILER_ZONE_TRANSIENT_GPU_VK(profilingContext, varname, cmdBuffer, name)
#define IGL_PROFILER_ZONE_GPU_END()

#define IGL_PROFILER_FUNCTION()
#define IGL_PROFILER_FUNCTION_COLOR(color)
#define IGL_PROFILER_ZONE(name, color) {
#define IGL_PROFILER_ZONE_END() }
#define IGL_PROFILER_THREAD(name)
#define IGL_PROFILER_FRAME(name)
#endif // IGL_WITH_TRACY

#define IGL_ENUM_TO_STRING(enum, res) \
  case enum ::res:                    \
    return IGL_TO_STRING(res);

// Define this to 1 to enable shader dumping. Currently only the Vulkan Device supports it.
// It will dump the SPIR-V code into files in the specified path below in
// Device::createShaderModule(...)
#define IGL_SHADER_DUMP 0

// Replace IGL_SHADER_DUMP_PATH with your own path according to the platform
// Ex. for Android your filepath should be specific to the package name:
// "/sdcard/Android/data/<packageName>/files/"
// @fb-only An example path used with the CodecAvatars app:
// @fb-only "/sdcard/Android/data/com.meta.ar.codecavatars/files/"
#define IGL_SHADER_DUMP_PATH "/path/to/output/file/"
