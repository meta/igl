/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/IContext.h>

#include <algorithm>
#include <cstring>
#if IGL_DEBUG
#include <mutex>
#endif
#include <string>
#include <igl/DeviceFeatures.h>
#include <igl/Macros.h>
#include <igl/opengl/GLFunc.h>
#include <igl/opengl/GLIncludes.h>

#if defined(IGL_WITH_TRACY_GPU)
#include "tracy/TracyOpenGL.hpp"
#endif

// Uncomment to enable shader validation
// #define IGL_VALIDATE_SHADERS

#if IGL_API_LOG

#include <sstream>

#define APILOG_DEC_DRAW_COUNT() \
  if (apiLogDrawsLeft_) {       \
    apiLogDrawsLeft_--;         \
  }
#define APILOG(format, ...)                    \
  if (apiLogDrawsLeft_ || apiLogEnabled_) {    \
    IGLLog(IGLLogInfo, format, ##__VA_ARGS__); \
  }
namespace {
uint32_t logSourceChunk(uint32_t line, const char* string, size_t length) {
  int printLen = 0;
  const char* lineString = string;
  while (length > 0) {
    if (*string == '\r' || *string == '\n') {
      if (printLen > 0) {
        IGLLog(IGLLogInfo, "%3u: %.*s\n", line++, printLen, lineString);
      }
      printLen = 0;
      lineString = string + 1;
    } else {
      printLen++;
    }
    ++string;
    --length;
  }
  if (printLen > 0) {
    IGLLog(IGLLogInfo, "%3u: %.*s\n", line++, printLen, lineString);
  }
  return line;
}

void logSource(const int count, const char** string, const int* length) {
  if (!string) {
    return;
  }
  uint32_t line = 1;
  for (int i = 0; i < count; ++i) {
    if (!string[i]) {
      continue;
    }
    line = logSourceChunk(
        line, string[i], !length ? std::strlen(string[i]) : static_cast<size_t>(length[i]));
  }
}
} // namespace
#define APILOG_SOURCE(count, string, length) logSource(count, string, length);

#else
#define APILOG_DEC_DRAW_COUNT() static_cast<void>(0)
#define APILOG(format, ...) static_cast<void>(0)
#define APILOG_SOURCE(count, string, length) static_cast<void>(0)
#endif // IGL_API_LOG

#define GLCALL(funcName)                                        \
  IGL_SOFT_ASSERT(isCurrentContext() || isCurrentSharegroup()); \
  callCounter_.fetch_add(1, std::memory_order_relaxed);         \
  gl##funcName

#define IGLCALL(funcName)                                       \
  IGL_SOFT_ASSERT(isCurrentContext() || isCurrentSharegroup()); \
  callCounter_.fetch_add(1, std::memory_order_relaxed);         \
  igl##funcName

#define GLCALL_WITH_RETURN(ret, funcName)                       \
  IGL_SOFT_ASSERT(isCurrentContext() || isCurrentSharegroup()); \
  callCounter_.fetch_add(1, std::memory_order_relaxed);         \
  ret = gl##funcName

#define IGLCALL_WITH_RETURN(ret, funcName)                      \
  IGL_SOFT_ASSERT(isCurrentContext() || isCurrentSharegroup()); \
  callCounter_.fetch_add(1, std::memory_order_relaxed);         \
  ret = igl##funcName

#define GLCALL_PROC(funcPtr, ...)                               \
  IGL_SOFT_ASSERT(isCurrentContext() || isCurrentSharegroup()); \
  if (IGL_DEBUG_VERIFY(funcPtr)) {                              \
    callCounter_.fetch_add(1, std::memory_order_relaxed);       \
    (*funcPtr)(__VA_ARGS__);                                    \
  }

#define GLCALL_PROC_WITH_RETURN(ret, funcPtr, returnOnError, ...) \
  IGL_SOFT_ASSERT(isCurrentContext() || isCurrentSharegroup());   \
  if (IGL_DEBUG_VERIFY(funcPtr)) {                                \
    callCounter_.fetch_add(1, std::memory_order_relaxed);         \
    ret = (*funcPtr)(__VA_ARGS__);                                \
  } else {                                                        \
    ret = returnOnError;                                          \
  }

#if IGL_DEBUG
#define GLCHECK_ERRORS()                      \
  do {                                        \
    if (alwaysCheckError_) {                  \
      checkForErrors(__FUNCTION__, __LINE__); \
    }                                         \
  } while (false)
#define GL_ASSERT_ERROR(condition, callerName, lineNum, errorCode) \
  IGL_DEBUG_ASSERT((condition),                                    \
                   "[IGL] OpenGL error [%s:%zu] 0x%04X: %s\n",     \
                   callerName,                                     \
                   lineNum,                                        \
                   errorCode,                                      \
                   GL_ERROR_TO_STRING(errorCode))
#else
#define GLCHECK_ERRORS() static_cast<void>(0)
#define GL_ASSERT_ERROR(condition, callerName, lineNum, errorCode) static_cast<void>(0)
#endif // IGL_DEBUG

#define RESULT_CASE(res) \
  case res:              \
    return #res;

#if IGL_API_LOG
  // Enclose this function with the #ifdef to stop the compiler
// from complaining that this function is unused (in the non debug/log path)
namespace {
std::string GLboolToString(GLboolean val) {
  return val ? "true" : "false";
}

std::string GLenumToString(GLenum code) {
  switch (code) {
    RESULT_CASE(GL_ACTIVE_ATTRIBUTES)
    RESULT_CASE(GL_ACTIVE_ATTRIBUTE_MAX_LENGTH)
    RESULT_CASE(GL_ACTIVE_RESOURCES)
    RESULT_CASE(GL_ACTIVE_TEXTURE)
    RESULT_CASE(GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH)
    RESULT_CASE(GL_ACTIVE_UNIFORM_BLOCKS)
    RESULT_CASE(GL_ACTIVE_UNIFORM_MAX_LENGTH)
    RESULT_CASE(GL_ACTIVE_UNIFORMS)
    RESULT_CASE(GL_ALIASED_LINE_WIDTH_RANGE)
    RESULT_CASE(GL_ALPHA_BITS)
    RESULT_CASE(GL_ALPHA8)
    RESULT_CASE(GL_ALWAYS)
    RESULT_CASE(GL_ARRAY_BUFFER)
    RESULT_CASE(GL_ARRAY_BUFFER_BINDING)
    RESULT_CASE(GL_ATTACHED_SHADERS)
    RESULT_CASE(GL_BACK)
    RESULT_CASE(GL_BGR)
    RESULT_CASE(GL_BGRA)
    RESULT_CASE(GL_BGRA8_EXT)
    RESULT_CASE(GL_BLEND)
    RESULT_CASE(GL_BLEND_COLOR)
    RESULT_CASE(GL_BLEND_DST_ALPHA)
    RESULT_CASE(GL_BLEND_EQUATION_RGB)
    RESULT_CASE(GL_BLEND_EQUATION_ALPHA)
    RESULT_CASE(GL_BLEND_SRC_ALPHA)
    RESULT_CASE(GL_BLEND_SRC_RGB)
    RESULT_CASE(GL_BLUE)
    RESULT_CASE(GL_BUFFER)
    RESULT_CASE(GL_BUFFER_BINDING)
    RESULT_CASE(GL_BUFFER_OBJECT_EXT)
    RESULT_CASE(GL_BUFFER_SIZE)
    RESULT_CASE(GL_BUFFER_USAGE)
    RESULT_CASE(GL_BYTE)
    RESULT_CASE(GL_CLAMP_TO_EDGE)
    RESULT_CASE(GL_COLOR)
    RESULT_CASE(GL_COLOR_ATTACHMENT0)
    RESULT_CASE(GL_COLOR_ATTACHMENT1)
    RESULT_CASE(GL_COMPARE_REF_TO_TEXTURE)
    RESULT_CASE(GL_COMPRESSED_R11_EAC)
    RESULT_CASE(GL_COMPRESSED_RG11_EAC)
    RESULT_CASE(GL_COMPRESSED_RGB8_ETC2)
    RESULT_CASE(GL_COMPRESSED_RGBA8_ETC2_EAC)
    RESULT_CASE(GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2)
    RESULT_CASE(GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG)
    RESULT_CASE(GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG)
    RESULT_CASE(GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG)
    RESULT_CASE(GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_10x10_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_10x6_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_10x8_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_10x5_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_12x10_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_12x12_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_4x4_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_5x4_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_5x5_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_6x5_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_6x6_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_8x5_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_8x6_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_ASTC_8x8_KHR)
    RESULT_CASE(GL_COMPRESSED_RGBA_BPTC_UNORM)
    RESULT_CASE(GL_COMPRESSED_SIGNED_R11_EAC)
    RESULT_CASE(GL_COMPRESSED_SIGNED_RG11_EAC)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ETC2)
    RESULT_CASE(GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC)
    RESULT_CASE(GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2)
    RESULT_CASE(GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM)
    RESULT_CASE(GL_COMPILE_STATUS)
    RESULT_CASE(GL_COMPUTE_SHADER)
    RESULT_CASE(GL_CONSTANT_ALPHA)
    RESULT_CASE(GL_CONSTANT_COLOR)
    RESULT_CASE(GL_COPY_READ_BUFFER)
    RESULT_CASE(GL_COPY_WRITE_BUFFER)
    RESULT_CASE(GL_CULL_FACE)
    RESULT_CASE(GL_CURRENT_PROGRAM)
    RESULT_CASE(GL_CW)
    RESULT_CASE(GL_CCW)
    RESULT_CASE(GL_CURRENT_VERTEX_ATTRIB)
    RESULT_CASE(GL_DEBUG_LOGGED_MESSAGES)
    RESULT_CASE(GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH)
    RESULT_CASE(GL_DEBUG_OUTPUT)
    RESULT_CASE(GL_DEBUG_SEVERITY_HIGH)
    RESULT_CASE(GL_DEBUG_SEVERITY_LOW)
    RESULT_CASE(GL_DEBUG_SEVERITY_MEDIUM)
    RESULT_CASE(GL_DEBUG_SEVERITY_NOTIFICATION)
    RESULT_CASE(GL_DEBUG_SOURCE_API)
    RESULT_CASE(GL_DEBUG_SOURCE_APPLICATION)
    RESULT_CASE(GL_DEBUG_SOURCE_OTHER)
    RESULT_CASE(GL_DEBUG_SOURCE_SHADER_COMPILER)
    RESULT_CASE(GL_DEBUG_SOURCE_THIRD_PARTY)
    RESULT_CASE(GL_DEBUG_SOURCE_WINDOW_SYSTEM)
    RESULT_CASE(GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR)
    RESULT_CASE(GL_DEBUG_TYPE_ERROR)
    RESULT_CASE(GL_DEBUG_TYPE_MARKER)
    RESULT_CASE(GL_DEBUG_TYPE_OTHER)
    RESULT_CASE(GL_DEBUG_TYPE_PERFORMANCE)
    RESULT_CASE(GL_DEBUG_TYPE_PORTABILITY)
    RESULT_CASE(GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR)
    RESULT_CASE(GL_DECR)
    RESULT_CASE(GL_DECR_WRAP)
    RESULT_CASE(GL_DELETE_STATUS)
    RESULT_CASE(GL_DEPTH)
    RESULT_CASE(GL_DEPTH_ATTACHMENT)
    RESULT_CASE(GL_DEPTH_COMPONENT)
    RESULT_CASE(GL_DEPTH_COMPONENT16)
    RESULT_CASE(GL_DEPTH_COMPONENT24)
    RESULT_CASE(GL_DEPTH_COMPONENT32)
    RESULT_CASE(GL_DEPTH_STENCIL)
    RESULT_CASE(GL_DEPTH24_STENCIL8)
    RESULT_CASE(GL_DEPTH32F_STENCIL8)
    RESULT_CASE(GL_DEPTH_TEST)
    RESULT_CASE(GL_DITHER)
    RESULT_CASE(GL_DONT_CARE)
    RESULT_CASE(GL_DRAW_FRAMEBUFFER)
    RESULT_CASE(GL_DRAW_INDIRECT_BUFFER)
    RESULT_CASE(GL_DST_ALPHA)
    RESULT_CASE(GL_DST_COLOR)
    RESULT_CASE(GL_DYNAMIC_COPY)
    RESULT_CASE(GL_DYNAMIC_DRAW)
    RESULT_CASE(GL_DYNAMIC_READ)
    RESULT_CASE(GL_ELEMENT_ARRAY_BUFFER)
    RESULT_CASE(GL_EQUAL)
    RESULT_CASE(GL_ETC1_RGB8_OES)
    RESULT_CASE(GL_EXTENSIONS)
    RESULT_CASE(GL_FASTEST)
    RESULT_CASE(GL_FILL)
    RESULT_CASE(GL_FLOAT)
    RESULT_CASE(GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
    RESULT_CASE(GL_FLOAT_MAT2)
    RESULT_CASE(GL_FLOAT_MAT3)
    RESULT_CASE(GL_FLOAT_MAT4)
    RESULT_CASE(GL_FLOAT_VEC2)
    RESULT_CASE(GL_FLOAT_VEC3)
    RESULT_CASE(GL_FLOAT_VEC4)
    RESULT_CASE(GL_FRAGMENT_SHADER)
    RESULT_CASE(GL_FRAGMENT_SHADER_DERIVATIVE_HINT)
    RESULT_CASE(GL_FRAMEBUFFER)
    RESULT_CASE(GL_FRAMEBUFFER_DEFAULT)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE)
    RESULT_CASE(GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL)
    RESULT_CASE(GL_FRAMEBUFFER_BINDING)
    RESULT_CASE(GL_FRAMEBUFFER_COMPLETE)
    RESULT_CASE(GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT)
    RESULT_CASE(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT)
    RESULT_CASE(GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS)
    RESULT_CASE(GL_FRAMEBUFFER_UNDEFINED)
    RESULT_CASE(GL_FRAMEBUFFER_UNSUPPORTED)
    RESULT_CASE(GL_FRAMEBUFFER_SRGB)
    RESULT_CASE(GL_FRONT)
    RESULT_CASE(GL_FRONT_AND_BACK)
    RESULT_CASE(GL_FUNC_ADD)
    RESULT_CASE(GL_FUNC_SUBTRACT)
    RESULT_CASE(GL_FUNC_REVERSE_SUBTRACT)
    RESULT_CASE(GL_KEEP)
    RESULT_CASE(GL_GENERATE_MIPMAP_HINT)
    RESULT_CASE(GL_GEQUAL)
    RESULT_CASE(GL_GREATER)
    RESULT_CASE(GL_GREEN)
    RESULT_CASE(GL_HALF_FLOAT)
    RESULT_CASE(GL_HALF_FLOAT_OES)
    RESULT_CASE(GL_HANDLE_TYPE_OPAQUE_FD_EXT)
    RESULT_CASE(GL_HIGH_FLOAT)
    RESULT_CASE(GL_HIGH_INT)
    RESULT_CASE(GL_IMAGE_1D)
    RESULT_CASE(GL_IMAGE_1D_ARRAY)
    RESULT_CASE(GL_IMAGE_2D)
    RESULT_CASE(GL_IMAGE_2D_MULTISAMPLE)
    RESULT_CASE(GL_IMAGE_2D_ARRAY)
    RESULT_CASE(GL_IMAGE_2D_MULTISAMPLE_ARRAY)
    RESULT_CASE(GL_IMAGE_3D)
    RESULT_CASE(GL_IMAGE_CUBE)
    RESULT_CASE(GL_INCR)
    RESULT_CASE(GL_INCR_WRAP)
    RESULT_CASE(GL_INFO_LOG_LENGTH)
    RESULT_CASE(GL_INT)
    RESULT_CASE(GL_INT_2_10_10_10_REV)
    RESULT_CASE(GL_INVERT)
    RESULT_CASE(GL_LESS)
    RESULT_CASE(GL_LEQUAL)
    RESULT_CASE(GL_LINE)
    RESULT_CASE(GL_LINE_STRIP)
    RESULT_CASE(GL_LINE_LOOP)
    RESULT_CASE(GL_LINEAR)
    RESULT_CASE(GL_LINEAR_MIPMAP_NEAREST)
    RESULT_CASE(GL_LINEAR_MIPMAP_LINEAR)
    RESULT_CASE(GL_LINES)
    RESULT_CASE(GL_LINK_STATUS)
    RESULT_CASE(GL_LOW_FLOAT)
    RESULT_CASE(GL_LOW_INT)
    RESULT_CASE(GL_LUMINANCE)
    RESULT_CASE(GL_LUMINANCE_ALPHA)
    RESULT_CASE(GL_LUMINANCE8)
    RESULT_CASE(GL_LUMINANCE8_ALPHA8)
    RESULT_CASE(GL_MAX)
    RESULT_CASE(GL_MAX_COMPUTE_UNIFORM_COMPONENTS)
    RESULT_CASE(GL_MAX_CUBE_MAP_TEXTURE_SIZE)
    RESULT_CASE(GL_MAX_DEBUG_GROUP_STACK_DEPTH)
    RESULT_CASE(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS)
    RESULT_CASE(GL_MAX_FRAGMENT_UNIFORM_VECTORS)
    RESULT_CASE(GL_MAX_NAME_LENGTH)
    RESULT_CASE(GL_MAX_SAMPLES)
    RESULT_CASE(GL_MAX_SAMPLES_IMG)
    RESULT_CASE(GL_MAX_TEXTURE_SIZE)
    RESULT_CASE(GL_MAX_UNIFORM_BLOCK_SIZE)
    RESULT_CASE(GL_MAX_VERTEX_ATTRIBS)
    RESULT_CASE(GL_MAX_VERTEX_UNIFORM_COMPONENTS)
    RESULT_CASE(GL_MAX_VERTEX_UNIFORM_VECTORS)
    RESULT_CASE(GL_MEDIUM_FLOAT)
    RESULT_CASE(GL_MEDIUM_INT)
    RESULT_CASE(GL_MIN)
    RESULT_CASE(GL_MIRRORED_REPEAT)
    RESULT_CASE(GL_NEAREST)
    RESULT_CASE(GL_NEAREST_MIPMAP_NEAREST)
    RESULT_CASE(GL_NEAREST_MIPMAP_LINEAR)
    RESULT_CASE(GL_NEVER)
    RESULT_CASE(GL_NICEST)
    RESULT_CASE(GL_NOTEQUAL)
    RESULT_CASE(GL_NUM_EXTENSIONS)
    RESULT_CASE(GL_ONE_MINUS_CONSTANT_ALPHA)
    RESULT_CASE(GL_ONE_MINUS_CONSTANT_COLOR)
    RESULT_CASE(GL_ONE_MINUS_DST_ALPHA)
    RESULT_CASE(GL_ONE_MINUS_DST_COLOR)
    RESULT_CASE(GL_ONE_MINUS_SRC_ALPHA)
    RESULT_CASE(GL_ONE_MINUS_SRC_COLOR)
    RESULT_CASE(GL_PACK_ALIGNMENT)
    RESULT_CASE(GL_PACK_ROW_LENGTH)
    RESULT_CASE(GL_PIXEL_PACK_BUFFER)
    RESULT_CASE(GL_PIXEL_UNPACK_BUFFER)
    RESULT_CASE(GL_POINTS)
    RESULT_CASE(GL_POLYGON_OFFSET_FILL)
    RESULT_CASE(GL_PROGRAM)
    RESULT_CASE(GL_PROGRAM_OBJECT_EXT)
    RESULT_CASE(GL_R16)
    RESULT_CASE(GL_R16F)
    RESULT_CASE(GL_R16UI)
    RESULT_CASE(GL_R8)
    RESULT_CASE(GL_R32F)
    RESULT_CASE(GL_READ_FRAMEBUFFER)
    RESULT_CASE(GL_READ_FRAMEBUFFER_BINDING)
    RESULT_CASE(GL_READ_ONLY)
    RESULT_CASE(GL_READ_WRITE)
    RESULT_CASE(GL_RED)
    RESULT_CASE(GL_RED_INTEGER)
    RESULT_CASE(GL_RENDERBUFFER)
    RESULT_CASE(GL_RENDERBUFFER_ALPHA_SIZE)
    RESULT_CASE(GL_RENDERBUFFER_BINDING)
    RESULT_CASE(GL_RENDERBUFFER_BLUE_SIZE)
    RESULT_CASE(GL_RENDERBUFFER_DEPTH_SIZE)
    RESULT_CASE(GL_RENDERBUFFER_GREEN_SIZE)
    RESULT_CASE(GL_RENDERBUFFER_HEIGHT)
    RESULT_CASE(GL_RENDERBUFFER_INTERNAL_FORMAT)
    RESULT_CASE(GL_RENDERBUFFER_RED_SIZE)
    RESULT_CASE(GL_RENDERBUFFER_STENCIL_SIZE)
    RESULT_CASE(GL_RENDERBUFFER_WIDTH)
    RESULT_CASE(GL_RENDERER)
    RESULT_CASE(GL_REPEAT)
    RESULT_CASE(GL_RG)
    RESULT_CASE(GL_RG_INTEGER)
    RESULT_CASE(GL_RG16)
    RESULT_CASE(GL_RG16F)
    RESULT_CASE(GL_RG16UI)
    RESULT_CASE(GL_RG8)
    RESULT_CASE(GL_RGB)
    RESULT_CASE(GL_RGB_422_APPLE)
    RESULT_CASE(GL_RGB_INTEGER)
    RESULT_CASE(GL_RGB_RAW_422_APPLE)
    RESULT_CASE(GL_RGB10_A2)
    RESULT_CASE(GL_RGB10_A2UI)
    RESULT_CASE(GL_RGB16F)
    RESULT_CASE(GL_RGB32F)
    RESULT_CASE(GL_RGB5_A1)
    RESULT_CASE(GL_RGB8)
    RESULT_CASE(GL_RGBA)
    RESULT_CASE(GL_RGBA_INTEGER)
    RESULT_CASE(GL_RGBA16F)
    RESULT_CASE(GL_RGBA32F)
    RESULT_CASE(GL_RGBA32UI)
    RESULT_CASE(GL_RGBA4)
    RESULT_CASE(GL_RGBA8)
    RESULT_CASE(GL_REPLACE)
    RESULT_CASE(GL_SAMPLE_ALPHA_TO_COVERAGE)
    RESULT_CASE(GL_SAMPLE_COVERAGE)
    RESULT_CASE(GL_SAMPLER_1D)
    RESULT_CASE(GL_SAMPLER_1D_ARRAY)
    RESULT_CASE(GL_SAMPLER_2D)
    RESULT_CASE(GL_SAMPLER_2D_ARRAY)
    RESULT_CASE(GL_SAMPLER_2D_MULTISAMPLE)
    RESULT_CASE(GL_SAMPLER_3D)
    RESULT_CASE(GL_SAMPLER_EXTERNAL_OES)
    RESULT_CASE(GL_SCISSOR_TEST)
    RESULT_CASE(GL_SHADER)
    RESULT_CASE(GL_SHADER_OBJECT_EXT)
    RESULT_CASE(GL_SHADER_SOURCE_LENGTH)
    RESULT_CASE(GL_SHADER_STORAGE_BLOCK)
    RESULT_CASE(GL_SHADER_STORAGE_BUFFER)
    RESULT_CASE(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT)
    RESULT_CASE(GL_SHADER_TYPE)
    RESULT_CASE(GL_SHADING_LANGUAGE_VERSION)
    RESULT_CASE(GL_SHORT)
    RESULT_CASE(GL_SIGNALED)
    RESULT_CASE(GL_SIGNED_NORMALIZED)
    RESULT_CASE(GL_SRGB)
    RESULT_CASE(GL_SRGB_ALPHA)
    RESULT_CASE(GL_SRGB8)
    RESULT_CASE(GL_SRGB8_ALPHA8)
    RESULT_CASE(GL_SRC_ALPHA)
    RESULT_CASE(GL_SRC_ALPHA_SATURATE)
    RESULT_CASE(GL_SRC_COLOR)
    RESULT_CASE(GL_STATIC_COPY)
    RESULT_CASE(GL_STATIC_DRAW)
    RESULT_CASE(GL_STATIC_READ)
    RESULT_CASE(GL_STENCIL)
    RESULT_CASE(GL_STENCIL_INDEX)
    RESULT_CASE(GL_STENCIL_INDEX8)
    RESULT_CASE(GL_STENCIL_TEST)
    RESULT_CASE(GL_STENCIL_ATTACHMENT)
    RESULT_CASE(GL_STREAM_COPY)
    RESULT_CASE(GL_STREAM_DRAW)
    RESULT_CASE(GL_STREAM_READ)
    RESULT_CASE(GL_SYNC_GPU_COMMANDS_COMPLETE)
    RESULT_CASE(GL_SYNC_STATUS)
    RESULT_CASE(GL_TEXTURE_SWIZZLE_A)
    RESULT_CASE(GL_TEXTURE_SWIZZLE_B)
    RESULT_CASE(GL_TEXTURE_SWIZZLE_G)
    RESULT_CASE(GL_TEXTURE_SWIZZLE_R)
    RESULT_CASE(GL_TEXTURE)
    RESULT_CASE(GL_TEXTURE_1D)
    RESULT_CASE(GL_TEXTURE_1D_ARRAY)
    RESULT_CASE(GL_TEXTURE_2D)
    RESULT_CASE(GL_TEXTURE_2D_MULTISAMPLE)
    RESULT_CASE(GL_TEXTURE_2D_ARRAY)
    RESULT_CASE(GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
    RESULT_CASE(GL_TEXTURE_3D)
    RESULT_CASE(GL_TEXTURE_COMPARE_FUNC)
    RESULT_CASE(GL_TEXTURE_COMPARE_MODE)
    RESULT_CASE(GL_TEXTURE_CUBE_MAP)
    RESULT_CASE(GL_TEXTURE_CUBE_MAP_NEGATIVE_X)
    RESULT_CASE(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y)
    RESULT_CASE(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z)
    RESULT_CASE(GL_TEXTURE_CUBE_MAP_POSITIVE_X)
    RESULT_CASE(GL_TEXTURE_CUBE_MAP_POSITIVE_Y)
    RESULT_CASE(GL_TEXTURE_CUBE_MAP_POSITIVE_Z)
    RESULT_CASE(GL_TEXTURE_CUBE_MAP_SEAMLESS)
    RESULT_CASE(GL_TEXTURE_EXTERNAL_OES)
    RESULT_CASE(GL_TEXTURE_MAG_FILTER)
    RESULT_CASE(GL_TEXTURE_MAX_LEVEL)
    RESULT_CASE(GL_TEXTURE_MAX_LOD)
    RESULT_CASE(GL_TEXTURE_MAX_ANISOTROPY)
    RESULT_CASE(GL_MAX_TEXTURE_MAX_ANISOTROPY)
    RESULT_CASE(GL_MAX_TEXTURE_LOD_BIAS)
    RESULT_CASE(GL_TEXTURE_LOD_BIAS)
    RESULT_CASE(GL_TEXTURE_MIN_FILTER)
    RESULT_CASE(GL_TEXTURE_MIN_LOD)
    RESULT_CASE(GL_TEXTURE_RECTANGLE)
    RESULT_CASE(GL_TEXTURE_WRAP_R)
    RESULT_CASE(GL_TEXTURE_WRAP_S)
    RESULT_CASE(GL_TEXTURE_WRAP_T)
    RESULT_CASE(GL_TEXTURE0)
    RESULT_CASE(GL_TEXTURE1)
    RESULT_CASE(GL_TEXTURE2)
    RESULT_CASE(GL_TEXTURE3)
    RESULT_CASE(GL_TEXTURE4)
    RESULT_CASE(GL_TEXTURE5)
    RESULT_CASE(GL_TEXTURE6)
    RESULT_CASE(GL_TEXTURE7)
    RESULT_CASE(GL_TEXTURE8)
    RESULT_CASE(GL_TRIANGLES)
    RESULT_CASE(GL_TRIANGLE_FAN)
    RESULT_CASE(GL_TRIANGLE_STRIP)
    RESULT_CASE(GL_TRANSFORM_FEEDBACK_BUFFER)
    RESULT_CASE(GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES)
    RESULT_CASE(GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS)
    RESULT_CASE(GL_UNIFORM_BLOCK_BINDING)
    RESULT_CASE(GL_UNIFORM_BLOCK_DATA_SIZE)
    RESULT_CASE(GL_UNIFORM_BUFFER)
    RESULT_CASE(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT)
    RESULT_CASE(GL_UNIFORM_NAME_LENGTH)
    RESULT_CASE(GL_UNIFORM_OFFSET)
    RESULT_CASE(GL_UNPACK_ALIGNMENT)
    RESULT_CASE(GL_UNPACK_ROW_LENGTH)
    RESULT_CASE(GL_UNSIGNED_BYTE)
    RESULT_CASE(GL_UNSIGNED_INT)
    RESULT_CASE(GL_UNSIGNED_INT_10F_11F_11F_REV)
    RESULT_CASE(GL_UNSIGNED_INT_2_10_10_10_REV)
    RESULT_CASE(GL_UNSIGNED_INT_24_8)
    RESULT_CASE(GL_UNSIGNED_INT_5_9_9_9_REV)
    RESULT_CASE(GL_UNSIGNED_INT_8_8_8_8_REV)
    RESULT_CASE(GL_UNSIGNED_SHORT)
    RESULT_CASE(GL_UNSIGNED_SHORT_1_5_5_5_REV)
    RESULT_CASE(GL_UNSIGNED_SHORT_4_4_4_4)
    RESULT_CASE(GL_UNSIGNED_SHORT_4_4_4_4_REV)
    RESULT_CASE(GL_UNSIGNED_SHORT_5_5_5_1)
    RESULT_CASE(GL_UNSIGNED_SHORT_5_6_5)
    RESULT_CASE(GL_UNSIGNED_SHORT_8_8_APPLE)
    RESULT_CASE(GL_UNSIGNED_SHORT_8_8_REV_APPLE)
    RESULT_CASE(GL_UNSIGNED_NORMALIZED)
    RESULT_CASE(GL_VALIDATE_STATUS)
    RESULT_CASE(GL_VENDOR)
    RESULT_CASE(GL_VERSION)
    RESULT_CASE(GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING)
    RESULT_CASE(GL_VERTEX_ATTRIB_ARRAY_ENABLED)
    RESULT_CASE(GL_VERTEX_ATTRIB_ARRAY_NORMALIZED)
    RESULT_CASE(GL_VERTEX_ATTRIB_ARRAY_SIZE)
    RESULT_CASE(GL_VERTEX_ATTRIB_ARRAY_STRIDE)
    RESULT_CASE(GL_VERTEX_ATTRIB_ARRAY_TYPE)
    RESULT_CASE(GL_VERTEX_ARRAY_BINDING)
    RESULT_CASE(GL_VERTEX_SHADER)
    RESULT_CASE(GL_VIEWPORT)
    RESULT_CASE(GL_WRITE_ONLY)
  default:
    std::stringstream stream;
    stream << "0x" << std::hex << code;
    return stream.str();
  }
}

// Helper: iterate over a table of {bit, name} pairs and build a "|"-separated string of the names
// whose bits are set in `bits`.
std::string bitsToString(GLbitfield bits,
                         std::initializer_list<std::pair<GLbitfield, const char*>> table) {
  std::stringstream ss;
  bool first = true;
  for (const auto& entry : table) {
    if ((bits & entry.first) != 0) {
      ss << (first ? "" : " | ") << entry.second;
      first = false;
    }
  }
  return ss.str();
}

std::string clearBitsToString(GLbitfield bits) {
  return bitsToString(bits,
                      {{GL_COLOR_BUFFER_BIT, "GL_COLOR_BUFFER_BIT"},
                       {GL_DEPTH_BUFFER_BIT, "GL_DEPTH_BUFFER_BIT"},
                       {GL_STENCIL_BUFFER_BIT, "GL_STENCIL_BUFFER_BIT"}});
}

std::string mapBufferRangeBitsToString(GLbitfield bits) {
  return bitsToString(bits,
                      {{GL_MAP_READ_BIT, "GL_MAP_READ_BIT"},
                       {GL_MAP_WRITE_BIT, "GL_MAP_WRITE_BIT"},
                       {GL_MAP_PERSISTENT_BIT, "GL_MAP_PERSISTENT_BIT"},
                       {GL_MAP_COHERENT_BIT, "GL_MAP_COHERENT_BIT"},
                       {GL_MAP_INVALIDATE_RANGE_BIT, "GL_MAP_INVALIDATE_RANGE_BIT"},
                       {GL_MAP_INVALIDATE_BUFFER_BIT, "GL_MAP_INVALIDATE_BUFFER_BIT"},
                       {GL_MAP_FLUSH_EXPLICIT_BIT, "GL_MAP_FLUSH_EXPLICIT_BIT"},
                       {GL_MAP_UNSYNCHRONIZED_BIT, "GL_MAP_UNSYNCHRONIZED_BIT"}});
}

std::string memoryBarrierBitsToString(GLbitfield bits) {
  return bitsToString(bits,
                      {{GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT, "GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT"},
                       {GL_ELEMENT_ARRAY_BARRIER_BIT, "GL_ELEMENT_ARRAY_BARRIER_BIT"},
                       {GL_UNIFORM_BARRIER_BIT, "GL_UNIFORM_BARRIER_BIT"},
                       {GL_TEXTURE_FETCH_BARRIER_BIT, "GL_TEXTURE_FETCH_BARRIER_BIT"},
                       {GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, "GL_SHADER_IMAGE_ACCESS_BARRIER_BIT"},
                       {GL_COMMAND_BARRIER_BIT, "GL_COMMAND_BARRIER_BIT"},
                       {GL_PIXEL_BUFFER_BARRIER_BIT, "GL_PIXEL_BUFFER_BARRIER_BIT"},
                       {GL_TEXTURE_UPDATE_BARRIER_BIT, "GL_TEXTURE_UPDATE_BARRIER_BIT"},
                       {GL_BUFFER_UPDATE_BARRIER_BIT, "GL_BUFFER_UPDATE_BARRIER_BIT"},
                       {GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT, "GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT"},
                       {GL_FRAMEBUFFER_BARRIER_BIT, "GL_FRAMEBUFFER_BARRIER_BIT"},
                       {GL_TRANSFORM_FEEDBACK_BARRIER_BIT, "GL_TRANSFORM_FEEDBACK_BARRIER_BIT"},
                       {GL_ATOMIC_COUNTER_BARRIER_BIT, "GL_ATOMIC_COUNTER_BARRIER_BIT"},
                       {GL_SHADER_STORAGE_BARRIER_BIT, "GL_SHADER_STORAGE_BARRIER_BIT"},
                       {GL_QUERY_BUFFER_BARRIER_BIT, "GL_QUERY_BUFFER_BARRIER_BIT"}});
}

#define GL_ENUM_TO_STRING(code) GLenumToString(code).c_str()
#define GL_BOOL_TO_STRING(code) GLboolToString(code).c_str()
#define GL_CLEAR_BITS_TO_STRING(bits) clearBitsToString(bits).c_str()
#define GL_MAP_BUFFER_RANGE_BITS_TO_STRING(bits) mapBufferRangeBitsToString(bits).c_str()
#define GL_MEMORY_BARRIER_BITS_TO_STRING(bits) memoryBarrierBitsToString(bits).c_str()

bool isTextureUniform(GLenum type) {
  switch (type) {
  case GL_SAMPLER_2D:
  case GL_SAMPLER_3D:
  case GL_SAMPLER_CUBE:
  case GL_SAMPLER_2D_SHADOW:
  case GL_SAMPLER_2D_ARRAY:
  case GL_SAMPLER_2D_ARRAY_SHADOW:
  case GL_SAMPLER_2D_MULTISAMPLE:
    return true;
  default:
    return false;
  }
}

bool isTextureImageUniform(GLenum type) {
  switch (type) {
  case GL_IMAGE_2D:
  case GL_IMAGE_3D:
  case GL_IMAGE_CUBE:
  case GL_IMAGE_2D_ARRAY:
  case GL_IMAGE_2D_MULTISAMPLE:
  case GL_IMAGE_2D_MULTISAMPLE_ARRAY:
    return true;
  default:
    return false;
  }
}
} // namespace
#endif // IGL_API_LOG

// Debug logging is not included in code coverage
// FIXME_DEPRECATED_COVERAGE_EXCLUDE_START

namespace {

#if IGL_DEBUG || IGL_API_LOG
const char* glDebugSeverityToString(GLenum severity) {
  switch (severity) {
    RESULT_CASE(GL_DEBUG_SEVERITY_HIGH)
    RESULT_CASE(GL_DEBUG_SEVERITY_LOW)
    RESULT_CASE(GL_DEBUG_SEVERITY_MEDIUM)
    RESULT_CASE(GL_DEBUG_SEVERITY_NOTIFICATION)
  default:
    return "GL_DEBUG_SEVERITY_UNKNOWN";
  }
}

const char* glDebugSourceToString(GLenum source) {
  switch (source) {
    RESULT_CASE(GL_DEBUG_SOURCE_API)
    RESULT_CASE(GL_DEBUG_SOURCE_APPLICATION)
    RESULT_CASE(GL_DEBUG_SOURCE_OTHER)
    RESULT_CASE(GL_DEBUG_SOURCE_SHADER_COMPILER)
    RESULT_CASE(GL_DEBUG_SOURCE_THIRD_PARTY)
    RESULT_CASE(GL_DEBUG_SOURCE_WINDOW_SYSTEM)
  default:
    return "GL_DEBUG_SOURCE_WINDOW_UNKNOWN";
  }
}

const char* glDebugTypeToString(GLenum type) {
  switch (type) {
    RESULT_CASE(GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR)
    RESULT_CASE(GL_DEBUG_TYPE_ERROR)
    RESULT_CASE(GL_DEBUG_TYPE_MARKER)
    RESULT_CASE(GL_DEBUG_TYPE_OTHER)
    RESULT_CASE(GL_DEBUG_TYPE_PERFORMANCE)
    RESULT_CASE(GL_DEBUG_TYPE_PORTABILITY)
    RESULT_CASE(GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR)
  default:
    return "GL_DEBUG_TYPE_UNDEFINED_UNKNOWN";
  }
}

void logDebugMessage(GLenum source,
                     GLenum type,
                     GLuint id,
                     GLenum severity,
                     GLsizei length,
                     const GLchar* message) {
  const auto logLevel = severity == GL_DEBUG_SEVERITY_HIGH
                            ? IGLLogError
                            : (severity == GL_DEBUG_SEVERITY_MEDIUM ? IGLLogWarning : IGLLogInfo);
  IGLLog(logLevel,
         "%s %s %u %s: %.*s\n",
         glDebugSourceToString(source),
         glDebugTypeToString(type),
         id,
         glDebugSeverityToString(severity),
         static_cast<int>(message ? length : 0),
         message ? message : "");
}
#if IGL_API_LOG
void logOpenGlDebugMessage(GLenum source,
                           GLenum type,
                           GLuint id,
                           GLenum severity,
                           GLsizei length,
                           const GLchar* message,
                           const void* userParam) {
  logDebugMessage(source, type, id, severity, length, message);
}
#endif // IGL_API_LOG
#endif // IGL_DEBUG || IGL_API_LOG
} // namespace

// FIXME_DEPRECATED_COVERAGE_EXCLUDE_END

namespace igl::opengl {
namespace {
const char* GLerrorToString(GLenum error) {
  switch (error) {
  case GL_NO_ERROR:
    return "";

    RESULT_CASE(GL_INVALID_ENUM)
    RESULT_CASE(GL_INVALID_VALUE)
    RESULT_CASE(GL_INVALID_OPERATION)
    RESULT_CASE(GL_INVALID_FRAMEBUFFER_OPERATION)
    RESULT_CASE(GL_OUT_OF_MEMORY)
  default:
    return "UNKNOWN GL ERROR";
  }
}

Result::Code GLerrorToCode(GLenum error) {
  switch (error) {
  case GL_NO_ERROR:
    return Result::Code::Ok;
  case GL_INVALID_ENUM:
  case GL_INVALID_VALUE:
    return Result::Code::ArgumentInvalid;
  case GL_INVALID_OPERATION:
  case GL_INVALID_FRAMEBUFFER_OPERATION:
    return Result::Code::InvalidOperation;
  default:
    return Result::Code::RuntimeError;
  }
}

#define GL_ERROR_TO_STRING(error) GLerrorToString(error)
#define GL_ERROR_TO_RESULT(error) Result(GLerrorToCode(error), GLerrorToString(error))
} // namespace

// NOLINTNEXTLINE(modernize-use-equals-default)
IContext::IContext() : deviceFeatureSet_(*this) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
#if IGL_DEBUG
  // In debug mode, we default to always checking errors after each OGL call
  alwaysCheckError_ = true;
#endif
#if defined(IGL_VALIDATE_SHADERS)
  shouldValidateShaders_ = true;
#endif
}

IContext::~IContext() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  IGL_SOFT_ASSERT(refCount_.load(std::memory_order_acquire) == 0,
                  "Dangling IContext reference left behind."
                  // @fb-only
  );
  // Clear the zombie guard explicitly so our "secret" stays secret.
  zombieGuard_ = 0;
}

#if IGL_API_LOG
void IContext::setBoundTexture(GLenum target, GLuint texture) {
  switch (target) {
  case GL_TEXTURE_2D:
    boundTexture2D_ = texture;
    break;
  case GL_TEXTURE_3D:
    boundTexture3D_ = texture;
    break;
  case GL_TEXTURE_CUBE_MAP:
  case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
  case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
  case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
  case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
  case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
  case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
    boundTextureCubeMap_ = texture;
    break;
  case GL_TEXTURE_2D_ARRAY:
    boundTexture2DArray_ = texture;
    break;
  case GL_TEXTURE_EXTERNAL_OES:
    boundTextureExternalOes_ = texture;
    break;
  default:
    IGL_DEBUG_ABORT("Unsupported texture target: %s", GL_ENUM_TO_STRING(target));
    break;
  }
}

GLuint IContext::boundTexture(GLenum target) const {
  switch (target) {
  case GL_TEXTURE_2D:
    return boundTexture2D_;
  case GL_TEXTURE_3D:
    return boundTexture3D_;
  case GL_TEXTURE_CUBE_MAP:
  case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
  case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
  case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
  case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
  case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
  case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
    return boundTextureCubeMap_;
  case GL_TEXTURE_2D_ARRAY:
    return boundTexture2DArray_;
  case GL_TEXTURE_EXTERNAL_OES:
    return boundTextureExternalOes_;
  default:
    IGL_DEBUG_ABORT("Unsupported texture target: %s", GL_ENUM_TO_STRING(target));
    return 0;
    break;
  }
}

void IContext::setBoundBuffer(GLenum target, GLuint buffer) {
  switch (target) {
  case GL_ARRAY_BUFFER:
    boundArrayBuffer_ = buffer;
    break;
  case GL_DRAW_INDIRECT_BUFFER:
    boundDrawIndirectBuffer_ = buffer;
    break;
  case GL_COPY_READ_BUFFER:
    boundCopyReadBuffer_ = buffer;
    break;
  case GL_COPY_WRITE_BUFFER:
    boundCopyWriteBuffer_ = buffer;
    break;
  case GL_ELEMENT_ARRAY_BUFFER:
    boundElementArrayBuffer_ = buffer;
    break;
  case GL_PIXEL_PACK_BUFFER:
    boundPixelPackBuffer_ = buffer;
    break;
  case GL_PIXEL_UNPACK_BUFFER:
    boundPixelPackBuffer_ = buffer;
    break;
  case GL_SHADER_STORAGE_BUFFER:
    boundShaderStorageBuffer_ = buffer;
    break;
  case GL_UNIFORM_BUFFER:
    boundUniformBuffer_ = buffer;
    break;
  default:
    IGL_DEBUG_ABORT("Unsupported buffer target: %s", GL_ENUM_TO_STRING(target));
    break;
  }
}

GLuint IContext::boundBuffer(GLenum target) const {
  switch (target) {
  case GL_ARRAY_BUFFER:
    return boundArrayBuffer_;
  case GL_DRAW_INDIRECT_BUFFER:
    return boundDrawIndirectBuffer_;
  case GL_COPY_READ_BUFFER:
    return boundCopyReadBuffer_;
  case GL_COPY_WRITE_BUFFER:
    return boundCopyWriteBuffer_;
  case GL_ELEMENT_ARRAY_BUFFER:
    return boundElementArrayBuffer_;
  case GL_PIXEL_PACK_BUFFER:
    return boundPixelPackBuffer_;
  case GL_PIXEL_UNPACK_BUFFER:
    return boundPixelPackBuffer_;
  case GL_SHADER_STORAGE_BUFFER:
    return boundShaderStorageBuffer_;
  case GL_UNIFORM_BUFFER:
    return boundUniformBuffer_;
  default:
    IGL_DEBUG_ABORT("Unsupported buffer target: %s", GL_ENUM_TO_STRING(target));
    return 0;
    break;
  }
}

void IContext::setBoundFramebuffer(GLenum target, GLuint framebuffer) {
  switch (target) {
  case GL_READ_FRAMEBUFFER:
    boundReadFramebuffer_ = framebuffer;
    break;
  case GL_DRAW_FRAMEBUFFER:
    boundWriteFramebuffer_ = framebuffer;
    break;
  case GL_FRAMEBUFFER:
    boundReadFramebuffer_ = framebuffer;
    boundWriteFramebuffer_ = framebuffer;
    break;
  default:
    IGL_DEBUG_ABORT("Unsupported framebuffer target: %s", GL_ENUM_TO_STRING(target));
    break;
  }
}

GLuint IContext::boundFramebuffer(GLenum target) const {
  switch (target) {
  case GL_READ_FRAMEBUFFER:
    return boundReadFramebuffer_;
  case GL_DRAW_FRAMEBUFFER:
  case GL_FRAMEBUFFER:
    return boundWriteFramebuffer_;
  default:
    IGL_DEBUG_ABORT("Unsupported framebuffer target: %s", GL_ENUM_TO_STRING(target));
    return 0;
    break;
  }
}

void IContext::setFramebufferAttachment(GLenum target,
                                        GLenum attachment,
                                        bool isRenderbuffer,
                                        GLuint object) {
  const auto framebuffer = boundFramebuffer(target);
  auto& attachments = framebufferAttachments_[framebuffer];
  if (attachment == GL_DEPTH_ATTACHMENT) {
    attachments.depthAttachment = object;
    attachments.depthIsRenderbuffer = isRenderbuffer;
  } else if (attachment == GL_STENCIL_ATTACHMENT) {
    attachments.stencilAttachment = object;
    attachments.stencilIsRenderbuffer = isRenderbuffer;
  } else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
    attachments.stencilAttachment = object;
    attachments.stencilIsRenderbuffer = isRenderbuffer;
    attachments.depthAttachment = object;
    attachments.depthIsRenderbuffer = isRenderbuffer;
  } else if (attachment >= GL_COLOR_ATTACHMENT0 &&
             (attachment - GL_COLOR_ATTACHMENT0) < attachments.colorAttachment.size()) {
    attachments.colorAttachment[attachment - GL_COLOR_ATTACHMENT0] = object;
    attachments.colorIsRenderbuffer.set(attachment - GL_COLOR_ATTACHMENT0, isRenderbuffer);
  }
}
const char* IContext::framebufferAttachmentType(GLenum target, GLenum attachment) const {
  bool isRenderbuffer = true;
  const auto framebuffer = boundFramebuffer(target);
  const auto it = framebufferAttachments_.find(framebuffer);
  if (it == framebufferAttachments_.end()) {
    return "unknown";
  }
  const auto& attachments = it->second;
  if (attachment == GL_DEPTH_ATTACHMENT) {
    isRenderbuffer = attachments.depthIsRenderbuffer;
  } else if (attachment == GL_STENCIL_ATTACHMENT) {
    isRenderbuffer = attachments.stencilIsRenderbuffer;
  } else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
    IGL_DEBUG_ASSERT(attachments.depthIsRenderbuffer == attachments.stencilIsRenderbuffer);
    if (attachments.depthIsRenderbuffer != attachments.stencilIsRenderbuffer) {
      return "unknown";
    }
    isRenderbuffer = attachments.depthIsRenderbuffer;
  } else if (attachment >= GL_COLOR_ATTACHMENT0 &&
             (attachment - GL_COLOR_ATTACHMENT0) < attachments.colorAttachment.size()) {
    isRenderbuffer = attachments.colorIsRenderbuffer.test(attachment - GL_COLOR_ATTACHMENT0);
  } else {
    return "unknown";
  }
  if (isRenderbuffer) {
    return "renderbuffer";
  } else {
    return "texture";
  }
}

const char* IContext::framebufferAttachmentType(GLenum target,
                                                GLenum buffer,
                                                GLint drawbuffer) const {
  if (buffer == GL_COLOR) {
    return framebufferAttachmentType(target, GL_COLOR_ATTACHMENT0 + drawbuffer);
  } else if (buffer == GL_DEPTH) {
    IGL_DEBUG_ASSERT(drawbuffer == 0);
    return framebufferAttachmentType(target, GL_DEPTH_ATTACHMENT);
  } else if (buffer == GL_STENCIL) {
    IGL_DEBUG_ASSERT(drawbuffer == 0);
    return framebufferAttachmentType(target, GL_STENCIL_ATTACHMENT);
  } else if (buffer == GL_DEPTH_STENCIL) {
    IGL_DEBUG_ASSERT(drawbuffer == 0);
    return framebufferAttachmentType(target, GL_DEPTH_STENCIL_ATTACHMENT);
  }

  return "unknown";
}

GLuint IContext::framebufferAttachment(GLenum target, GLenum attachment) const {
  const auto framebuffer = boundFramebuffer(target);
  const auto it = framebufferAttachments_.find(framebuffer);
  if (it == framebufferAttachments_.end()) {
    return 0;
  }
  const auto& attachments = it->second;
  if (attachment == GL_DEPTH_ATTACHMENT) {
    return attachments.depthAttachment;
  } else if (attachment == GL_STENCIL_ATTACHMENT) {
    return attachments.stencilAttachment;
  } else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
    IGL_DEBUG_ASSERT(attachments.depthAttachment == attachments.stencilAttachment);
    if (attachments.depthAttachment != attachments.stencilAttachment) {
      return -1;
    } else {
      return attachments.depthAttachment;
    }
  } else if (attachment >= GL_COLOR_ATTACHMENT0 &&
             (attachment - GL_COLOR_ATTACHMENT0) < attachments.colorAttachment.size()) {
    return attachments.colorAttachment[attachment - GL_COLOR_ATTACHMENT0];
  } else {
    return 0;
  }
}

GLuint IContext::framebufferAttachment(GLenum target, GLenum buffer, GLint drawbuffer) const {
  if (buffer == GL_COLOR) {
    return framebufferAttachment(target, GL_COLOR_ATTACHMENT0 + drawbuffer);
  } else if (buffer == GL_DEPTH) {
    IGL_DEBUG_ASSERT(drawbuffer == 0);
    return framebufferAttachment(target, GL_DEPTH_ATTACHMENT);
  } else if (buffer == GL_STENCIL) {
    IGL_DEBUG_ASSERT(drawbuffer == 0);
    return framebufferAttachment(target, GL_STENCIL_ATTACHMENT);
  } else if (buffer == GL_DEPTH_STENCIL) {
    IGL_DEBUG_ASSERT(drawbuffer == 0);
    return framebufferAttachment(target, GL_DEPTH_STENCIL_ATTACHMENT);
  }

  return -1;
}

const char* IContext::framebufferReadBufferType(GLenum target) const {
  const auto framebuffer = boundFramebuffer(target);
  const auto it = framebufferAttachments_.find(framebuffer);
  if (it == framebufferAttachments_.end()) {
    return "unknown";
  }
  return framebufferAttachmentType(target, GL_COLOR_ATTACHMENT0 + it->second.readBufferIndex);
}

GLuint IContext::framebufferReadBuffer(GLenum target) const {
  const auto framebuffer = boundFramebuffer(target);
  const auto it = framebufferAttachments_.find(framebuffer);
  if (it == framebufferAttachments_.end()) {
    return 0;
  }
  return framebufferAttachment(target, GL_COLOR_ATTACHMENT0 + it->second.readBufferIndex);
}

std::string IContext::affectedFramebufferAttachments(GLenum target, GLbitfield mask) const {
  std::stringstream ss;
  const auto framebuffer = boundFramebuffer(target);
  const auto it = framebufferAttachments_.find(framebuffer);
  if (it == framebufferAttachments_.end()) {
    return ss.str();
  }
  const auto& attachments = it->second;
  bool first = true;
  if ((mask & GL_COLOR_BUFFER_BIT) != 0) {
    for (uint32_t i = 0; i < attachments.colorAttachment.size(); ++i) {
      if (attachments.colorAttachment[i] != 0) {
        ss << (first ? "" : " ") << "color " << i << " ("
           << (attachments.colorIsRenderbuffer.test(i) ? "renderbuffer" : "texture") << ": "
           << attachments.colorAttachment[i] << ")";
        first = false;
      }
    }
  }
  if ((mask & GL_DEPTH_BUFFER_BIT) != 0 && attachments.depthAttachment != 0) {
    ss << (first ? "" : " ") << "depth ("
       << (attachments.depthIsRenderbuffer ? "renderbuffer" : "texture") << ": "
       << attachments.depthAttachment << ")";
    first = false;
  }
  if ((mask & GL_STENCIL_BUFFER_BIT) != 0 && attachments.stencilAttachment != 0) {
    ss << (first ? "" : " ") << "stencil ("
       << (attachments.depthIsRenderbuffer ? "renderbuffer" : "texture") << ": "
       << attachments.depthAttachment << ")";
  }
  return ss.str();
}
std::string IContext::boundFramebufferAttachments(GLenum target) const {
  return affectedFramebufferAttachments(
      target, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

std::string IContext::boundDrawBuffers() const {
  std::stringstream ss;
  bool first = true;
  for (uint32_t i = 0; i < enabledAttributes_.size(); ++i) {
    if (enabledAttributes_.test(i)) {
      ss << (first ? "" : " ") << "attribute " << i << " (buffer: " << boundAttributes_[i] << ")";
      first = false;
    }
  }
  return ss.str();
}

std::string IContext::boundImageTextures() const {
  std::stringstream ss;
  bool first = true;
  for (uint32_t i = 0; i < boundImageTextures_.size(); ++i) {
    if (boundImageTextures_[i] != 0) {
      ss << (first ? "" : " ") << "image " << i << " (texture: " << boundImageTextures_[i] << ")";
      first = false;
    }
  }
  return ss.str();
}

std::string IContext::boundDrawTextures() const {
  std::stringstream ss;
  bool first = true;
  for (uint32_t i = 0; i < boundDrawTextures_.size(); ++i) {
    if (boundDrawTextures_[i] != 0) {
      ss << (first ? "" : " ") << "draw " << i << " (texture: " << boundDrawTextures_[i] << ")";
      first = false;
    }
  }
  return ss.str();
}

std::string IContext::identifierLabel(GLuint identifier, GLuint name) const {
  std::stringstream ss;
  switch (identifier) {
  case GL_BUFFER:
    ss << "(buffer: " << name << ")";
    break;
  case GL_FRAMEBUFFER:
    ss << "(framebuffer: " << name << ")";
    break;
  case GL_PROGRAM:
    ss << "(program: " << name << ")";
    break;
  case GL_RENDERBUFFER:
    ss << "(renderbuffer: " << name << ")";
    break;
  case GL_TEXTURE:
    ss << "(texture: " << name << ")";
    break;
  default:
    break;
  }
  return ss.str();
}

namespace {
void appendWithSep(std::stringstream& ss, bool& first, const std::string& text) {
  if (!first) {
    ss << " ";
  }
  ss << text;
  first = false;
}
} // namespace

std::string IContext::boundPixelBufferObjects() const {
  std::stringstream ss;
  if (boundPixelUnpackBuffer_ != 0) {
    ss << "unpack (buffer: " << boundPixelUnpackBuffer_ << ")";
  }
  if (boundPixelPackBuffer_ != 0) {
    if (boundPixelUnpackBuffer_ != 0) {
      ss << " ";
    }
    ss << "pack (buffer: " << boundPixelPackBuffer_ << ")";
  }
  return ss.str();
}

std::string IContext::affectedMemoryBarrierObjects(GLbitfield bits) const {
  constexpr const GLbitfield kProgramBits =
      GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT |
      GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_COMMAND_BARRIER_BIT |
      GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT |
      GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT |
      GL_TRANSFORM_FEEDBACK_BARRIER_BIT | GL_QUERY_BUFFER_BARRIER_BIT;
  constexpr const GLbitfield kTextureBits =
      GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
      GL_TEXTURE_UPDATE_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT;
  bool first = true;
  std::stringstream ss;
  if ((bits & kProgramBits) != 0) {
    std::stringstream prog;
    prog << "(program: " << boundProgram_ << ")";
    appendWithSep(ss, first, prog.str());
  }
  if ((bits & GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT) != 0) {
    appendWithSep(ss, first, boundDrawBuffers());
  }
  if ((bits & GL_ELEMENT_ARRAY_BARRIER_BIT) != 0) {
    std::stringstream elem;
    elem << "element (buffer: " << boundElementArrayBuffer_ << ")";
    appendWithSep(ss, first, elem.str());
  }
  if ((bits & kTextureBits) != 0) {
    const std::string texStr =
        lastCommandWasCompute_
            ? boundImageTextures()
            : (boundDrawTextures() + " " + boundFramebufferAttachments(GL_DRAW_FRAMEBUFFER) + " " +
               boundFramebufferAttachments(GL_READ_FRAMEBUFFER));
    appendWithSep(ss, first, texStr);
  }
  if ((bits & GL_PIXEL_BUFFER_BARRIER_BIT) != 0 &&
      (boundPixelPackBuffer_ != 0 || boundPixelUnpackBuffer_ != 0)) {
    appendWithSep(ss, first, boundPixelBufferObjects());
  }
  if ((bits & GL_UNIFORM_BARRIER_BIT) != 0) {
    appendWithSep(ss, first, boundBuffersByIndex(/* ssbos */ false, /* ubos */ true));
  }
  if ((bits & GL_SHADER_STORAGE_BARRIER_BIT) != 0) {
    appendWithSep(ss, first, boundBuffersByIndex(/* ssbos */ true, /* ubos */ false));
  }

  return ss.str();
}

void IContext::addProgramUniform(GLuint program,
                                 GLsizei* length,
                                 GLenum* type,
                                 GLchar* name) const {
  std::string nameStr(std::string_view(name, length == nullptr ? std::strlen(name) : *length));
  auto& programUniforms = programUniforms_[program];
  auto uniformIt = programUniforms.find(nameStr);
  if (uniformIt == programUniforms.end()) {
    uniformIt = programUniforms.emplace(std::move(nameStr), ProgramUniform{}).first;
    if (uniformIt == programUniforms.end()) {
      return;
    }
  }
  auto& uniform = uniformIt->second;
  uniform.type = (type == nullptr ? GL_NONE : *type);
  uniform.isTexture = isTextureUniform(uniform.type);
  uniform.isImage = isTextureImageUniform(uniform.type);
}

void IContext::setProgramUniformLocation(GLuint program, const GLchar* name, GLint location) const {
  if (location < 0) {
    return;
  }
  std::string nameStr(name);
  auto& programUniforms = programUniforms_[program];
  auto uniformIt = programUniforms.find(nameStr);
  if (uniformIt == programUniforms.end()) {
    return;
  }

  auto& programUniformsByLocation = programUniformsByLocation_[program];
  programUniformsByLocation[location] = std::move(nameStr);
}

void IContext::setProgramResourceIndex(GLuint program,
                                       GLenum programInterface,
                                       const GLchar* name,
                                       GLuint index) const {
  if (programInterface == GL_SHADER_STORAGE_BLOCK) {
    programSsbosByIndex_[program][index] = name;
  }
}

std::string IContext::boundUniformTexture(GLuint program, GLint location, GLint unit) const {
  const auto programUniformsIt = programUniforms_.find(program);
  if (programUniformsIt == programUniforms_.end()) {
    return {};
  }
  const auto& programUniforms = programUniformsIt->second;
  const auto programUniformsByLocationIt = programUniformsByLocation_.find(program);
  if (programUniformsByLocationIt == programUniformsByLocation_.end()) {
    return {};
  }
  const auto& programUniformsByLocation = programUniformsByLocationIt->second;

  const auto uniformByLocationIt = programUniformsByLocation.find(location);
  if (uniformByLocationIt == programUniformsByLocation.end()) {
    return {};
  }
  const auto uniformIt = programUniforms.find(uniformByLocationIt->second);
  if (uniformIt == programUniforms.end()) {
    return {};
  }
  if (uniformIt->second.isTexture && unit >= 0 && unit < boundDrawTextures_.size()) {
    std::stringstream ss;
    ss << "(texture: " << boundDrawTextures_[unit] << ")";
    return ss.str();
  }
  if (uniformIt->second.isImage && unit >= 0 && unit < boundImageTextures_.size()) {
    std::stringstream ss;
    ss << "(texture: " << boundImageTextures_[unit] << ")";
    return ss.str();
  }

  return {};
}

std::string IContext::boundBufferName(GLuint program, GLenum target, GLuint index) const {
  if (target == GL_SHADER_STORAGE_BUFFER) {
    const auto programSsbosByLocationIt = programSsbosByLocation_.find(program);
    if (programSsbosByLocationIt == programSsbosByLocation_.end()) {
      return {};
    }
    const auto it = programSsbosByLocationIt->second.find(index);
    if (it == programSsbosByLocationIt->second.end()) {
      return {};
    }
    std::stringstream ss;
    ss << "(ssbo: " << it->second << ")";
    return ss.str();
  }
  if (target == GL_UNIFORM_BUFFER) {
    const auto programUbosByLocationIt = programUbosByLocation_.find(program);
    if (programUbosByLocationIt == programUbosByLocation_.end()) {
      return {};
    }
    const auto it = programUbosByLocationIt->second.find(index);
    if (it == programUbosByLocationIt->second.end()) {
      return {};
    }
    std::stringstream ss;
    ss << "(ubo: " << it->second << ")";
    return ss.str();
  }
  return {};
}

void IContext::setBoundBufferByIndex(GLenum target, GLuint index, GLuint buffer) {
  if (target == GL_SHADER_STORAGE_BUFFER) {
    if (index < boundSsbos_.size()) {
      boundSsbos_[index] = buffer;
    }
  } else if (target == GL_UNIFORM_BUFFER) {
    if (index < boundUbos_.size()) {
      boundUbos_[index] = buffer;
    }
  }
}

GLuint IContext::boundBufferByIndex(GLenum target, GLuint index) const {
  if (target == GL_SHADER_STORAGE_BUFFER) {
    if (index < boundSsbos_.size()) {
      return boundSsbos_[index];
    }
  } else if (target == GL_UNIFORM_BUFFER) {
    if (index < boundUbos_.size()) {
      return boundUbos_[index];
    }
  }

  return 0;
}

std::string IContext::boundBuffersByIndex(bool ssbos, bool ubos) const {
  bool first = true;
  std::stringstream ss;
  if (ubos) {
    for (uint32_t i = 0; i < boundUbos_.size(); ++i) {
      if (boundUbos_[i] != 0) {
        ss << (first ? "" : " ") << "ubo " << i << " (buffer: " << boundUbos_[i] << ")";
        first = false;
      }
    }
  }
  if (ssbos) {
    for (uint32_t i = 0; i < boundSsbos_.size(); ++i) {
      if (boundSsbos_[i] != 0) {
        ss << (first ? "" : " ") << "ssbo " << i << " (buffer: " << boundSsbos_[i] << ")";
        first = false;
      }
    }
  }

  return ss.str();
}
#endif // IGL_API_LOG

#if IGL_DEBUG
namespace {
// The map can be accessed from multiple threads during context creation/destruction.
std::mutex& getMutex() {
  static std::mutex mutex; // NOLINT(facebook-static-object-destructor-check)
  return mutex;
}
} // namespace
#endif

// Creates a global map to ensure multiple IContexts are not created for a single glContext
std::unordered_map<void* IGL_NULLABLE, IContext*>& IContext::getExistingContexts() {
  static auto& map = *(new std::unordered_map<void* IGL_NULLABLE, IContext*>());
  return map;
}

// IContext is a thin forwarding layer over the GL/EGL proc table. The nullable-dereference
// findings across the wrappers below are false positives: nullable pointer arguments
// (data/value/buffers/objects/...) are forwarded verbatim to the driver, which permits null
// for these calls per the GL contract, and are never dereferenced here; the GLCALL_PROC
// proc-pointer dereference is guarded by a release-effective IGL_DEBUG_VERIFY; and the few
// remaining hits are debug-only paths over non-null string literals. clang-tidy cannot model
// any of these, so suppress the signal for the whole forwarding section.
// NOLINTBEGIN(facebook-hte-NullableDereference)
void IContext::registerContext(void* IGL_NULLABLE glContext, IContext* context) {
  IGL_PROFILER_FUNCTION();
#if IGL_DEBUG
  std::lock_guard<std::mutex> lock(getMutex());
  auto result = IContext::getExistingContexts().find(glContext);
  if (result != IContext::getExistingContexts().end()) {
    const char* errorMessage =
        "Your application creates multiple IContext wrappers for the same underlying context "
        "object, which can result in problems if those contexts are used simultaneously across "
        "different threads. It's recommended to preserve a one-to-one relationship between "
        "native "
        "and IGL contexts. Ignore this warning at your own risk.";
#if IGL_PLATFORM_ANDROID
    IGL_LOG_ERROR(errorMessage);
#else
    IGL_DEBUG_ABORT(errorMessage);
#endif
  }
  IContext::getExistingContexts().insert({glContext, context});
#endif
}

void IContext::willDestroy(void* IGL_NULLABLE glContext) {
  IGL_PROFILER_FUNCTION();
  // Clear pool explicitly, since it might have reference back to IContext.
  getAdapterPool().clear();
  getComputeAdapterPool().clear();
  // Unregister context
  if (glContext != nullptr) {
    IContext::unregisterContext(glContext);
  }
}

void IContext::unregisterContext(void* IGL_NULLABLE glContext) {
#if IGL_DEBUG
  std::lock_guard<std::mutex> lock(getMutex());
  IContext::getExistingContexts().erase(glContext);
#endif
}

void IContext::flushDeletionQueue() {
  deletionQueues_.flushDeletionQueue(*this);
}

bool IContext::shouldQueueAPI() const {
  return !isCurrentContext() && !isCurrentSharegroup();
}

void IContext::activeTexture(GLenum texture) {
#if IGL_API_LOG
  activeTextureUnit_ = texture - GL_TEXTURE0;
#endif
  APILOG("glActiveTexture(%s)\n", GL_ENUM_TO_STRING(texture));
  GLCALL(ActiveTexture)(texture);
  GLCHECK_ERRORS();
}

void IContext::attachShader(GLuint program, GLuint shader) {
  APILOG("glAttachShader(%u, %u) (program: %u)\n", program, shader, program);
  GLCALL(AttachShader)(program, shader);
  GLCHECK_ERRORS();
}

void IContext::bindBuffer(GLenum target, GLuint buffer) {
#if IGL_API_LOG
  GLuint unboundBuffer = boundBuffer(target);
  setBoundBuffer(target, buffer);
#endif // IGL_API_LOG
  APILOG("glBindBuffer(%s, %u) unbound (buffer: %u) bound (buffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         buffer,
         unboundBuffer,
         buffer);
  GLCALL(BindBuffer)(target, buffer);
  GLCHECK_ERRORS();
}

void IContext::bindBufferBase(GLenum target, GLuint index, GLuint buffer) {
#if IGL_API_LOG
  GLuint unboundBuffer = boundBufferByIndex(target, index);
  setBoundBufferByIndex(target, index, buffer);
#endif
  APILOG("glBindBufferBase(%s, %u, %u) unbound (buffer: %u) bound (buffer: %u) %s\n",
         GL_ENUM_TO_STRING(target),
         index,
         buffer,
         unboundBuffer,
         buffer,
         boundBufferName(boundProgram_, target, index).c_str());
  IGLCALL(BindBufferBase)(target, index, buffer);
  GLCHECK_ERRORS();
}

void IContext::bindBufferRange(GLenum target,
                               GLuint index,
                               GLuint buffer,
                               GLintptr offset,
                               GLsizeiptr size) {
#if IGL_API_LOG
  GLuint unboundBuffer = boundBufferByIndex(target, index);
  setBoundBufferByIndex(target, index, buffer);
#endif
  APILOG("glBindBufferRange(%s, %u, %u) unbound (buffer: %u) bound (buffer: %u) %s\n",
         GL_ENUM_TO_STRING(target),
         index,
         buffer,
         unboundBuffer,
         buffer,
         boundBufferName(boundProgram_, target, index).c_str());
  IGLCALL(BindBufferRange)(target, index, buffer, offset, size);
  GLCHECK_ERRORS();
}

void IContext::bindFramebuffer(GLenum target, GLuint framebuffer) {
#if IGL_API_LOG
  GLuint unboundFramebuffer = boundFramebuffer(target);
  setBoundFramebuffer(target, framebuffer);
#endif
  APILOG("glBindFramebuffer(%s, %u) unbound (framebuffer: %u) bound (framebuffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         framebuffer,
         unboundFramebuffer,
         framebuffer);
  IGLCALL(BindFramebuffer)(target, framebuffer);
  GLCHECK_ERRORS();
}

void IContext::bindRenderbuffer(GLenum target, GLuint renderbuffer) {
#if IGL_API_LOG
  GLuint unboundRenderbuffer = boundRenderbuffer_;
  boundRenderbuffer_ = renderbuffer;
#endif
  APILOG("glBindRenderbuffer(%s, %u) unbound (renderbuffer: %u) bound (renderbuffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         renderbuffer,
         unboundRenderbuffer,
         renderbuffer);
  IGLCALL(BindRenderbuffer)(target, renderbuffer);
  GLCHECK_ERRORS();
}

void IContext::bindTexture(GLenum target, GLuint texture) {
#if IGL_API_LOG
  GLuint unboundTexture = boundTexture(target);
  setBoundTexture(target, texture);
  if (activeTextureUnit_ < boundDrawTextures_.size()) {
    boundDrawTextures_[activeTextureUnit_] = texture;
  }
#endif
  APILOG("glBindTexture(%s, %u) (unit: %u) unbound (texture: %u) bound (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         activeTextureUnit_,
         texture,
         unboundTexture,
         texture);
  GLCALL(BindTexture)(target, texture);
  GLCHECK_ERRORS();
}

void IContext::bindImageTexture(GLuint unit,
                                GLuint texture,
                                GLint level,
                                GLboolean layered,
                                GLint layer,
                                GLenum access,
                                GLenum format) {
#if IGL_API_LOG
  GLuint unboundTexture = 0;
  if (unit < boundImageTextures_.size()) {
    unboundTexture = boundImageTextures_[unit];
    boundImageTextures_[unit] = texture;
  }
#endif
  if (bindImageTexturerProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::ShaderImageLoadStoreExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::ShaderImageLoadStore)) {
        bindImageTexturerProc_ = iglBindImageTextureEXT;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::ShaderImageLoadStore)) {
      bindImageTexturerProc_ = iglBindImageTexture;
    }
    IGL_DEBUG_ASSERT(bindImageTexturerProc_, "No supported function for glBindImageTexture\n");
  }
  APILOG(
      "glBindImageTexture(%u, %u, %i, %s, %i %s %s) unbound (texture: %u) bound (texture: %u) "
      "(uniform: %s)\n",
      unit,
      texture,
      level,
      GL_BOOL_TO_STRING(layered),
      layer,
      GL_ENUM_TO_STRING(access),
      GL_ENUM_TO_STRING(format),
      unboundTexture,
      texture,
      programUniformsByLocation_[boundProgram_][unit].c_str());
  GLCALL_PROC(bindImageTexturerProc_, unit, texture, level, layered, layer, access, format);
  GLCHECK_ERRORS();
}

void IContext::bindVertexArray(GLuint vao) {
#if IGL_API_LOG
  GLuint unboundVao = boundVao_;
  boundVao_ = vao;
#endif
  if (bindVertexArrayProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::VertexArrayObjectExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::VertexArrayObject)) {
        bindVertexArrayProc_ = iglBindVertexArrayOES;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::VertexArrayObject)) {
      bindVertexArrayProc_ = iglBindVertexArray;
    }
    IGL_DEBUG_ASSERT(bindVertexArrayProc_, "No supported function for glBindVertexArray\n");
  }
  APILOG("glBindVertexArray(%u) unbound (VAO: %u) bound (VAO: %u)\n", vao, unboundVao, vao);
  GLCALL_PROC(bindVertexArrayProc_, vao);
  GLCHECK_ERRORS();
}

void IContext::blendColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
  APILOG("glBlendColor(%f, %f, %f, %f)\n", red, green, blue, alpha);
  GLCALL(BlendColor)(red, green, blue, alpha);
  GLCHECK_ERRORS();
}

void IContext::blendEquation(GLenum mode) {
  APILOG("glBlendEquation(%s)\n", GL_ENUM_TO_STRING(mode));
  GLCALL(BlendEquation)(mode);
  GLCHECK_ERRORS();
}

void IContext::blendEquationSeparate(GLenum modeRGB, GLenum modeAlpha) {
  APILOG("glBlendEquationSeparate(%s, %s)\n",
         GL_ENUM_TO_STRING(modeRGB),
         GL_ENUM_TO_STRING(modeAlpha));
  GLCALL(BlendEquationSeparate)(modeRGB, modeAlpha);
  GLCHECK_ERRORS();
}

namespace {
// Returns the function that sets the state of one draw buffer: `core` if the context has it as a
// core function, otherwise `oes` or `ext` if the context has that extension, or nullptr
template<typename Proc>
Proc IGL_NULLABLE
selectDrawBuffersIndexedProc(const DeviceFeatureSet& features, Proc core, Proc oes, Proc ext) {
  if (features.hasInternalRequirement(InternalRequirement::DrawBuffersIndexedExtReq)) {
    if (features.hasExtension(Extensions::DrawBuffersIndexedOes)) {
      return oes;
    }
    if (features.hasExtension(Extensions::DrawBuffersIndexedExt)) {
      return ext;
    }
    return nullptr;
  }
  return features.hasInternalFeature(InternalFeatures::DrawBuffersIndexed) ? core : nullptr;
}
} // namespace

void IContext::blendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha) {
  if (blendEquationSeparateiProc_ == nullptr) {
    blendEquationSeparateiProc_ = selectDrawBuffersIndexedProc(deviceFeatureSet_,
                                                               iglBlendEquationSeparatei,
                                                               iglBlendEquationSeparateiOES,
                                                               iglBlendEquationSeparateiEXT);
    IGL_DEBUG_ASSERT(blendEquationSeparateiProc_,
                     "No supported function for glBlendEquationSeparatei\n");
  }
  APILOG("glBlendEquationSeparatei(%u, %s, %s)\n",
         buf,
         GL_ENUM_TO_STRING(modeRGB),
         GL_ENUM_TO_STRING(modeAlpha));
  GLCALL_PROC(blendEquationSeparateiProc_, buf, modeRGB, modeAlpha);
  GLCHECK_ERRORS();
}

void IContext::blendFunc(GLenum sfactor, GLenum dfactor) {
  APILOG("glBlendFunc(%s, %s)\n", GL_ENUM_TO_STRING(sfactor), GL_ENUM_TO_STRING(dfactor));
  GLCALL(BlendFunc)(sfactor, dfactor);
  GLCHECK_ERRORS();
}

void IContext::blendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) {
  APILOG("glBlendFuncSeparate(%s, %s, %s, %s)\n",
         GL_ENUM_TO_STRING(srcRGB),
         GL_ENUM_TO_STRING(dstRGB),
         GL_ENUM_TO_STRING(srcAlpha),
         GL_ENUM_TO_STRING(dstAlpha));
  GLCALL(BlendFuncSeparate)(srcRGB, dstRGB, srcAlpha, dstAlpha);
  GLCHECK_ERRORS();
}

void IContext::blendFuncSeparatei(GLuint buf,
                                  GLenum srcRGB,
                                  GLenum dstRGB,
                                  GLenum srcAlpha,
                                  GLenum dstAlpha) {
  if (blendFuncSeparateiProc_ == nullptr) {
    blendFuncSeparateiProc_ = selectDrawBuffersIndexedProc(deviceFeatureSet_,
                                                           iglBlendFuncSeparatei,
                                                           iglBlendFuncSeparateiOES,
                                                           iglBlendFuncSeparateiEXT);
    IGL_DEBUG_ASSERT(blendFuncSeparateiProc_, "No supported function for glBlendFuncSeparatei\n");
  }
  APILOG("glBlendFuncSeparatei(%u, %s, %s, %s, %s)\n",
         buf,
         GL_ENUM_TO_STRING(srcRGB),
         GL_ENUM_TO_STRING(dstRGB),
         GL_ENUM_TO_STRING(srcAlpha),
         GL_ENUM_TO_STRING(dstAlpha));
  GLCALL_PROC(blendFuncSeparateiProc_, buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
  GLCHECK_ERRORS();
}

void IContext::blitFramebuffer(GLint srcX0,
                               GLint srcY0,
                               GLint srcX1,
                               GLint srcY1,
                               GLint dstX0,
                               GLint dstY0,
                               GLint dstX1,
                               GLint dstY1,
                               GLbitfield mask,
                               GLenum filter) {
  if (blitFramebufferProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::FramebufferBlitExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::FramebufferBlit)) {
        blitFramebufferProc_ = iglBlitFramebufferEXT;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::FramebufferBlit)) {
      blitFramebufferProc_ = iglBlitFramebuffer;
    }
    IGL_DEBUG_ASSERT(blitFramebufferProc_, "No supported function for glBlitFramebuffer\n");
  }
  APILOG(
      "glBlitFramebuffer(%d, %d, %d, %d, %d, %d, %d, %d, 0x%x [%s], %s) read (framebuffer: %u) "
      "write (framebuffer: %u)\n",
      srcX0,
      srcY0,
      srcX1,
      srcY1,
      dstX0,
      dstY0,
      dstX1,
      dstY1,
      mask,
      GL_CLEAR_BITS_TO_STRING(mask),
      GL_ENUM_TO_STRING(filter),
      boundFramebuffer(GL_READ_FRAMEBUFFER),
      boundFramebuffer(GL_DRAW_FRAMEBUFFER));
  GLCALL_PROC(
      blitFramebufferProc_, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
  GLCHECK_ERRORS();
}

void IContext::bufferData(GLenum target,
                          GLsizeiptr size,
                          const GLvoid* IGL_NULLABLE data,
                          GLenum usage) {
  APILOG("glBufferData(%s, %zu, %p, %s) (buffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         size,
         data,
         GL_ENUM_TO_STRING(usage),
         boundBuffer(target));
  GLCALL(BufferData)(target, size, data, usage);
  GLCHECK_ERRORS();
}

void IContext::bufferSubData(GLenum target,
                             GLintptr offset,
                             GLsizeiptr size,
                             const GLvoid* IGL_NULLABLE data) {
  APILOG("glBufferSubData(%s, %zu, %zu, %p) (buffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         offset,
         size,
         data,
         boundBuffer(target));
  GLCALL(BufferSubData)(target, offset, size, data);
  GLCHECK_ERRORS();
}

GLenum IContext::checkFramebufferStatus(GLenum target) {
  GLenum ret = 0;

  IGLCALL_WITH_RETURN(ret, CheckFramebufferStatus)(target);
  // NOTE: Must log after call due to return value
  APILOG("glCheckFramebufferStatus(%s) = %s (framebuffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         GL_ENUM_TO_STRING(ret),
         boundFramebuffer(target));
  GLCHECK_ERRORS();
  return ret;
}

void IContext::clear(GLbitfield mask) {
  APILOG("glClear(0x%x [%s]) (framebuffer: %u) %s\n",
         mask,
         GL_CLEAR_BITS_TO_STRING(mask),
         boundFramebuffer(GL_DRAW_FRAMEBUFFER),
         affectedFramebufferAttachments(GL_DRAW_FRAMEBUFFER, mask).c_str());
  GLCALL(Clear)(mask);
  GLCHECK_ERRORS();
}

void IContext::clearBufferfv(GLenum buffer, GLint drawBuffer, const GLfloat* value) {
  if (!deviceFeatureSet_.hasInternalFeature(InternalFeatures::ClearBufferfv)) {
    IGL_DEBUG_ASSERT(0, "No supported function for glClearBufferfv\n");
    return;
  }

  APILOG("glClearBufferfv(%s, %d, %f) (framebuffer: %u) (%s: %u)\n",
         GL_ENUM_TO_STRING(buffer),
         drawBuffer,
         *value,
         boundFramebuffer(GL_DRAW_FRAMEBUFFER),
         framebufferAttachmentType(GL_DRAW_FRAMEBUFFER, buffer, drawBuffer),
         framebufferAttachment(GL_DRAW_FRAMEBUFFER, buffer, drawBuffer));
  IGLCALL(ClearBufferfv)(buffer, drawBuffer, value);
  GLCHECK_ERRORS();
}

void IContext::clearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
  APILOG("glClearColor(%f, %f, %f, %f)\n", red, green, blue, alpha);
  GLCALL(ClearColor)(red, green, blue, alpha);
  GLCHECK_ERRORS();
}

void IContext::clearDepthf(GLfloat depth) {
  if (clearDepthfProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::ClearDepthf)) {
      clearDepthfProc_ = iglClearDepthf;
    } else {
      clearDepthfProc_ = iglClearDepth;
    }
    IGL_DEBUG_ASSERT(clearDepthfProc_, "No supported function for glClearDepthf\n");
  }

  APILOG("glClearDepthf(%f)\n", depth);
  GLCALL_PROC(clearDepthfProc_, depth);
  GLCHECK_ERRORS();
}

void IContext::clearStencil(GLint s) {
  APILOG("glClearStencil(%d)\n", s);
  GLCALL(ClearStencil)(s);
  GLCHECK_ERRORS();
}

GLenum IContext::clientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeoutNs) {
  if (clientWaitSyncProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::Sync)) {
      clientWaitSyncProc_ = iglClientWaitSync;
    }
    IGL_DEBUG_ASSERT(clientWaitSyncProc_, "No supported function for glClientWaitSync\n");
  }

  GLenum ret = GL_WAIT_FAILED;
  GLCALL_PROC_WITH_RETURN(ret, clientWaitSyncProc_, GL_WAIT_FAILED, sync, flags, timeoutNs);
  APILOG("glClientWaitSync(%p, %u, %llu) = %s (sync: %p)\n",
         sync,
         flags,
         timeoutNs,
         GL_ENUM_TO_STRING(ret),
         sync);
  GLCHECK_ERRORS();
  return ret;
}

void IContext::colorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
  APILOG("glColorMask(%s, %s, %s, %s) (framebuffer: %u)\n",
         GL_BOOL_TO_STRING(red),
         GL_BOOL_TO_STRING(green),
         GL_BOOL_TO_STRING(blue),
         GL_BOOL_TO_STRING(alpha),
         boundFramebuffer(GL_DRAW_FRAMEBUFFER));
  GLCALL(ColorMask)(red, green, blue, alpha);
  GLCHECK_ERRORS();
}

void IContext::colorMaski(GLuint index,
                          GLboolean red,
                          GLboolean green,
                          GLboolean blue,
                          GLboolean alpha) {
  if (colorMaskiProc_ == nullptr) {
    colorMaskiProc_ = selectDrawBuffersIndexedProc(
        deviceFeatureSet_, iglColorMaski, iglColorMaskiOES, iglColorMaskiEXT);
    IGL_DEBUG_ASSERT(colorMaskiProc_, "No supported function for glColorMaski\n");
  }
  APILOG("glColorMaski(%u, %s, %s, %s, %s) (framebuffer: %u)\n",
         index,
         GL_BOOL_TO_STRING(red),
         GL_BOOL_TO_STRING(green),
         GL_BOOL_TO_STRING(blue),
         GL_BOOL_TO_STRING(alpha),
         boundFramebuffer(GL_DRAW_FRAMEBUFFER));
  GLCALL_PROC(colorMaskiProc_, index, red, green, blue, alpha);
  GLCHECK_ERRORS();
}

void IContext::compileShader(GLuint shader) {
  APILOG("glCompileShader(%u)\n", shader);
  GLCALL(CompileShader)(shader);
  GLCHECK_ERRORS();
  shaderCompilationCount_++;
}

void IContext::compressedTexImage2D(GLenum target,
                                    GLint level,
                                    GLenum internalformat,
                                    GLsizei width,
                                    GLsizei height,
                                    GLint border,
                                    GLsizei imageSize,
                                    const GLvoid* IGL_NULLABLE data) {
  APILOG("glCompressedTexImage2D(%s, %d, %s, %u, %u, %d, %u, %p) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         border,
         imageSize,
         data,
         boundTexture(target));
  GLCALL(CompressedTexImage2D)
  (target, level, internalformat, width, height, border, imageSize, data);
  GLCHECK_ERRORS();
}

void IContext::compressedTexImage3D(GLenum target,
                                    GLint level,
                                    GLenum internalformat,
                                    GLsizei width,
                                    GLsizei height,
                                    GLsizei depth,
                                    GLint border,
                                    GLsizei imageSize,
                                    const GLvoid* data) {
  if (compressedTexImage3DProc_ == nullptr) {
    if (deviceFeatureSet_.hasFeature(DeviceFeatures::Texture3D)) {
      if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::Texture3DExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::Texture3D)) {
          compressedTexImage3DProc_ = iglCompressedTexImage3DOES;
        }
      } else {
        compressedTexImage3DProc_ = iglCompressedTexImage3D;
      }
    }
    IGL_DEBUG_ASSERT(compressedTexImage3DProc_,
                     "No supported function for glCompressedTexImage3D\n");
  }
  APILOG("glCompressedTexImage3D(%s, %d, %s, %u, %u, %u, %d, %u, %p) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         depth,
         border,
         imageSize,
         data,
         boundTexture(target));
  GLCALL_PROC(compressedTexImage3DProc_,
              target,
              level,
              internalformat,
              width,
              height,
              depth,
              border,
              imageSize,
              data);
  GLCHECK_ERRORS();
}

void IContext::compressedTexSubImage2D(GLenum target,
                                       GLint level,
                                       GLint xoffset,
                                       GLint yoffset,
                                       GLsizei width,
                                       GLsizei height,
                                       GLenum format,
                                       GLsizei imageSize,
                                       const GLvoid* data) {
  APILOG("glCompressedTexSubImage2D(%s, %d, %d, %d, %u, %u, %s, %u, %p) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         xoffset,
         yoffset,
         width,
         height,
         GL_ENUM_TO_STRING(format),
         imageSize,
         data,
         boundTexture(target));
  GLCALL(CompressedTexSubImage2D)
  (target, level, xoffset, yoffset, width, height, format, imageSize, data);
  GLCHECK_ERRORS();
}

void IContext::compressedTexSubImage3D(GLenum target,
                                       GLint level,
                                       GLint xoffset,
                                       GLint yoffset,
                                       GLint zoffset,
                                       GLsizei width,
                                       GLsizei height,
                                       GLsizei depth,
                                       GLenum format,
                                       GLsizei imageSize,
                                       const GLvoid* data) {
  if (compressedTexSubImage3DProc_ == nullptr) {
    if (deviceFeatureSet_.hasFeature(DeviceFeatures::Texture3D)) {
      if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::Texture3DExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::Texture3D)) {
          compressedTexSubImage3DProc_ = iglCompressedTexSubImage3DOES;
        }
      } else {
        compressedTexSubImage3DProc_ = iglCompressedTexSubImage3D;
      }
    }
    IGL_DEBUG_ASSERT(compressedTexSubImage3DProc_,
                     "No supported function for glCompressedTexSubImage3D\n");
  }

  APILOG("glCompressedTexSubImage3D(%s, %d, %d, %d, %d, %u, %u, %u, %s, %u, %p) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         xoffset,
         yoffset,
         zoffset,
         width,
         height,
         depth,
         GL_ENUM_TO_STRING(format),
         imageSize,
         data,
         boundTexture(target));
  GLCALL_PROC(compressedTexSubImage3DProc_,
              target,
              level,
              xoffset,
              yoffset,
              zoffset,
              width,
              height,
              depth,
              format,
              imageSize,
              data);
  GLCHECK_ERRORS();
}

void IContext::copyBufferSubData(GLenum readtarget,
                                 GLenum writetarget,
                                 GLintptr readoffset,
                                 GLintptr writeoffset,
                                 GLsizeiptr size) {
#if IGL_PLATFORM_APPLE || IGL_PLATFORM_ANDROID
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
#else
  APILOG("glCopyBufferSubData(%s, %s, %d, %d, %u) read (buffer: %u) write (buffer: %u)\n",
         GL_ENUM_TO_STRING(readtarget),
         GL_ENUM_TO_STRING(writetarget),
         readoffset,
         writeoffset,
         size,
         boundBuffer(readTarget),
         boundBuffer(writeTarget));
  GLCALL(CopyBufferSubData)(readtarget, writetarget, readoffset, writeoffset, size);
  GLCHECK_ERRORS();
#endif // IGL_PLATFORM_APPLE
}

void IContext::copyTexSubImage2D(GLenum target,
                                 GLint level,
                                 GLint xoffset,
                                 GLint yoffset,
                                 GLint x,
                                 GLint y,
                                 GLsizei width,
                                 GLsizei height) {
  APILOG("glCopyTexSubImage2D(%s, %d, %d, %d, %d, %d, %u, %u) read (%s: %u) write (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         xoffset,
         yoffset,
         x,
         y,
         width,
         height,
         framebufferReadBufferType(GL_READ_FRAMEBUFFER),
         framebufferReadBuffer(GL_READ_FRAMEBUFFER));
  GLCALL(CopyTexSubImage2D)(target, level, xoffset, yoffset, x, y, width, height);
  GLCHECK_ERRORS();
}

void IContext::createMemoryObjects(GLsizei n, GLuint* objects) {
  APILOG("glCreateMemoryObjectsEXT(%u, %p) (memory object: %u)\n",
         n,
         objects,
         objects == nullptr ? 0 : *objects);
  IGLCALL(CreateMemoryObjectsEXT)(n, objects);
  GLCHECK_ERRORS();
}

GLuint IContext::createProgram() {
  GLuint ret = 0;

  GLCALL_WITH_RETURN(ret, CreateProgram)();
  // NOTE: Must log after call due to return value
  APILOG("glCreateProgram() (program: %u)\n", ret);
  GLCHECK_ERRORS();

#if IGL_API_LOG
  programUniforms_[ret] = {};
  programUniformsByLocation_[ret] = {};
  programSsbosByIndex_[ret] = {};
  programSsbosByLocation_[ret] = {};
  programUbosByIndex_[ret] = {};
  programUbosByLocation_[ret] = {};
#endif
  return ret;
}

GLuint IContext::createShader(GLenum shaderType) {
  GLuint ret = 0;

  GLCALL_WITH_RETURN(ret, CreateShader)(shaderType);
  // NOTE: Must log after call due to return value
  APILOG("glCreateShader(%s) = %u\n", GL_ENUM_TO_STRING(shaderType), ret);
  GLCHECK_ERRORS();

  return ret;
}

void IContext::cullFace(GLint mode) {
  APILOG("glCullFace(%s)\n", GL_ENUM_TO_STRING(mode));
  GLCALL(CullFace)(mode);
  GLCHECK_ERRORS();
}

void IContext::debugMessageCallback(PFNIGLDEBUGPROC callback, const void* userParam) {
  if (debugMessageCallbackProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessageCallback)) {
      if (deviceFeatureSet_.hasInternalRequirement(
              InternalRequirement::DebugMessageCallbackExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::Debug)) {
          debugMessageCallbackProc_ = iglDebugMessageCallbackKHR;
        }
      } else {
        debugMessageCallbackProc_ = iglDebugMessageCallback;
      }
    }
    IGL_DEBUG_ASSERT(debugMessageCallbackProc_,
                     "No supported function for glDebugMessageCallback\n");
  }

  APILOG("glDebugMessageCallback(%p, %p)\n", callback, userParam);
  GLCALL_PROC(debugMessageCallbackProc_, callback, userParam);
  GLCHECK_ERRORS();
}

void IContext::debugMessageInsert(GLenum source,
                                  GLenum type,
                                  GLuint id,
                                  GLenum severity,
                                  GLsizei length,
                                  const GLchar* buf) {
  if (debugMessageInsertProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessage)) {
      if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::DebugMessageExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::Debug)) {
          debugMessageInsertProc_ = iglDebugMessageInsertKHR;
        } else if (deviceFeatureSet_.hasExtension(Extensions::DebugMarker)) {
          debugMessageInsertProc_ = iglInsertEventMarkerEXT;
        }
      } else {
        debugMessageInsertProc_ = iglDebugMessageInsert;
      }
    }
    IGL_DEBUG_ASSERT(debugMessageInsertProc_, "No supported function for glDebugMessageInsert\n");
  }

  APILOG("glDebugMessageInsert(%s, %s, %u, %s, %u, %s)\n",
         GL_ENUM_TO_STRING(source),
         GL_ENUM_TO_STRING(type),
         id,
         GL_ENUM_TO_STRING(severity),
         length,
         buf);
  GLCALL_PROC(debugMessageInsertProc_, source, type, id, severity, length, buf);
  GLCHECK_ERRORS();
}

void IContext::deleteBuffers(GLsizei n, const GLuint* buffers) {
  if (isDestructionAllowed() && IGL_DEBUG_VERIFY(buffers != nullptr)) {
    if (shouldQueueAPI()) {
      deletionQueues_.queueDeleteBuffers(n, buffers);
    } else {
      APILOG(
          "glDeleteBuffers(%u, %p) (buffer: %u)\n", n, buffers, buffers == nullptr ? 0 : *buffers);
      GLCALL(DeleteBuffers)(n, buffers);
      GLCHECK_ERRORS();
    }
  }
}

void IContext::deleteMemoryObjects(GLsizei n, const GLuint* objects) {
  APILOG("glDeleteMemoryObjectsEXT(%u, %p) (memory object: %u)\n",
         n,
         objects,
         objects == nullptr ? 0 : *objects);
  IGLCALL(DeleteMemoryObjectsEXT)(n, objects);
  GLCHECK_ERRORS();
}

void IContext::unbindBuffer(GLenum target) {
  if (shouldQueueAPI()) {
    deletionQueues_.queueUnbindBuffer(target);
  } else {
    bindBuffer(target, 0);
  }
}

void IContext::deleteFramebuffers(GLsizei n, const GLuint* framebuffers) {
  if (isDestructionAllowed() && IGL_DEBUG_VERIFY(framebuffers != nullptr)) {
    if (shouldQueueAPI()) {
      deletionQueues_.queueDeleteFramebuffers(n, framebuffers);
    } else {
      APILOG("glDeleteFramebuffers(%u, %p) (framebuffer: %u)\n",
             n,
             framebuffers,
             framebuffers == nullptr ? 0 : *framebuffers);
      IGLCALL(DeleteFramebuffers)(n, framebuffers);
      GLCHECK_ERRORS();
    }
  }
}

void IContext::deleteProgram(GLuint program) {
  if (isDestructionAllowed()) {
    if (shouldQueueAPI()) {
      deletionQueues_.queueDeleteProgram(program);
    } else {
      APILOG("glDeleteProgram(%u) (program: %u)\n", program, program);
      GLCALL(DeleteProgram)(program);
      GLCHECK_ERRORS();
    }
  }
}

void IContext::deleteRenderbuffers(GLsizei n, const GLuint* renderbuffers) {
  if (isDestructionAllowed() && IGL_DEBUG_VERIFY(renderbuffers != nullptr)) {
    if (shouldQueueAPI()) {
      deletionQueues_.queueDeleteRenderbuffers(n, renderbuffers);
    } else {
      APILOG("glDeleteRenderbuffers(%u, %p) (renderbuffer: %u)\n",
             n,
             renderbuffers,
             renderbuffers == nullptr ? 0 : *renderbuffers);
      IGLCALL(DeleteRenderbuffers)(n, renderbuffers);
      GLCHECK_ERRORS();
    }
  }
}

void IContext::deleteVertexArrays(GLsizei n, const GLuint* vertexArrays) {
  if (deleteVertexArraysProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::VertexArrayObjectExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::VertexArrayObject)) {
        deleteVertexArraysProc_ = iglDeleteVertexArraysOES;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::VertexArrayObject)) {
      deleteVertexArraysProc_ = iglDeleteVertexArrays;
    }
    IGL_DEBUG_ASSERT(deleteVertexArraysProc_, "No supported function for glDeleteVertexArrays\n");
  }
  if (isDestructionAllowed() && IGL_DEBUG_VERIFY(vertexArrays != nullptr)) {
    if (shouldQueueAPI()) {
      deletionQueues_.queueDeleteVertexArrays(n, vertexArrays);
    } else {
      APILOG("glDeleteVertexArrays(%u, %p) (VAO: %u)\n",
             n,
             vertexArrays,
             vertexArrays == nullptr ? 0 : *vertexArrays);
      GLCALL_PROC(deleteVertexArraysProc_, n, vertexArrays);
      GLCHECK_ERRORS();
    }
  }
}

void IContext::deleteShader(GLuint shaderId) {
  if (isDestructionAllowed()) {
    if (shouldQueueAPI()) {
      deletionQueues_.queueDeleteShader(shaderId);
    } else {
      APILOG("glDeleteShader(%u)\n", shaderId);
      GLCALL(DeleteShader)(shaderId);
      GLCHECK_ERRORS();
    }
  }
}

void IContext::deleteSync(GLsync sync) {
  if (deleteSyncProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::SyncExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::Sync)) {
        deleteSyncProc_ = iglDeleteSyncAPPLE;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::Sync)) {
      deleteSyncProc_ = iglDeleteSync;
    }
    IGL_DEBUG_ASSERT(deleteSyncProc_, "No supported function for glDeleteSync\n");
  }

  APILOG("glDeleteSync(%p) (sync: %p)\n", sync, sync);
  GLCALL_PROC(deleteSyncProc_, sync);
  GLCHECK_ERRORS();
}

void IContext::deleteTextures(GLsizei n, const GLuint* textures) {
  if (isDestructionAllowed() && IGL_DEBUG_VERIFY(textures != nullptr)) {
    if (shouldQueueAPI()) {
      deletionQueues_.queueDeleteTextures(n, textures);
    } else {
      APILOG("glDeleteTextures(%u, %p) (texture: %u)\n",
             n,
             textures,
             textures == nullptr ? 0 : *textures);
      GLCALL(DeleteTextures)(n, textures);
      GLCHECK_ERRORS();
    }
  }
}

void IContext::depthFunc(GLenum func) {
  APILOG("glDepthFunc(%s)\n", GL_ENUM_TO_STRING(func));
  GLCALL(DepthFunc)(func);
  GLCHECK_ERRORS();
}

void IContext::depthMask(GLboolean flag) {
  APILOG("glDepthMask(%s)\n", GL_BOOL_TO_STRING(flag));
  GLCALL(DepthMask)(flag);
  GLCHECK_ERRORS();
}

void IContext::depthRangef(GLfloat n, GLfloat f) {
  APILOG("glDepthRangef(%f, %f)\n", n, f);
  GLCALL(DepthRangef)(n, f);
  GLCHECK_ERRORS();
}

void IContext::detachShader(GLuint program, GLuint shader) {
  APILOG("glDetachShader(%u, %u) (program: %u)\n", program, shader, program);
  GLCALL(DetachShader)(program, shader);
  GLCHECK_ERRORS();
}

void IContext::disable(GLenum cap) {
  APILOG("glDisable(%s)\n", GL_ENUM_TO_STRING(cap));
  GLCALL(Disable)(cap);
  GLCHECK_ERRORS();
}

void IContext::disablei(GLenum cap, GLuint index) {
  if (disableiProc_ == nullptr) {
    disableiProc_ = selectDrawBuffersIndexedProc(
        deviceFeatureSet_, iglDisablei, iglDisableiOES, iglDisableiEXT);
    IGL_DEBUG_ASSERT(disableiProc_, "No supported function for glDisablei\n");
  }
  APILOG("glDisablei(%s, %u)\n", GL_ENUM_TO_STRING(cap), index);
  GLCALL_PROC(disableiProc_, cap, index);
  GLCHECK_ERRORS();
}

void IContext::disableVertexAttribArray(GLuint index) {
#if IGL_API_LOG
  if (index < enabledAttributes_.size()) {
    enabledAttributes_.reset(index);
  }
#endif
  APILOG("glDisableVertexAttribArray(%u) (VAO: %u) (buffer: %u\n",
         index,
         boundVao_,
         index < boundAttributes_.size() ? boundAttributes_[index] : 0);
  GLCALL(DisableVertexAttribArray)(index);
  GLCHECK_ERRORS();
}

void IContext::drawArrays(GLenum mode, GLint first, GLsizei count) {
#if IGL_API_LOG
  lastCommandWasCompute_ = false;
#endif
  drawCallCount_++;

  IGL_PROFILER_ZONE_GPU_OGL("drawArrays()");

  APILOG("glDrawArrays(%s, %d, %u) (program: %u) (VAO: %u) (framebuffer: %u) %s %s %s %s\n",
         GL_ENUM_TO_STRING(mode),
         first,
         count,
         boundProgram_,
         boundVao_,
         boundFramebuffer(GL_DRAW_FRAMEBUFFER),
         boundDrawBuffers().c_str(),
         boundDrawTextures().c_str(),
         boundFramebufferAttachments(GL_DRAW_FRAMEBUFFER).c_str(),
         boundBuffersByIndex().c_str());
  GLCALL(DrawArrays)(mode, first, count);
  GLCHECK_ERRORS();
  APILOG_DEC_DRAW_COUNT();
}

void IContext::drawArraysIndirect(GLenum mode, const GLvoid* indirect) {
#if IGL_API_LOG
  lastCommandWasCompute_ = false;
#endif
  drawCallCount_++;

  IGL_PROFILER_ZONE_GPU_COLOR_OGL("drawArraysIndirect()", IGL_PROFILER_COLOR_DRAW);

  APILOG(
      "glDrawArraysIndirect(%s, %p) (program: %u) (VAO: %u) (framebuffer: %u) indirect (buffer: "
      "%u) %s %s %s %s\n",
      GL_ENUM_TO_STRING(mode),
      indirect,
      boundProgram_,
      boundVao_,
      boundFramebuffer(GL_DRAW_FRAMEBUFFER),
      boundBuffer(GL_DRAW_INDIRECT_BUFFER),
      boundDrawBuffers().c_str(),
      boundDrawTextures().c_str(),
      boundFramebufferAttachments(GL_DRAW_FRAMEBUFFER).c_str(),
      boundBuffersByIndex().c_str());
  IGLCALL(DrawArraysIndirect)(mode, indirect);
  GLCHECK_ERRORS();
  APILOG_DEC_DRAW_COUNT();
}

void IContext::drawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
#if IGL_API_LOG
  lastCommandWasCompute_ = false;
#endif
  drawCallCount_++;

  IGL_PROFILER_ZONE_GPU_OGL("drawArraysInstanced()");

  APILOG(
      "glDrawArraysInstanced(%s, %d, %u, %u) (program: %u) (VAO: %u) (framebuffer: %u) %s %s %s "
      "%s\n",
      GL_ENUM_TO_STRING(mode),
      first,
      count,
      instancecount,
      boundProgram_,
      boundVao_,
      boundFramebuffer(GL_DRAW_FRAMEBUFFER),
      boundDrawBuffers().c_str(),
      boundDrawTextures().c_str(),
      boundFramebufferAttachments(GL_DRAW_FRAMEBUFFER).c_str(),
      boundBuffersByIndex().c_str());
  IGLCALL(DrawArraysInstanced)(mode, first, count, instancecount);
  GLCHECK_ERRORS();
  APILOG_DEC_DRAW_COUNT();
}

void IContext::drawBuffers(GLsizei n, GLenum* buffers) {
  if (drawBuffersProc_ == nullptr) {
    if (deviceFeatureSet_.hasFeature(DeviceFeatures::MultipleRenderTargets)) {
      if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::DrawBuffersExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::DrawBuffers)) {
          drawBuffersProc_ = iglDrawBuffersEXT;
        }
      } else {
        drawBuffersProc_ = iglDrawBuffers;
      }
    }
    IGL_DEBUG_ASSERT(drawBuffersProc_, "No supported function for glDrawBuffers\n");
  }
  IGL_PROFILER_ZONE_GPU_COLOR_OGL("drawBuffers()", IGL_PROFILER_COLOR_DRAW);

  APILOG("glDrawBuffers(%u, %u)\n", n, buffers);
  GLCALL_PROC(drawBuffersProc_, n, buffers);
  GLCHECK_ERRORS();
}

void IContext::drawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid* indices) {
#if IGL_API_LOG
  lastCommandWasCompute_ = false;
#endif
  drawCallCount_++;

  IGL_PROFILER_ZONE_GPU_COLOR_OGL("drawElements()", IGL_PROFILER_COLOR_DRAW);

  APILOG(
      "glDrawElements(%s, %u, %s, %p) (program: %u) (VAO: %u) (framebuffer: %u) element (buffer: "
      "%u) %s %s %s %s\n",
      GL_ENUM_TO_STRING(mode),
      count,
      GL_ENUM_TO_STRING(type),
      indices,
      boundProgram_,
      boundVao_,
      boundFramebuffer(GL_DRAW_FRAMEBUFFER),
      boundBuffer(GL_ELEMENT_ARRAY_BUFFER),
      boundDrawBuffers().c_str(),
      boundDrawTextures().c_str(),
      boundFramebufferAttachments(GL_DRAW_FRAMEBUFFER).c_str(),
      boundBuffersByIndex().c_str());
  GLCALL(DrawElements)(mode, count, type, indices);
  GLCHECK_ERRORS();
  APILOG_DEC_DRAW_COUNT();
}

void IContext::drawElementsInstanced(GLenum mode,
                                     GLsizei count,
                                     GLenum type,
                                     const GLvoid* indices,
                                     GLsizei instancecount) {
#if IGL_API_LOG
  lastCommandWasCompute_ = false;
#endif
  drawCallCount_++;

  IGL_PROFILER_ZONE_GPU_COLOR_OGL("drawElementsInstanced()", IGL_PROFILER_COLOR_DRAW);

  APILOG(
      "glDrawElementsInstanced(%s, %u, %s, %p, %u) (program: %u) (VAO: %u) (framebuffer: %u) "
      "element (buffer: %u) %s %s %s %s\n",
      GL_ENUM_TO_STRING(mode),
      count,
      GL_ENUM_TO_STRING(type),
      indices,
      instancecount,
      boundProgram_,
      boundVao_,
      boundFramebuffer(GL_DRAW_FRAMEBUFFER),
      boundBuffer(GL_ELEMENT_ARRAY_BUFFER),
      boundDrawBuffers().c_str(),
      boundDrawTextures().c_str(),
      boundFramebufferAttachments(GL_DRAW_FRAMEBUFFER).c_str(),
      boundBuffersByIndex().c_str());
  IGLCALL(DrawElementsInstanced)(mode, count, type, indices, instancecount);
  GLCHECK_ERRORS();
  APILOG_DEC_DRAW_COUNT();
}

void IContext::drawElementsIndirect(GLenum mode, GLenum type, const GLvoid* indirect) {
#if IGL_API_LOG
  lastCommandWasCompute_ = false;
#endif
  drawCallCount_++;

  IGL_PROFILER_ZONE_GPU_COLOR_OGL("drawElementsIndirect()", IGL_PROFILER_COLOR_DRAW);

  APILOG(
      "glDrawElementsIndirect(%s, %s, %p) (program: %u) (VAO: %u) (framebuffer: %u) indirect "
      "(buffer: %u) element (buffer: %u) %s %s %s %s\n",
      GL_ENUM_TO_STRING(mode),
      GL_ENUM_TO_STRING(type),
      indirect,
      boundProgram_,
      boundVao_,
      boundFramebuffer(GL_DRAW_FRAMEBUFFER),
      boundBuffer(GL_DRAW_INDIRECT_BUFFER),
      boundBuffer(GL_ELEMENT_ARRAY_BUFFER),
      boundDrawBuffers().c_str(),
      boundDrawTextures().c_str(),
      boundFramebufferAttachments(GL_DRAW_FRAMEBUFFER).c_str(),
      boundBuffersByIndex().c_str());
  IGLCALL(DrawElementsIndirect)(mode, type, indirect);
  GLCHECK_ERRORS();
  APILOG_DEC_DRAW_COUNT();
}

void IContext::multiDrawArraysIndirect(GLenum mode,
                                       const void* indirect,
                                       GLsizei drawcount,
                                       GLsizei stride) {
#if IGL_API_LOG
  lastCommandWasCompute_ = false;
#endif
  drawCallCount_ += drawcount;

  IGL_PROFILER_ZONE_GPU_COLOR_OGL("multiDrawArraysIndirect()", IGL_PROFILER_COLOR_DRAW);

  APILOG("glMultiDrawArraysIndirect(%s, %p, %d, %d)\n",
         GL_ENUM_TO_STRING(mode),
         indirect,
         drawcount,
         stride);
  // `indirect` is a byte offset into the bound GL_DRAW_INDIRECT_BUFFER; offset 0
  // (a null pointer) is valid, so do not assert it is non-null.
  IGLCALL(MultiDrawArraysIndirect)(mode, indirect, drawcount, stride);
  GLCHECK_ERRORS();
  APILOG_DEC_DRAW_COUNT();
}

void IContext::multiDrawElementsIndirect(GLenum mode,
                                         GLenum type,
                                         const void* indirect,
                                         GLsizei drawcount,
                                         GLsizei stride) {
#if IGL_API_LOG
  lastCommandWasCompute_ = false;
#endif
  drawCallCount_ += drawcount;

  IGL_PROFILER_ZONE_GPU_COLOR_OGL("multiDrawElementsIndirect()", IGL_PROFILER_COLOR_DRAW);

  APILOG("glMultiDrawElementsIndirect(%s, %s, %p, %d, %d)\n",
         GL_ENUM_TO_STRING(mode),
         GL_ENUM_TO_STRING(type),
         indirect,
         drawcount,
         stride);
  // `indirect` is a byte offset into the bound GL_DRAW_INDIRECT_BUFFER; offset 0
  // (a null pointer) is valid, so do not assert it is non-null.
  IGLCALL(MultiDrawElementsIndirect)(mode, type, indirect, drawcount, stride);
  GLCHECK_ERRORS();
  APILOG_DEC_DRAW_COUNT();
}

void IContext::enable(GLenum cap) {
  APILOG("glEnable(%s)\n", GL_ENUM_TO_STRING(cap));
  GLCALL(Enable)(cap);
  GLCHECK_ERRORS();
}

void IContext::enablei(GLenum cap, GLuint index) {
  if (enableiProc_ == nullptr) {
    enableiProc_ =
        selectDrawBuffersIndexedProc(deviceFeatureSet_, iglEnablei, iglEnableiOES, iglEnableiEXT);
    IGL_DEBUG_ASSERT(enableiProc_, "No supported function for glEnablei\n");
  }
  APILOG("glEnablei(%s, %u)\n", GL_ENUM_TO_STRING(cap), index);
  GLCALL_PROC(enableiProc_, cap, index);
  GLCHECK_ERRORS();
}

void IContext::enableVertexAttribArray(GLuint index) {
#if IGL_API_LOG
  if (index < enabledAttributes_.size()) {
    enabledAttributes_.set(index);
  }
#endif
  APILOG("glEnableVertexAttribArray(%u) (VAO: %u) (buffer: %u)\n",
         index,
         boundVao_,
         index < boundAttributes_.size() ? boundAttributes_[index] : 0);
  GLCALL(EnableVertexAttribArray)(index);
  GLCHECK_ERRORS();
}

GLsync IContext::fenceSync(GLenum condition, GLbitfield flags) {
  if (fenceSyncProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::SyncExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::Sync)) {
        fenceSyncProc_ = iglFenceSyncAPPLE;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::Sync)) {
      fenceSyncProc_ = iglFenceSync;
    }
    IGL_DEBUG_ASSERT(fenceSyncProc_, "No supported function for glFenceSync\n");
  }

  GLsync sync = nullptr;

  APILOG("glFenceSync(%s, %u) (sync: %p)\n", GL_ENUM_TO_STRING(condition), flags, sync);
  GLCALL_PROC_WITH_RETURN(sync, fenceSyncProc_, GL_ZERO, condition, flags);
  GLCHECK_ERRORS();

  return sync;
}

void IContext::finish() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);
  APILOG("glFinish\n");
  GLCALL(Finish)();
  GLCHECK_ERRORS();
}

void IContext::flush() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);
  APILOG("glFlush\n");
  GLCALL(Flush)();
  GLCHECK_ERRORS();
}

void IContext::framebufferRenderbuffer(GLenum target,
                                       GLenum attachment,
                                       GLenum renderbuffertarget,
                                       GLuint renderbuffer) {
#if IGL_API_LOG
  const char* unboundAttachmentType = framebufferAttachmentType(target, attachment);
  GLuint unboundAttachment = framebufferAttachment(target, attachment);
  setFramebufferAttachment(target, attachment, true, renderbuffer);
#endif
  APILOG(
      "glFramebufferRenderbuffer(%s, %s, %s, %u) (framebuffer: %u) unbound (%s: %u) bound "
      "(renderbuffer: %u)\n",
      GL_ENUM_TO_STRING(target),
      GL_ENUM_TO_STRING(attachment),
      GL_ENUM_TO_STRING(renderbuffertarget),
      boundFramebuffer(target),
      unboundAttachmentType,
      unboundAttachment,
      renderbuffer);
  IGLCALL(FramebufferRenderbuffer)(target, attachment, renderbuffertarget, renderbuffer);
  GLCHECK_ERRORS();
}

void IContext::framebufferTexture2D(GLenum target,
                                    GLenum attachment,
                                    GLenum textarget,
                                    GLuint texture,
                                    GLint level) {
#if IGL_API_LOG
  const char* unboundAttachmentType = framebufferAttachmentType(target, attachment);
  GLuint unboundAttachment = framebufferAttachment(target, attachment);
  setFramebufferAttachment(target, attachment, false, texture);
#endif
  APILOG(
      "glFramebufferTexture2D(%s, %s, %s, %u, %d) (framebuffer: %u) unbound (%s: %u) bound "
      "(texture: %u)\n",
      GL_ENUM_TO_STRING(target),
      GL_ENUM_TO_STRING(attachment),
      GL_ENUM_TO_STRING(textarget),
      texture,
      level,
      boundFramebuffer(target),
      unboundAttachmentType,
      unboundAttachment,
      texture);
  IGLCALL(FramebufferTexture2D)(target, attachment, textarget, texture, level);
  GLCHECK_ERRORS();
}

// NOLINTNEXTLINE(misc-no-recursion)
void IContext::framebufferTexture2DMultisample(GLenum target,
                                               GLenum attachment,
                                               GLenum textarget,
                                               GLuint texture,
                                               GLint level,
                                               GLsizei samples) {
  if (multisampleNeedsFramebufferTarget_ &&
      (target == GL_DRAW_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER)) {
    target = GL_FRAMEBUFFER;
  }
#if IGL_API_LOG
  const char* unboundAttachmentType = framebufferAttachmentType(target, attachment);
  GLuint unboundAttachment = framebufferAttachment(target, attachment);
  setFramebufferAttachment(target, attachment, false, texture);
#endif
  if (framebufferTexture2DMultisampleProc_ == nullptr) {
    // Use runtime checks to determine which of several potential methods are supported by the
    // context.
    if (deviceFeatureSet_.hasExtension(Extensions::MultiSampleExt)) {
      framebufferTexture2DMultisampleProc_ = iglFramebufferTexture2DMultisampleEXT;
    } else if (deviceFeatureSet_.hasExtension(Extensions::MultiSampleImg)) {
      framebufferTexture2DMultisampleProc_ = iglFramebufferTexture2DMultisampleIMG;
    }
    IGL_DEBUG_ASSERT(framebufferTexture2DMultisampleProc_,
                     "No supported function for glFramebufferTexture2DMultisample\n");
  }

  if (framebufferTexture2DMultisampleProc_ != nullptr) {
    samples = std::min(samples, getMaxMultisampledRenderToTextureSamples());
  }

  // Certain drivers (I am looking at you Adreno) misbehave according to spec,
  // where GL_DRAW_FRAMEBUFFER is perfectly valid framebuffer target, but it only recognizes
  // GL_FRAMEBUFFER meaning that it behaves as if it is `IMG_multisampled_render_to_texture` not
  // `EXT_multisampled_render_to_texture`. The first call with such a target finds out which kind
  // of driver this is, so that glGetError() is not needed after every attachment.
  const bool verifiesTarget = (target == GL_DRAW_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) &&
                              !multisampleDrawTargetVerified_ &&
                              deviceFeatureSet_.hasExtension(Extensions::MultiSampleExt);
  if (verifiesTarget) {
    // An error that an earlier call left behind must not look like the driver rejecting the target
    constexpr int kMaxPendingErrors = 8;
    for (int i = 0; i < kMaxPendingErrors && getError() != GL_NO_ERROR; ++i) {
    }
  }

  APILOG(
      "glFramebufferTexture2DMultisample(%s, %s, %s, %u, %d, %u) (framebuffer: %u) unbound (%s: "
      "%u) bound (texture: %u)\n",
      GL_ENUM_TO_STRING(target),
      GL_ENUM_TO_STRING(attachment),
      GL_ENUM_TO_STRING(textarget),
      texture,
      level,
      samples,
      boundFramebuffer(target),
      unboundAttachmentType,
      unboundAttachment,
      texture);
  GLCALL_PROC(
      framebufferTexture2DMultisampleProc_, target, attachment, textarget, texture, level, samples);

  if (verifiesTarget || alwaysCheckError_) {
    const GLenum error = getError();
    if (verifiesTarget && error == GL_INVALID_ENUM) {
      multisampleNeedsFramebufferTarget_ = true;
      // Repeat again, now using explicitly GL_FRAMEBUFFER not GL_DRAW/GL_READ.
      framebufferTexture2DMultisample(
          GL_FRAMEBUFFER, attachment, textarget, texture, level, samples);
      return;
    }
    multisampleDrawTargetVerified_ = multisampleDrawTargetVerified_ ||
                                     (verifiesTarget && error == GL_NO_ERROR);
    if (alwaysCheckError_) {
      lastError_.store(error, std::memory_order_relaxed);
      GL_ASSERT_ERROR(error == GL_NO_ERROR, __FUNCTION__, __LINE__, error);
    }
  }
}

void IContext::framebufferTextureLayer(GLenum target,
                                       GLenum attachment,
                                       GLuint texture,
                                       GLint level,
                                       GLint layer) {
#if IGL_API_LOG
  const char* unboundAttachmentType = framebufferAttachmentType(target, attachment);
  GLuint unboundAttachment = framebufferAttachment(target, attachment);
  setFramebufferAttachment(target, attachment, false, texture);
#endif
  APILOG(
      "glFramebufferTextureLayer(%s, %s, %u, %d, %d) (framebuffer: %u) unbound (%s: %u) bound "
      "(texture: %u)\n",
      GL_ENUM_TO_STRING(target),
      GL_ENUM_TO_STRING(attachment),
      texture,
      level,
      layer,
      boundFramebuffer(target),
      unboundAttachmentType,
      unboundAttachment,
      texture);
  IGLCALL(FramebufferTextureLayer)(target, attachment, texture, level, layer);
  GLCHECK_ERRORS();
}

void IContext::frontFace(GLenum mode) {
  APILOG("glFrontFace(%s)\n", GL_ENUM_TO_STRING(mode));
  GLCALL(FrontFace)(mode);
  GLCHECK_ERRORS();
}

void IContext::polygonFillMode(IGL_MAYBE_UNUSED GLenum mode) {
#if IGL_OPENGL
  APILOG("glPolygonMode(%s)\n", GL_ENUM_TO_STRING(mode));
  GLCALL(PolygonMode)(GL_FRONT_AND_BACK, mode);
  GLCHECK_ERRORS();
#else
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
#endif
}

void IContext::generateMipmap(GLenum target) {
  APILOG("glGenerateMipmap(%s)\n", GL_ENUM_TO_STRING(target));
  IGLCALL(GenerateMipmap)(target);
  GLCHECK_ERRORS();
}

void IContext::genBuffers(GLsizei n, GLuint* buffers) {
  GLCALL(GenBuffers)(n, buffers);
  // NOTE: Must log after call due to return value
  APILOG("glGenBuffers(%u, %p) (buffer: %u)\n", n, buffers, buffers == nullptr ? 0 : *buffers);
  GLCHECK_ERRORS();
}

void IContext::genFramebuffers(GLsizei n, GLuint* framebuffers) {
  IGLCALL(GenFramebuffers)(n, framebuffers);
  // NOTE: Must log after call due to return value
  APILOG("glGenFramebuffers(%u, %p) (framebuffer: %u)\n",
         n,
         framebuffers,
         framebuffers == nullptr ? 0 : *framebuffers);
  GLCHECK_ERRORS();
#if IGL_API_LOG
  for (GLsizei i = 0; i < n; ++i) {
    framebufferAttachments_[framebuffers[i]] = {};
  }
#endif
}

void IContext::genRenderbuffers(GLsizei n, GLuint* renderbuffers) {
  IGLCALL(GenRenderbuffers)(n, renderbuffers);
  // NOTE: Must log after call due to return value
  APILOG("glGenRenderbuffers(%u, %p) (renderbuffer: %u)\n",
         n,
         renderbuffers,
         renderbuffers == nullptr ? 0 : *renderbuffers);
  GLCHECK_ERRORS();
}

void IContext::genTextures(GLsizei n, GLuint* textures) {
  GLCALL(GenTextures)(n, textures);
  // NOTE: Must log after call due to return value
  APILOG("glGenTextures(%u, %p) (texture: %u)\n", n, textures, textures == nullptr ? 0 : *textures);
  GLCHECK_ERRORS();
}

void IContext::genVertexArrays(GLsizei n, GLuint* vertexArrays) {
  if (genVertexArraysProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::VertexArrayObjectExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::VertexArrayObject)) {
        genVertexArraysProc_ = iglGenVertexArraysOES;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::VertexArrayObject)) {
      genVertexArraysProc_ = iglGenVertexArrays;
    }
    IGL_DEBUG_ASSERT(genVertexArraysProc_, "No supported function for glGenVertexArrays\n");
  }
  GLCALL_PROC(genVertexArraysProc_, n, vertexArrays);
  // NOTE: Must log after call due to return value
  APILOG("glGenVertexArrays(%u, %p) (VAO: %u)\n",
         n,
         vertexArrays,
         vertexArrays == nullptr ? 0 : *vertexArrays);
  GLCHECK_ERRORS();
}

void IContext::getActiveAttrib(GLuint program,
                               GLuint index,
                               GLsizei bufsize,
                               GLsizei* length,
                               GLint* size,
                               GLenum* type,
                               GLchar* name) const {
  GLCALL(GetActiveAttrib)(program, index, bufsize, length, size, type, name);
  // NOTE: Must log after call due to return value
  APILOG("glGetActiveAttrib(%u, %u, %u, %p, %p, %p, %p) = (%d, %s, %.*s) (program: %u)\n",
         program,
         index,
         bufsize,
         length,
         size,
         type,
         name,
         size == nullptr ? 0 : *size,
         type == nullptr ? GL_ENUM_TO_STRING(0) : GL_ENUM_TO_STRING(*type),
         static_cast<int>(length == nullptr
                              ? (name == nullptr ? 0 : static_cast<int>(std::strlen(name)))
                              : *length),
         name == nullptr ? "" : name,
         program);
  GLCHECK_ERRORS();
}

void IContext::getActiveUniform(GLuint program,
                                GLuint index,
                                GLsizei bufsize,
                                GLsizei* length,
                                GLint* size,
                                GLenum* type,
                                GLchar* name) const {
  GLCALL(GetActiveUniform)(program, index, bufsize, length, size, type, name);
  // NOTE: Must log after call due to return value
  APILOG("glGetActiveUniform(%u, %u, %u, %p, %p, %p, %p) = (%d, %s, %.*s) (program: %u)\n",
         program,
         index,
         bufsize,
         length,
         size,
         type,
         name,
         size == nullptr ? 0 : *size,
         type == nullptr ? GL_ENUM_TO_STRING(0) : GL_ENUM_TO_STRING(*type),
         static_cast<int>(length == nullptr
                              ? (name == nullptr ? 0 : static_cast<int>(std::strlen(name)))
                              : *length),
         name == nullptr ? "" : name,
         program);
  GLCHECK_ERRORS();
#if IGL_API_LOG
  addProgramUniform(program, length, type, name);
#endif
}

void IContext::getActiveUniformsiv(GLuint program,
                                   GLsizei uniformCount,
                                   const GLuint* uniformIndices,
                                   GLenum pname,
                                   GLint* params) const {
  IGLCALL(GetActiveUniformsiv)(program, uniformCount, uniformIndices, pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetActiveUniformsiv(%u, %u, %p, %s, %p) (program: %u)\n",
         program,
         uniformCount,
         uniformIndices,
         GL_ENUM_TO_STRING(pname),
         params,
         program);
  GLCHECK_ERRORS();
}

void IContext::getActiveUniformBlockiv(GLuint program,
                                       GLuint index,
                                       GLenum pname,
                                       GLint* params) const {
  IGLCALL(GetActiveUniformBlockiv)(program, index, pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetActiveUniformBlockiv(%u, %u, %s, %p) = %d (program: %u)\n",
         program,
         index,
         GL_ENUM_TO_STRING(pname),
         params,
         params == nullptr ? 0 : *params,
         program);
  GLCHECK_ERRORS();
#if IGL_API_LOG
  if (pname == GL_UNIFORM_BLOCK_BINDING) {
    programUbosByLocation_[program][params == nullptr ? 0 : *params] =
        programUbosByIndex_[program][index];
  }
#endif
}

void IContext::getActiveUniformBlockName(GLuint program,
                                         GLuint index,
                                         GLsizei bufSize,
                                         GLsizei* length,
                                         GLchar* uniformBlockName) const {
  IGLCALL(GetActiveUniformBlockName)(program, index, bufSize, length, uniformBlockName);
  // NOTE: Must log after call due to return value
  APILOG(
      "glGetActiveUniformBlockName(%u, %u, %u, %p, %p) = %.*s (program: %u)\n",
      program,
      index,
      bufSize,
      length,
      uniformBlockName,
      static_cast<int>(
          length == nullptr
              ? (uniformBlockName == nullptr ? 0 : static_cast<int>(std::strlen(uniformBlockName)))
              : *length),
      uniformBlockName == nullptr ? "" : uniformBlockName,
      program);
  GLCHECK_ERRORS();
#if IGL_API_LOG
  programUbosByIndex_[program][index] = (uniformBlockName == nullptr) ? "" : uniformBlockName;
#endif
}

GLint IContext::getAttribLocation(GLuint program, const GLchar* name) const {
  GLint ret = 0;

  GLCALL_WITH_RETURN(ret, GetAttribLocation)(program, name);
  // NOTE: Must log after call due to return value
  APILOG("glGetAttribLocation(%u, %s) = %d (program: %u)\n", program, name, ret, program);
  GLCHECK_ERRORS();

  return ret;
}

void IContext::getBooleanv(GLenum pname, GLboolean* params) const {
  GLCALL(GetBooleanv)(pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetBooleanv(%s, %p) = %s\n",
         GL_ENUM_TO_STRING(pname),
         params,
         params == nullptr ? "" : GL_BOOL_TO_STRING(*params));
  GLCHECK_ERRORS();
}

void IContext::getBufferParameteriv(GLenum target, GLenum pname, GLint* params) const {
  GLCALL(GetBufferParameteriv)(target, pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetBufferParameteriv(%s, %s, %p) = %d (buffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         GL_ENUM_TO_STRING(pname),
         params,
         params == nullptr ? 0 : *params,
         boundBuffer(target));
  GLCHECK_ERRORS();
}

GLenum IContext::getError() const {
  // Using direct GL call here instead of wrapped one,
  // since we will add error call counting at some point.
  return glGetError();
}

// NOLINTNEXTLINE(misc-no-recursion)
GLuint IContext::getDebugMessageLog(GLuint count,
                                    GLsizei bufSize,
                                    GLenum* sources,
                                    GLenum* types,
                                    GLuint* ids,
                                    GLenum* severities,
                                    GLsizei* lengths,
                                    GLchar* messageLog) const {
  if (getDebugMessageLogProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::DebugMessageCallbackExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::Debug)) {
        getDebugMessageLogProc_ = iglGetDebugMessageLogKHR;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessageCallback)) {
      getDebugMessageLogProc_ = iglGetDebugMessageLog;
    }
    IGL_DEBUG_ASSERT(getDebugMessageLogProc_, "No supported function for glGetDebugMessageLog\n");
  }
  GLuint ret = 0;
  GLCALL_PROC_WITH_RETURN(ret,
                          getDebugMessageLogProc_,
                          0,
                          count,
                          bufSize,
                          sources,
                          types,
                          ids,
                          severities,
                          lengths,
                          messageLog);
  // NOTE: Must log after call due to return value
  APILOG("glGetDebugMessageLog(%u, %d, %p, %p, %p, %p, %p, %p)\n",
         count,
         bufSize,
         sources,
         types,
         ids,
         severities,
         lengths,
         messageLog);
  GLCHECK_ERRORS();
  return ret;
}

void IContext::getFramebufferAttachmentParameteriv(GLenum target,
                                                   GLenum attachment,
                                                   GLenum pname,
                                                   GLint* params) const {
  IGLCALL(GetFramebufferAttachmentParameteriv)(target, attachment, pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetFramebufferAttachmentParameteriv(%s, %s, %s, %p) = %d (framebuffer: %u) (%s: %u)\n",
         GL_ENUM_TO_STRING(target),
         GL_ENUM_TO_STRING(attachment),
         GL_ENUM_TO_STRING(pname),
         params,
         params == nullptr ? 0 : *params,
         boundFramebuffer(target),
         framebufferAttachmentType(target, attachment),
         framebufferAttachment(target, attachment));
  GLCHECK_ERRORS();
}

// NOLINTNEXTLINE(misc-no-recursion)
void IContext::getIntegerv(GLenum pname, GLint* params) const {
  GLCALL(GetIntegerv)(pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetIntegerv(%s, %p) = %d\n",
         GL_ENUM_TO_STRING(pname),
         params,
         params == nullptr ? 0 : *params);
  GLCHECK_ERRORS();
}

void IContext::getInternalformativ(GLenum target,
                                   GLenum internalformat,
                                   GLenum pname,
                                   GLsizei bufSize,
                                   GLint* params) const {
  IGLCALL(GetInternalformativ)(target, internalformat, pname, bufSize, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetInternalformativ(%s, %s, %s, %d, %p) = %d\n",
         GL_ENUM_TO_STRING(target),
         GL_ENUM_TO_STRING(internalformat),
         GL_ENUM_TO_STRING(pname),
         bufSize,
         params,
         params == nullptr ? 0 : *params);
  GLCHECK_ERRORS();
}

void IContext::getProgramiv(GLuint program, GLenum pname, GLint* params) const {
  GLCALL(GetProgramiv)(program, pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetProgramiv(%u, %s, %p) = %d (program: %u)\n",
         program,
         GL_ENUM_TO_STRING(pname),
         params,
         params == nullptr ? 0 : *params,
         program);
  GLCHECK_ERRORS();
}

void IContext::getProgramInterfaceiv(GLuint program,
                                     GLenum programInterface,
                                     GLenum pname,
                                     GLint* params) const {
  IGLCALL(GetProgramInterfaceiv)(program, programInterface, pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetProgramInterfaceiv(%u, %s, %s, %p) = %d (program: %u)\n",
         program,
         GL_ENUM_TO_STRING(programInterface),
         GL_ENUM_TO_STRING(pname),
         params,
         params == nullptr ? 0 : *params,
         program);
  GLCHECK_ERRORS();
}

void IContext::getProgramInfoLog(GLuint program,
                                 GLsizei bufsize,
                                 GLsizei* length,
                                 GLchar* infolog) const {
  GLCALL(GetProgramInfoLog)(program, bufsize, length, infolog);
  // NOTE: Must log after call due to return value
  APILOG("glGetProgramInfoLog(%u, %u, %p, %p) = %.*s (progam: %u)\n",
         program,
         bufsize,
         length,
         infolog,
         static_cast<int>(length == nullptr
                              ? (infolog == nullptr ? 0 : static_cast<int>(std::strlen(infolog)))
                              : *length),
         infolog == nullptr ? "" : infolog,
         program);
  GLCHECK_ERRORS();
}

GLuint IContext::getProgramResourceIndex(GLuint program,
                                         GLenum programInterface,
                                         const GLchar* name) const {
  GLuint ret = 0;

  IGLCALL_WITH_RETURN(ret, GetProgramResourceIndex)(program, programInterface, name);
  // NOTE: Must log after call due to return value
  APILOG("glGetProgramResourceIndex(%u, %s, %s) = %u (program: %u)\n",
         program,
         GL_ENUM_TO_STRING(programInterface),
         name,
         ret,
         program);
  GLCHECK_ERRORS();

#if IGL_API_LOG
  setProgramResourceIndex(program, programInterface, name, ret);
#endif

  return ret;
}

void IContext::getProgramResourceiv(GLuint program,
                                    GLenum programInterface,
                                    GLuint index,
                                    GLsizei propCount,
                                    const GLenum* IGL_NULLABLE props,
                                    GLsizei bufSize,
                                    GLsizei* IGL_NULLABLE length,
                                    GLint* IGL_NULLABLE params) {
  IGLCALL(GetProgramResourceiv)
  (program, programInterface, index, propCount, props, bufSize, length, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetProgramResourceiv(%u, %s, %u, %u, %p [%s], %u, %p, %p) = (%u, %d) (program: %u)\n",
         program,
         GL_ENUM_TO_STRING(programInterface),
         index,
         propCount,
         props,
         GL_ENUM_TO_STRING(props == nullptr ? 0 : *props),
         bufSize,
         length,
         params,
         length == nullptr ? 0 : *length,
         params == nullptr ? 0 : *params,
         program);
  GLCHECK_ERRORS();

#if IGL_API_LOG
  if (programInterface == GL_SHADER_STORAGE_BLOCK && propCount > 0 && props != nullptr &&
      *props == GL_BUFFER_BINDING) {
    programSsbosByLocation_[program][params == nullptr ? 0 : *params] =
        programSsbosByIndex_[program][index];
  }
#endif
}

void IContext::getProgramResourceName(GLuint program,
                                      GLenum programInterface,
                                      GLuint index,
                                      GLsizei bufSize,
                                      GLsizei* length,
                                      char* name) const {
  IGLCALL(GetProgramResourceName)(program, programInterface, index, bufSize, length, name);
  // NOTE: Must log after call due to return value
  APILOG("glGetProgramResourceName(%u, %s, %u, %u, %p, %p) = %.*s (program: %u)\n",
         program,
         GL_ENUM_TO_STRING(programInterface),
         index,
         bufSize,
         length,
         name,
         static_cast<int>(length == nullptr
                              ? (name == nullptr ? 0 : static_cast<int>(std::strlen(name)))
                              : *length),
         name == nullptr ? "" : name,
         program);
  GLCHECK_ERRORS();
}

void IContext::getShaderiv(GLuint shader, GLenum pname, GLint* params) const {
  GLCALL(GetShaderiv)(shader, pname, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetShaderiv(%u, %s, %p) = %d\n",
         shader,
         GL_ENUM_TO_STRING(pname),
         params,
         params == nullptr ? 0 : *params);
  GLCHECK_ERRORS();
}

void IContext::getShaderInfoLog(GLuint shader,
                                GLsizei maxLength,
                                GLsizei* length,
                                GLchar* infoLog) const {
  GLCALL(GetShaderInfoLog)(shader, maxLength, length, infoLog);
  // NOTE: Must log after call due to return value
  APILOG("glGetShaderInfoLog(%u, %u, %p, %p) = %.*s\n",
         shader,
         maxLength,
         length,
         infoLog,
         static_cast<int>(length == nullptr
                              ? (infoLog == nullptr ? 0 : static_cast<int>(std::strlen(infoLog)))
                              : *length),
         infoLog == nullptr ? "" : infoLog);
  GLCHECK_ERRORS();
}

const GLubyte* IContext::getString(GLenum name) const {
  const GLubyte* ret = nullptr;

  GLCALL_WITH_RETURN(ret, GetString)(name);
  // NOTE: Must log after call due to return value
  APILOG("glGetString(%s) = %s\n", GL_ENUM_TO_STRING(name), reinterpret_cast<const char*>(ret));
  GLCHECK_ERRORS();

  return ret;
}

const GLubyte* IContext::getStringi(GLenum name, GLuint index) const {
  const GLubyte* ret = nullptr;

  IGLCALL_WITH_RETURN(ret, GetStringi)(name, index);
  // NOTE: Must log after call due to return value
  APILOG("glGetStringi(%s, %d) = %s\n",
         GL_ENUM_TO_STRING(name),
         index,
         reinterpret_cast<const char*>(ret));
  GLCHECK_ERRORS();

  return ret;
}

void IContext::getSynciv(GLsync sync,
                         GLenum pname,
                         GLsizei bufSize,
                         GLsizei* length,
                         GLint* values) const {
  if (getSyncivProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::SyncExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::Sync)) {
        getSyncivProc_ = iglGetSyncivAPPLE;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::Sync)) {
      getSyncivProc_ = iglGetSynciv;
    }
    IGL_DEBUG_ASSERT(getSyncivProc_, "No supported function for glGetSynciv\n");
  }
  GLCALL_PROC(getSyncivProc_, sync, pname, bufSize, length, values);
  // NOTE: Must log after call due to return value
  APILOG("glGetSynciv(%p, %s, %u, %p, %p) = %d (sync: %p)\n",
         sync,
         GL_ENUM_TO_STRING(pname),
         bufSize,
         length,
         values,
         values == nullptr ? 0 : *values,
         sync);
  GLCHECK_ERRORS();
}

void IContext::getUniformiv(GLuint program, GLint location, GLint* params) const {
  GLCALL(GetUniformiv)(program, location, params);
  // NOTE: Must log after call due to return value
  APILOG("glGetUniformiv(%u, %d, %p) = %d (program: %u)\n",
         program,
         location,
         params,
         params == nullptr ? 0 : *params,
         program);
  GLCHECK_ERRORS();
}

GLint IContext::getUniformLocation(GLuint program, const GLchar* name) const {
  GLint ret = 0;

  GLCALL_WITH_RETURN(ret, GetUniformLocation)(program, name);
  // NOTE: Must log after call due to return value
  APILOG("glGetUniformLocation(%u, %s) = %d (program: %u)\n", program, name, ret, program);
  GLCHECK_ERRORS();

#if IGL_API_LOG
  setProgramUniformLocation(program, name, ret);
#endif

  return ret;
}

void IContext::importMemoryFd(GLuint memory, GLuint64 size, GLenum handleType, GLint fd) {
  APILOG(
      "glImportMemoryFdEXT(%u, %llu, %s, %d)\n", memory, size, GL_ENUM_TO_STRING(handleType), fd);
  IGLCALL(ImportMemoryFdEXT)(memory, size, handleType, fd);
  GLCHECK_ERRORS();
}

void IContext::invalidateFramebuffer(IGL_MAYBE_UNUSED GLenum target,
                                     IGL_MAYBE_UNUSED GLsizei numAttachments,
                                     IGL_MAYBE_UNUSED const GLenum* attachments) {
  if (invalidateFramebufferProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(
            InternalRequirement::InvalidateFramebufferExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::DiscardFramebuffer)) {
        invalidateFramebufferProc_ = iglDiscardFramebufferEXT;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::InvalidateFramebuffer)) {
      invalidateFramebufferProc_ = iglInvalidateFramebuffer;
    }
    IGL_DEBUG_ASSERT(invalidateFramebufferProc_,
                     "No supported function for glInvalidateFramebuffer\n");
  }
  APILOG("glInvalidateFramebuffer(%s, %u, %p) (framebuffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         numAttachments,
         attachments,
         boundFramebuffer(target));
  GLCALL_PROC(invalidateFramebufferProc_, target, numAttachments, attachments);
  GLCHECK_ERRORS();
}

GLboolean IContext::isEnabled(GLenum cap) {
  GLboolean ret = 0;

  GLCALL_WITH_RETURN(ret, IsEnabled)(cap);
  // NOTE: Must log after call due to return value
  APILOG("glIsEnabled(%s) = %s\n", GL_ENUM_TO_STRING(cap), GL_BOOL_TO_STRING(ret));
  GLCHECK_ERRORS();

  return ret;
}

GLboolean IContext::isTexture(GLuint texture) {
  GLboolean ret = 0;

  GLCALL_WITH_RETURN(ret, IsTexture)(texture);
  // NOTE: Must log after call due to return value
  APILOG("glIsTexture(%u) = %s\n", texture, GL_BOOL_TO_STRING(ret));
  GLCHECK_ERRORS();

  return ret;
}

void IContext::linkProgram(GLuint program) {
  APILOG("glLinkProgram(%u) (program: %u)\n", program, program);
  GLCALL(LinkProgram)(program);

  // NOTE: Explicitly *not* checking for errors
  // If there is an error, we want the client code to get error message and report
  // to the user/logs, rather than assert here.
  // @fb-only
  // @fb-only
  // @fb-only
}

void* IContext::mapBuffer(GLenum target, GLenum access) {
  if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::MapBufferExtReq)) {
    if (deviceFeatureSet_.hasExtension(Extensions::MapBuffer)) {
      mapBufferProc_ = iglMapBufferOES;
    }
  } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::MapBuffer)) {
    mapBufferProc_ = iglMapBuffer;
  }
  void* ret = nullptr;
  GLCALL_PROC_WITH_RETURN(ret, mapBufferProc_, nullptr, target, access);
  // NOTE: Must log after call due to return value
  APILOG("glMapBuffer(%s, %zu, %zu, %s) = %p (buffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         GL_ENUM_TO_STRING(access),
         ret,
         boundBuffer(target));
  GLCHECK_ERRORS();
  return ret;
}

void* IContext::mapBufferRange(GLenum target,
                               GLintptr offset,
                               GLsizeiptr length,
                               GLbitfield access) {
  if (mapBufferRangeProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::MapBufferRangeExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::MapBufferRange)) {
        mapBufferRangeProc_ = iglMapBufferRangeEXT;
      }
    } else if (deviceFeatureSet_.hasFeature(DeviceFeatures::MapBufferRange)) {
      mapBufferRangeProc_ = iglMapBufferRange;
    }
    IGL_DEBUG_ASSERT(mapBufferRangeProc_, "No supported function for glMapBufferRange\n");
  }
  void* ret = nullptr;
  GLCALL_PROC_WITH_RETURN(ret, mapBufferRangeProc_, nullptr, target, offset, length, access);
  // NOTE: Must log after call due to return value
  APILOG("glMapBufferRange(%s, %zu, %zu, 0x%x [%s]) = %p (buffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         offset,
         length,
         access,
         GL_MAP_BUFFER_RANGE_BITS_TO_STRING(access),
         ret,
         boundBuffer(target));
  GLCHECK_ERRORS();
  return ret;
}

void IContext::objectLabel(GLenum identifier, GLuint name, GLsizei length, const char* label) {
  if (objectLabelProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::DebugLabelExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::Debug)) {
        objectLabelProc_ = iglObjectLabelKHR;
      } else if (deviceFeatureSet_.hasExtension(Extensions::DebugLabel)) {
        objectLabelProc_ = iglLabelObjectEXT;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugLabel)) {
      objectLabelProc_ = iglObjectLabel;
    }
    IGL_DEBUG_ASSERT(objectLabelProc_, "No supported function for glObjectLabel\n");
  }
  APILOG("glObjectLabel(%s, %u, %u, %s) %s\n",
         GL_ENUM_TO_STRING(identifier),
         name,
         length,
         label,
         identifierLabel(identifier, name).c_str());
  GLCALL_PROC(objectLabelProc_, identifier, name, length, label);
  GLCHECK_ERRORS();
}

void IContext::pixelStorei(GLenum pname, GLint param) {
  APILOG("glPixelStorei(%s, %d)\n", GL_ENUM_TO_STRING(pname), param);
  GLCALL(PixelStorei)(pname, param);
  GLCHECK_ERRORS();
}

void IContext::polygonOffsetClamp(GLfloat factor, GLfloat units, float clamp) {
  const bool hasPolygonOffsetClamp = deviceFeatureSet_.hasExtension(Extensions::PolygonOffsetClamp);

  if (!hasPolygonOffsetClamp) {
    IGL_DEBUG_ASSERT(clamp == 0, "PolygonOffsetClamp is not supported\n");
  }

  if (clamp == 0 || !hasPolygonOffsetClamp) {
    APILOG("glPolygonOffset(%f, %f)\n", factor, units);
    GLCALL(PolygonOffset)(factor, units);
  } else {
    APILOG("glPolygonOffsetClamp(%f, %f, %f)\n", factor, units, clamp);
    IGLCALL(PolygonOffsetClamp)(factor, units, clamp);
  }
  GLCHECK_ERRORS();
}

void IContext::pushDebugGroup(GLenum source, GLuint id, GLsizei length, const GLchar* message) {
  if (pushDebugGroupProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessage)) {
      if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::DebugMessageExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::Debug)) {
          pushDebugGroupProc_ = iglPushDebugGroupKHR;
          getIntegerv(GL_MAX_DEBUG_GROUP_STACK_DEPTH, &maxDebugStackSize_);
        } else if (deviceFeatureSet_.hasExtension(Extensions::DebugMarker)) {
          pushDebugGroupProc_ = iglPushGroupMarkerEXT;
          // GL_EXT_debug_marker does not provide a way to query the max
          maxDebugStackSize_ = std::numeric_limits<int>::max();
        }
      } else {
        pushDebugGroupProc_ = iglPushDebugGroup;
        getIntegerv(GL_MAX_DEBUG_GROUP_STACK_DEPTH, &maxDebugStackSize_);
      }
    }
    IGL_DEBUG_ASSERT(pushDebugGroupProc_, "No supported function for glPushDebugGroup\n");
  }

  if (++debugStackSize_ < maxDebugStackSize_) {
    APILOG("glPushDebugGroup(%s, %u, %u, %s)\n", GL_ENUM_TO_STRING(source), id, length, message);
    GLCALL_PROC(pushDebugGroupProc_, source, id, length, message);
    GLCHECK_ERRORS();
  } else {
    IGL_LOG_ERROR_ONCE("Exceeded max debug group stack size of %d, ignoring push\n",
                       maxDebugStackSize_);
  }
}

void IContext::popDebugGroup() {
  if (popDebugGroupProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessage)) {
      if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::DebugMessageExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::Debug)) {
          popDebugGroupProc_ = iglPopDebugGroupKHR;
        } else if (deviceFeatureSet_.hasExtension(Extensions::DebugMarker)) {
          popDebugGroupProc_ = iglPopGroupMarkerEXT;
        }
      } else {
        popDebugGroupProc_ = iglPopDebugGroup;
      }
    }
    IGL_DEBUG_ASSERT(popDebugGroupProc_, "No supported function for glPopDebugGroup\n");
  }

  if (debugStackSize_ < maxDebugStackSize_ && maxDebugStackSize_ > 0) {
    --debugStackSize_;
    APILOG("glPopDebugGroup()\n");
    GLCALL_PROC(popDebugGroupProc_);
    GLCHECK_ERRORS();
  } else if (debugStackSize_ >= maxDebugStackSize_) {
    --debugStackSize_;
    IGL_LOG_ERROR_ONCE("Debug group stack size was exceeded, ignoring pop\n");
  } else if (maxDebugStackSize_ == 0) {
    IGL_LOG_ERROR_ONCE("Debug group stack is empty, can't pop\n");
  }
}

void IContext::readBuffer(GLenum src) {
  if (readBufferProc_ == nullptr) {
    if (deviceFeatureSet_.hasFeature(DeviceFeatures::ReadWriteFramebuffer)) {
      readBufferProc_ = iglReadBuffer;
    }
    IGL_DEBUG_ASSERT(readBufferProc_, "No supported function for glReadBuffer\n");
  }
  IGL_PROFILER_ZONE_GPU_COLOR_OGL("readBuffer()", IGL_PROFILER_COLOR_DRAW);

  APILOG("glReadBuffer(%s) read (framebuffer: %u)\n",
         GL_ENUM_TO_STRING(src),
         boundFramebuffer(GL_READ_FRAMEBUFFER));
  GLCALL_PROC(readBufferProc_, src);
  GLCHECK_ERRORS();
}

void IContext::readPixels(GLint x,
                          GLint y,
                          GLsizei width,
                          GLsizei height,
                          GLenum format,
                          GLenum type,
                          GLvoid* pixels) {
  APILOG("glReadPixels(%d, %u, %u, %d, %s, %s, %p) (buffer: %u) (%s: %u)\n",
         x,
         y,
         width,
         height,
         GL_ENUM_TO_STRING(format),
         GL_ENUM_TO_STRING(type),
         pixels,
         boundBuffer(GL_PIXEL_PACK_BUFFER),
         framebufferReadBufferType(GL_READ_FRAMEBUFFER),
         framebufferReadBuffer(GL_READ_FRAMEBUFFER));
  GLCALL(ReadPixels)(x, y, width, height, format, type, pixels);
  GLCHECK_ERRORS();
}

void IContext::renderbufferStorage(GLenum target,
                                   GLenum internalformat,
                                   GLsizei width,
                                   GLsizei height) {
  APILOG("glRenderbufferStorage(%s, %s, %u, %u) (renderbuffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         boundRenderbuffer_);
  IGLCALL(RenderbufferStorage)(target, internalformat, width, height);
  GLCHECK_ERRORS();
}

void IContext::renderbufferStorageMultisample(GLenum target,
                                              GLsizei samples,
                                              GLenum internalformat,
                                              GLsizei width,
                                              GLsizei height) {
  if (renderbufferStorageMultisampleProc_ == nullptr) {
    // Use runtime checks to determine which of several potential methods are supported by the
    // context.
    if (deviceFeatureSet_.hasFeature(DeviceFeatures::MultiSample)) {
      if (!deviceFeatureSet_.hasInternalRequirement(InternalRequirement::MultiSampleExtReq)) {
        renderbufferStorageMultisampleProc_ = iglRenderbufferStorageMultisample;
      } else if (deviceFeatureSet_.hasExtension(Extensions::MultiSampleExt)) {
        renderbufferStorageMultisampleProc_ = iglRenderbufferStorageMultisampleEXT;
      } else if (deviceFeatureSet_.hasExtension(Extensions::MultiSampleImg)) {
        renderbufferStorageMultisampleProc_ = iglRenderbufferStorageMultisampleIMG;
      } else if (deviceFeatureSet_.hasExtension(Extensions::MultiSampleApple)) {
        renderbufferStorageMultisampleProc_ = iglRenderbufferStorageMultisampleAPPLE;
      }
    }
    IGL_DEBUG_ASSERT(renderbufferStorageMultisampleProc_,
                     "No supported function for glRenderbufferStorageMultisampleProc\n");
  }

  APILOG("glRenderbufferStorageMultisampleProc(%s, %u, %s, %u, %u) (renderbuffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         samples,
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         boundRenderbuffer_);
  GLCALL_PROC(renderbufferStorageMultisampleProc_, target, samples, internalformat, width, height);
  GLCHECK_ERRORS();
}

void IContext::renderbufferStorageMultisampleImplicit(GLenum target,
                                                      GLsizei samples,
                                                      GLenum internalformat,
                                                      GLsizei width,
                                                      GLsizei height) {
  if (renderbufferStorageMultisampleImplicitProc_ == nullptr) {
    if (deviceFeatureSet_.hasExtension(Extensions::MultiSampleExt)) {
      renderbufferStorageMultisampleImplicitProc_ = iglRenderbufferStorageMultisampleEXT;
    } else if (deviceFeatureSet_.hasExtension(Extensions::MultiSampleImg)) {
      renderbufferStorageMultisampleImplicitProc_ = iglRenderbufferStorageMultisampleIMG;
    }
    IGL_DEBUG_ASSERT(renderbufferStorageMultisampleImplicitProc_,
                     "No supported function for glRenderbufferStorageMultisampleEXT\n");
  }
  if (renderbufferStorageMultisampleImplicitProc_ != nullptr) {
    samples = std::min(samples, getMaxMultisampledRenderToTextureSamples());
  }

  APILOG("glRenderbufferStorageMultisampleEXT(%s, %u, %s, %u, %u) (renderbuffer: %u)\n",
         GL_ENUM_TO_STRING(target),
         samples,
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         boundRenderbuffer_);
  GLCALL_PROC(
      renderbufferStorageMultisampleImplicitProc_, target, samples, internalformat, width, height);
  GLCHECK_ERRORS();
}

bool IContext::usesImgMultisampledRenderToTexture() const {
  return !deviceFeatureSet_.hasExtension(Extensions::MultiSampleExt) &&
         deviceFeatureSet_.hasExtension(Extensions::MultiSampleImg);
}

GLint IContext::getMaxMultisampledRenderToTextureSamples() {
  if (maxSamples_ == -1) {
    if (usesImgMultisampledRenderToTexture()) {
      getIntegerv(GL_MAX_SAMPLES_IMG, &maxSamples_);
    } else {
      // GL_MAX_SAMPLES_EXT of the EXT version has the value of GL_MAX_SAMPLES
      getIntegerv(GL_MAX_SAMPLES, &maxSamples_);
    }
  }
  return maxSamples_;
}

GLint IContext::getMaxMultisampledRenderToTextureSamples(GLenum internalFormat) {
  const GLint maxSamples = getMaxMultisampledRenderToTextureSamples();
  if (maxSamples <= 0 ||
      !deviceFeatureSet_.hasInternalFeature(InternalFeatures::InternalFormatQuery)) {
    return maxSamples;
  }
  const auto cached = maxSamplesByFormat_.find(internalFormat);
  if (cached != maxSamplesByFormat_.end()) {
    return cached->second;
  }
  // The sample counts come in descending order, so the first one is the largest. A format that
  // cannot be multisampled has none, and the query leaves the 0.
  GLint maxFormatSamples = 0;
  getInternalformativ(GL_RENDERBUFFER, internalFormat, GL_SAMPLES, 1, &maxFormatSamples);
  const GLint result = std::min(maxSamples, maxFormatSamples);
  maxSamplesByFormat_.emplace(internalFormat, result);
  return result;
}

void IContext::framebufferTextureMultiview(GLenum target,
                                           GLenum attachment,
                                           GLuint texture,
                                           GLint level,
                                           GLint baseViewIndex,
                                           GLsizei numViews) {
#if IGL_API_LOG
  const char* unboundAttachmentType = framebufferAttachmentType(target, attachment);
  GLuint unboundAttachment = framebufferAttachment(target, attachment);
  setFramebufferAttachment(target, attachment, false, texture);
#endif
  APILOG(
      "glFramebufferTextureMultiviewOVR(%s, %s, %u, %d, %d, %u) (framebuffer: %u) unbound (%s: %u) "
      "bound (texture: %u)\n",
      GL_ENUM_TO_STRING(target),
      GL_ENUM_TO_STRING(attachment),
      texture,
      level,
      baseViewIndex,
      numViews,
      boundFramebuffer(target),
      unboundAttachmentType,
      unboundAttachment,
      texture);
  IGLCALL(FramebufferTextureMultiviewOVR)
  (target, attachment, texture, level, baseViewIndex, numViews);
  GLCHECK_ERRORS();
}

void IContext::framebufferTextureMultisampleMultiview(GLenum target,
                                                      GLenum attachment,
                                                      GLuint texture,
                                                      GLint level,
                                                      GLsizei samples,
                                                      GLint baseViewIndex,
                                                      GLsizei numViews) {
#if IGL_API_LOG
  const char* unboundAttachmentType = framebufferAttachmentType(target, attachment);
  GLuint unboundAttachment = framebufferAttachment(target, attachment);
  setFramebufferAttachment(target, attachment, false, texture);
#endif
  APILOG(
      "glFramebufferTextureMultisampleMultiview(%s, %s, %u, %d, %u, %d, %u) (framebuffer: %u) "
      "unbound (%s: %u) bound (texture: %u)\n",
      GL_ENUM_TO_STRING(target),
      GL_ENUM_TO_STRING(attachment),
      texture,
      level,
      samples,
      baseViewIndex,
      numViews,
      boundFramebuffer(target),
      unboundAttachmentType,
      unboundAttachment,
      texture);
  IGLCALL(FramebufferTextureMultisampleMultiviewOVR)
  (target, attachment, texture, level, samples, baseViewIndex, numViews);
  GLCHECK_ERRORS();
}

void IContext::scissor(GLint x, GLint y, GLsizei width, GLsizei height) {
  APILOG("glScissor(%d, %d, %u, %u)\n", x, y, width, height);
  GLCALL(Scissor)(x, y, width, height);
  GLCHECK_ERRORS();
}

void IContext::setEnabled(bool shouldEnable, GLenum cap) {
  if (shouldEnable) {
    APILOG("glEnable(%s)\n", GL_ENUM_TO_STRING(cap));
    GLCALL(Enable)(cap);
  } else {
    APILOG("glDisable(%s)\n", GL_ENUM_TO_STRING(cap));
    GLCALL(Disable)(cap);
  }
  GLCHECK_ERRORS();
}

void IContext::shaderSource(GLuint shader,
                            GLsizei count,
                            const GLchar** string,
                            const GLint* length) {
  APILOG("glShaderSource(%d, %u, %p, %p)\n", shader, count, string, length);
  APILOG_SOURCE(count, string, length);
  GLCALL(ShaderSource)(shader, count, string, length);
  GLCHECK_ERRORS();
}

void IContext::stencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask) {
  APILOG("glStencilFuncSeparate(%s, %s, %d, 0x%x)\n",
         GL_ENUM_TO_STRING(face),
         GL_ENUM_TO_STRING(func),
         ref,
         mask);
  GLCALL(StencilFuncSeparate)(face, func, ref, mask);
  GLCHECK_ERRORS();
}

void IContext::stencilMask(GLuint mask) {
  APILOG("glStencilMask(0x%x)\n", mask);
  GLCALL(StencilMask)(mask);
  GLCHECK_ERRORS();
}

void IContext::stencilMaskSeparate(GLenum face, GLuint mask) {
  APILOG("glStencilMaskSeparate(%s, 0x%x)\n", GL_ENUM_TO_STRING(face), mask);
  GLCALL(StencilMaskSeparate)(face, mask);
  GLCHECK_ERRORS();
}

void IContext::stencilOpSeparate(GLenum face, GLenum fail, GLenum zfail, GLenum zpass) {
  APILOG("glStencilOpSeparate(%s, %s, %s, %s)\n",
         GL_ENUM_TO_STRING(face),
         GL_ENUM_TO_STRING(fail),
         GL_ENUM_TO_STRING(zfail),
         GL_ENUM_TO_STRING(zpass));
  GLCALL(StencilOpSeparate)(face, fail, zfail, zpass);
  GLCHECK_ERRORS();
}

void IContext::texImage2D(GLenum target,
                          GLint level,
                          GLint internalformat,
                          GLsizei width,
                          GLsizei height,
                          GLint border,
                          GLenum format,
                          GLenum type,
                          const GLvoid* data) {
  APILOG("glTexImage2D(%s, %d, %s, %u, %u, %d, %s, %s, 0x%x) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         border,
         GL_ENUM_TO_STRING(format),
         GL_ENUM_TO_STRING(type),
         data,
         boundTexture(target));
  GLCALL(TexImage2D)(target, level, internalformat, width, height, border, format, type, data);
  GLCHECK_ERRORS();
}

void IContext::texStorage2D(GLenum target,
                            GLsizei levels,
                            GLenum internalformat,
                            GLsizei width,
                            GLsizei height) {
  if (texStorage2DProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::TexStorageExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::TexStorage)) {
        texStorage2DProc_ = iglTexStorage2DEXT;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::TexStorage)) {
      texStorage2DProc_ = iglTexStorage2D;
    }
    IGL_DEBUG_ASSERT(texStorage2DProc_, "No supported function for glTexStorage2D\n");
  }

  APILOG("glTexStorage2D(%s, %u, %s, %u, %u) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         levels,
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         boundTexture(target));
  GLCALL_PROC(texStorage2DProc_, target, levels, internalformat, width, height);
  GLCHECK_ERRORS();
}

void IContext::texStorage3D(GLenum target,
                            GLsizei levels,
                            GLenum internalformat,
                            GLsizei width,
                            GLsizei height,
                            GLsizei depth) {
  if (texStorage3DProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::TexStorageExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::TexStorage)) {
        texStorage3DProc_ = iglTexStorage3DEXT;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::TexStorage)) {
      texStorage3DProc_ = iglTexStorage3D;
    }
    IGL_DEBUG_ASSERT(texStorage3DProc_, "No supported function for glTexStorage3D\n");
  }
  APILOG("glTexStorage3D(%s, %u, %s, %u, %u, %u) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         levels,
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         depth,
         boundTexture(target));
  GLCALL_PROC(texStorage3DProc_, target, levels, internalformat, width, height, depth);
  GLCHECK_ERRORS();
}

void IContext::texStorageMem2D(GLenum target,
                               GLsizei levels,
                               GLenum internalFormat,
                               GLsizei width,
                               GLsizei height,
                               GLuint memory,
                               GLuint64 offset) {
  APILOG("glTexStorageMem2DEXT(%s, %u, %s, %u, %u, %u, %lu) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         levels,
         GL_ENUM_TO_STRING(internalFormat),
         width,
         height,
         memory,
         offset,
         boundTexture(target));
  IGLCALL(TexStorageMem2DEXT)(target, levels, internalFormat, width, height, memory, offset);
  GLCHECK_ERRORS();
}

void IContext::texStorageMem3D(GLenum target,
                               GLsizei levels,
                               GLenum internalFormat,
                               GLsizei width,
                               GLsizei height,
                               GLsizei depth,
                               GLuint memory,
                               GLuint64 offset) {
  APILOG("glTexStorageMem3DEXT(%s, %u, %s, %u, %u, %u, %u, %lu) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         levels,
         GL_ENUM_TO_STRING(internalFormat),
         width,
         height,
         depth,
         memory,
         offset,
         boundTexture(target));
  IGLCALL(TexStorageMem3DEXT)
  (target, levels, internalFormat, width, height, depth, memory, offset);
  GLCHECK_ERRORS();
}

void IContext::texImage3D(GLenum target,
                          GLint level,
                          GLint internalformat,
                          GLsizei width,
                          GLsizei height,
                          GLsizei depth,
                          GLint border,
                          GLenum format,
                          GLenum type,
                          const GLvoid* data) {
  if (texImage3DProc_ == nullptr) {
    if (deviceFeatureSet_.hasFeature(DeviceFeatures::Texture3D)) {
      if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::Texture3DExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::Texture3D)) {
          texImage3DProc_ = iglTexImage3DOES;
        }
      } else {
        texImage3DProc_ = iglTexImage3D;
      }
    }
    IGL_DEBUG_ASSERT(texImage3DProc_, "No supported function for glTexImage3D\n");
  }

  APILOG("glTexImage3D(%s, %d, %s, %u, %u, %u, %d, %s, %s, 0x%x) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         GL_ENUM_TO_STRING(internalformat),
         width,
         height,
         depth,
         border,
         GL_ENUM_TO_STRING(format),
         GL_ENUM_TO_STRING(type),
         data,
         boundTexture(target));
  GLCALL_PROC(texImage3DProc_,
              target,
              level,
              internalformat,
              width,
              height,
              depth,
              border,
              format,
              type,
              data);
  GLCHECK_ERRORS();
}

void IContext::texParameteri(GLenum target, GLenum pname, GLint param) {
  // GLCALL(TexParameteri, target, pname, param);
  APILOG("glTexParameteri(%s, %s, %s) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         GL_ENUM_TO_STRING(pname),
         GL_ENUM_TO_STRING(param),
         boundTexture(target));
  GLCALL(TexParameteri)(target, pname, param);
  GLCHECK_ERRORS();
}

void IContext::texSubImage2D(GLenum target,
                             GLint level,
                             GLint xoffset,
                             GLint yoffset,
                             GLsizei width,
                             GLsizei height,
                             GLenum format,
                             GLenum type,
                             const GLvoid* pixels) {
  APILOG("glTexSubImage2D(%s, %d, %d, %d, %u, %u, %s, %s, %p) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         xoffset,
         yoffset,
         width,
         height,
         GL_ENUM_TO_STRING(format),
         GL_ENUM_TO_STRING(type),
         pixels,
         boundTexture(target));
  GLCALL(TexSubImage2D)(target, level, xoffset, yoffset, width, height, format, type, pixels);
  GLCHECK_ERRORS();
}

void IContext::texSubImage3D(GLenum target,
                             GLint level,
                             GLint xoffset,
                             GLint yoffset,
                             GLint zoffset,
                             GLsizei width,
                             GLsizei height,
                             GLsizei depth,
                             GLenum format,
                             GLenum type,
                             const GLvoid* pixels) {
  if (texSubImage3DProc_ == nullptr) {
    if (deviceFeatureSet_.hasFeature(DeviceFeatures::Texture3D)) {
      if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::Texture3DExtReq)) {
        if (deviceFeatureSet_.hasExtension(Extensions::Texture3D)) {
          texSubImage3DProc_ = iglTexSubImage3DOES;
        }
      } else {
        texSubImage3DProc_ = iglTexSubImage3D;
      }
    }
    IGL_DEBUG_ASSERT(texSubImage3DProc_, "No supported function for glTexSubImage3D\n");
  }

  APILOG("glTexSubImage3D(%s, %d, %d, %d, %d, %u, %u, %u, %s, %s, %p) (texture: %u)\n",
         GL_ENUM_TO_STRING(target),
         level,
         xoffset,
         yoffset,
         zoffset,
         width,
         height,
         depth,
         GL_ENUM_TO_STRING(format),
         GL_ENUM_TO_STRING(type),
         pixels,
         boundTexture(target));
  GLCALL_PROC(texSubImage3DProc_,
              target,
              level,
              xoffset,
              yoffset,
              zoffset,
              width,
              height,
              depth,
              format,
              type,
              pixels);
  GLCHECK_ERRORS();
}

void IContext::uniform1f(GLint location, GLfloat x) {
  APILOG("glUniform1f(%d, %f) (program: %u) (uniform: %s)\n",
         location,
         x,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform1f)(location, x);
  GLCHECK_ERRORS();
}

void IContext::uniform1fv(GLint location, GLsizei count, const GLfloat* v) {
  APILOG("glUniform1fv(%d, %u, %p [%f]) (program: %u) (uniform: %s)\n",
         location,
         count,
         v,
         v == nullptr ? 0.f : *v,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform1fv)(location, count, v);
  GLCHECK_ERRORS();
}

void IContext::uniform1i(GLint location, GLint x) {
  APILOG("glUniform1i(%d, %d) (program: %u) (uniform: %s) %s\n",
         location,
         x,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str(),
         boundUniformTexture(boundProgram_, location, x).c_str());
  GLCALL(Uniform1i)(location, x);
  GLCHECK_ERRORS();
}

void IContext::uniform1iv(GLint location, GLsizei count, const GLint* v) {
  APILOG("glUniform1iv(%d, %u, %p [%d]) (program: %u) (uniform: %s)\n",
         location,
         count,
         v,
         v == nullptr ? 0 : *v,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform1iv)(location, count, v);
  GLCHECK_ERRORS();
}

void IContext::uniform2f(GLint location, GLfloat x, GLfloat y) {
  APILOG("glUniform2f(%d, %f, %f) (program: %u) (uniform: %s)\n",
         location,
         x,
         y,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform2f)(location, x, y);
  GLCHECK_ERRORS();
}

void IContext::uniform2fv(GLint location, GLsizei count, const GLfloat* v) {
  APILOG("glUniform2fv(%d, %u, %p [%f %f]) (program: %u) (uniform: %s)\n",
         location,
         count,
         v,
         v == nullptr ? 0.f : v[0],
         v == nullptr ? 0.f : v[1],
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform2fv)(location, count, v);
  GLCHECK_ERRORS();
}

void IContext::uniform2i(GLint location, GLint x, GLint y) {
  APILOG("glUniform2i(%d, %d, %d) (program: %u) (uniform: %s)\n",
         location,
         x,
         y,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform2i)(location, x, y);
  GLCHECK_ERRORS();
}

void IContext::uniform2iv(GLint location, GLsizei count, const GLint* v) {
  APILOG("glUniform2iv(%d, %u, %p [%d %d]) (program: %u) (uniform: %s)\n",
         location,
         count,
         v,
         v == nullptr ? 0 : v[0],
         v == nullptr ? 0 : v[1],
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform2iv)(location, count, v);
  GLCHECK_ERRORS();
}

void IContext::uniform3f(GLint location, GLfloat x, GLfloat y, GLfloat z) {
  APILOG("glUniform3f(%d, %f, %f, %f) (program: %u) (uniform: %s)\n",
         location,
         x,
         y,
         z,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform3f)(location, x, y, z);
  GLCHECK_ERRORS();
}

void IContext::uniform3fv(GLint location, GLsizei count, const GLfloat* v) {
  APILOG("glUniform3fv(%d, %u, %p [%f %f %f]) (program: %u) (uniform: %s)\n",
         location,
         count,
         v,
         v == nullptr ? 0.f : v[0],
         v == nullptr ? 0.f : v[1],
         v == nullptr ? 0.f : v[2],
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform3fv)(location, count, v);
  GLCHECK_ERRORS();
}

void IContext::uniform3i(GLint location, GLint x, GLint y, GLint z) {
  APILOG("glUniform3i(%d, %d, %d, %d) (program: %u) (uniform: %s)\n",
         location,
         x,
         y,
         z,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform3i)(location, x, y, z);
  GLCHECK_ERRORS();
}

void IContext::uniform3iv(GLint location, GLsizei count, const GLint* v) {
  APILOG("glUniform3fv(%d, %u, %p [%d %d %d]) (program: %u) (uniform: %s)\n",
         location,
         count,
         v,
         v == nullptr ? 0 : v[0],
         v == nullptr ? 0 : v[1],
         v == nullptr ? 0 : v[2],
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform3iv)(location, count, v);
  GLCHECK_ERRORS();
}

void IContext::uniform4f(GLint location, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
  APILOG("glUniform4f(%d, %f, %f, %f, %f) (program: %u) (uniform: %s)\n",
         location,
         x,
         y,
         z,
         w,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform4f)(location, x, y, z, w);
  GLCHECK_ERRORS();
}

void IContext::uniform4fv(GLint location, GLsizei count, const GLfloat* v) {
  APILOG("glUniform4fv(%d, %u, %p [%f %f %f %f]) (program: %u) (uniform: %s)\n",
         location,
         count,
         v,
         v == nullptr ? 0.f : v[0],
         v == nullptr ? 0.f : v[1],
         v == nullptr ? 0.f : v[2],
         v == nullptr ? 0.f : v[3],
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform4fv)(location, count, v);
  GLCHECK_ERRORS();
}

void IContext::uniform4i(GLint location, GLint x, GLint y, GLint z, GLint w) {
  APILOG("glUniform4i(%d, %d, %d, %d, %d) (program: %u) (uniform: %s)\n",
         location,
         x,
         y,
         z,
         w,
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform4i)(location, x, y, z, w);
  GLCHECK_ERRORS();
}

void IContext::uniform4iv(GLint location, GLsizei count, const GLint* v) {
  APILOG("glUniform4iv(%d, %u, %p [%d %d %d %d]) (program: %u) (uniform: %s)\n",
         location,
         count,
         v,
         v == nullptr ? 0 : v[0],
         v == nullptr ? 0 : v[1],
         v == nullptr ? 0 : v[2],
         v == nullptr ? 0 : v[3],
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(Uniform4iv)(location, count, v);
  GLCHECK_ERRORS();
}

void IContext::uniformBlockBinding(GLuint program,
                                   GLuint uniformBlockIndex,
                                   GLuint uniformBlockBinding) {
  APILOG("glUniformBlockBinding(%u, %u, %u) (program: %u)\n",
         program,
         uniformBlockIndex,
         uniformBlockBinding,
         program);
  IGLCALL(UniformBlockBinding)(program, uniformBlockIndex, uniformBlockBinding);
  GLCHECK_ERRORS();
}

void IContext::uniformMatrix2fv(GLint location,
                                GLsizei count,
                                GLboolean transpose,
                                const GLfloat* value) {
  APILOG("glUniformMatrix2fv(%d, %u, %s, %p [[%f %f] [%f %f]]) (program: %u) (uniform: %s)\n",
         location,
         count,
         GL_BOOL_TO_STRING(transpose),
         value,
         value == nullptr ? 0.f : value[0],
         value == nullptr ? 0.f : value[1],
         value == nullptr ? 0.f : value[2],
         value == nullptr ? 0.f : value[3],
         boundProgram_,
         programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(UniformMatrix2fv)(location, count, transpose, value);
  GLCHECK_ERRORS();
}

void IContext::uniformMatrix3fv(GLint location,
                                GLsizei count,
                                GLboolean transpose,
                                const GLfloat* value) {
  APILOG(
      "glUniformMatrix3fv(%d, %u, %s, %p [[%f %f %f] [%f %f %f] [%f %f %f]]) (program: %u) "
      "(uniform: %s)\n",
      location,
      count,
      GL_BOOL_TO_STRING(transpose),
      value,
      value == nullptr ? 0.f : value[0],
      value == nullptr ? 0.f : value[1],
      value == nullptr ? 0.f : value[2],
      value == nullptr ? 0.f : value[3],
      value == nullptr ? 0.f : value[4],
      value == nullptr ? 0.f : value[5],
      value == nullptr ? 0.f : value[6],
      value == nullptr ? 0.f : value[7],
      value == nullptr ? 0.f : value[8],
      boundProgram_,
      programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(UniformMatrix3fv)(location, count, transpose, value);
  GLCHECK_ERRORS();
}

void IContext::uniformMatrix4fv(GLint location,
                                GLsizei count,
                                GLboolean transpose,
                                const GLfloat* value) {
#if IGL_API_LOG
  // Use a zero-filled fallback so we can log matrix elements without 16 null-checks.
  constexpr GLfloat kZeroMatrix[16] = {};
  const GLfloat* const v = (value != nullptr) ? value : kZeroMatrix;
#endif
  APILOG(
      "glUniformMatrix4fv(%d, %u, %s, %p [[%f %f %f %f] [%f %f %f %f] [%f %f %f %f] [%f %f %f "
      "%f]]) (program: %u) (uniform: %s)\n",
      location,
      count,
      GL_BOOL_TO_STRING(transpose),
      value,
      v[0],
      v[1],
      v[2],
      v[3],
      v[4],
      v[5],
      v[6],
      v[7],
      v[8],
      v[9],
      v[10],
      v[11],
      v[12],
      v[13],
      v[14],
      v[15],
      boundProgram_,
      programUniformsByLocation_[boundProgram_][location].c_str());
  GLCALL(UniformMatrix4fv)(location, count, transpose, value);
  GLCHECK_ERRORS();
}

void IContext::unmapBuffer(GLenum target) {
  if (unmapBufferProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::UnmapBufferExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::MapBuffer) ||
          deviceFeatureSet_.hasExtension(Extensions::MapBufferRange)) {
        unmapBufferProc_ = iglUnmapBufferOES;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::UnmapBuffer)) {
      unmapBufferProc_ = iglUnmapBuffer;
    }
    IGL_DEBUG_ASSERT(unmapBufferProc_, "No supported function for glUnmapBuffer\n");
  }
  APILOG("glUnmapBuffer(%s) (buffer: %u)\n", GL_ENUM_TO_STRING(target), boundBuffer(target));
  GLCALL_PROC(unmapBufferProc_, target);
  GLCHECK_ERRORS();
}

void IContext::useProgram(GLuint program) {
#if IGL_API_LOG
  GLuint unboundProgram = boundProgram_;
  boundProgram_ = program;
#endif
  APILOG("glUseProgram(%u) unbound (program: %u) bound (program: %u)\n",
         program,
         unboundProgram,
         program);
  GLCALL(UseProgram)(program);
  GLCHECK_ERRORS();
}

void IContext::validateProgram(GLuint program) {
  APILOG("glValidateProgram(%u) (program:  %u)\n", program, program);
  GLCALL(ValidateProgram)(program);
  GLCHECK_ERRORS();
}

void IContext::viewport(GLint x, GLint y, GLsizei width, GLsizei height) {
  APILOG("glViewport(%d, %d, %u, %u)\n", x, y, width, height);
  GLCALL(Viewport)(x, y, width, height);
  GLCHECK_ERRORS();
}

void IContext::waitSync(GLsync sync, GLbitfield flags, GLuint64 timeoutNs) {
  if (waitSyncProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::Sync)) {
      waitSyncProc_ = iglWaitSync;
    }
    IGL_DEBUG_ASSERT(waitSyncProc_, "No supported function for glWaitSync\n");
  }

  APILOG("glWaitSync(%p, %u, %llu) (sync: %p)\n", sync, flags, timeoutNs, sync);
  GLCALL_PROC(waitSyncProc_, sync, flags, timeoutNs);
  GLCHECK_ERRORS();
}

GLuint64 IContext::getTextureHandle(GLuint texture) {
  if (getTextureHandleProc_ == nullptr) {
    if (deviceFeatureSet_.hasExtension(Extensions::BindlessTextureArb)) {
      getTextureHandleProc_ = iglGetTextureHandleARB;
    } else if (deviceFeatureSet_.hasExtension(Extensions::BindlessTextureNv)) {
      getTextureHandleProc_ = iglGetTextureHandleNV;
    }
    IGL_DEBUG_ASSERT(getTextureHandleProc_, "No supported function for glGetTextureHandle\n");
  }

  GLuint64 ret = 0;
  GLCALL_PROC_WITH_RETURN(ret, getTextureHandleProc_, GL_ZERO, texture);
  // NOTE: Must log after call due to return value
  APILOG("glGetTextureHandle(%u) = %llu (texture: %u)\n", texture, ret, texture);
  GLCHECK_ERRORS();
  return ret;
}

void IContext::makeTextureHandleResident(GLuint64 handle) {
  if (makeTextureHandleResidentProc_ == nullptr) {
    if (deviceFeatureSet_.hasExtension(Extensions::BindlessTextureArb)) {
      makeTextureHandleResidentProc_ = iglMakeTextureHandleResidentARB;
    } else if (deviceFeatureSet_.hasExtension(Extensions::BindlessTextureNv)) {
      makeTextureHandleResidentProc_ = iglMakeTextureHandleResidentNV;
    }
    IGL_DEBUG_ASSERT(makeTextureHandleResidentProc_,
                     "No supported function for glMakeTextureHandleResidentARB\n");
  }

  APILOG("glMakeTextureHandleResidentARB(%p)\n", handle);
  GLCALL_PROC(makeTextureHandleResidentProc_, handle);
  GLCHECK_ERRORS();
}

void IContext::makeTextureHandleNonResident(GLuint64 handle) {
  if (makeTextureHandleNonResidentProc_ == nullptr) {
    if (deviceFeatureSet_.hasExtension(Extensions::BindlessTextureArb)) {
      makeTextureHandleNonResidentProc_ = iglMakeTextureHandleNonResidentARB;
    } else if (deviceFeatureSet_.hasExtension(Extensions::BindlessTextureNv)) {
      makeTextureHandleNonResidentProc_ = iglMakeTextureHandleNonResidentNV;
    }
    IGL_DEBUG_ASSERT(makeTextureHandleNonResidentProc_,
                     "No supported function for glMakeTextureHandleNonResidentARB\n");
  }

  APILOG("glMakeTextureHandleNonResidentARB(%p)\n", handle);
  GLCALL_PROC(makeTextureHandleNonResidentProc_, handle);
  GLCHECK_ERRORS();
}

void IContext::dispatchCompute(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ) {
#if IGL_API_LOG
  lastCommandWasCompute_ = true;
#endif
  APILOG("glDispatchCompute(%u, %u, %u) (program: %u) %s %s\n",
         numGroupsX,
         numGroupsY,
         numGroupsZ,
         boundProgram_,
         boundImageTextures().c_str(),
         boundBuffersByIndex().c_str());
  IGLCALL(DispatchCompute)(numGroupsX, numGroupsY, numGroupsZ);
  GLCHECK_ERRORS();
}

void IContext::dispatchComputeIndirect(GLintptr indirect) {
#if IGL_API_LOG
  lastCommandWasCompute_ = true;
#endif
  APILOG("glDispatchComputeIndirect(%zd) (program: %u) %s %s\n",
         static_cast<ptrdiff_t>(indirect),
         boundProgram_,
         boundImageTextures().c_str(),
         boundBuffersByIndex().c_str());
  IGLCALL(DispatchComputeIndirect)(indirect);
  GLCHECK_ERRORS();
}

void IContext::memoryBarrier(GLbitfield barriers) {
  if (memoryBarrierProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::ShaderImageLoadStoreExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::ShaderImageLoadStore)) {
        memoryBarrierProc_ = iglMemoryBarrierEXT;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::ShaderImageLoadStore)) {
      memoryBarrierProc_ = iglMemoryBarrier;
    }
    IGL_DEBUG_ASSERT(memoryBarrierProc_, "No supported function for glMemoryBarrier\n");
  }
  APILOG("glMemoryBarrier(0x%x [%s]) %s\n",
         barriers,
         GL_MEMORY_BARRIER_BITS_TO_STRING(barriers),
         affectedMemoryBarrierObjects(barriers).c_str());
  GLCALL_PROC(memoryBarrierProc_, barriers);
  GLCHECK_ERRORS();
}

void IContext::vertexAttribPointer(GLuint index,
                                   GLint size,
                                   GLenum type,
                                   GLboolean normalized,
                                   GLsizei stride,
                                   const GLvoid* ptr) {
#if IGL_API_LOG
  GLuint unboundBuffer = 0;
  if (index < boundAttributes_.size()) {
    unboundBuffer = boundAttributes_[index];
    boundAttributes_[index] = ptr == nullptr ? boundBuffer(GL_ARRAY_BUFFER) : 0;
  }
#endif
  APILOG("glVertexAttribPointer(%u, %d, %s, %s, %u, %p) unbound (buffer: %u) bound (buffer: %u)\n",
         index,
         size,
         GL_ENUM_TO_STRING(type),
         GL_BOOL_TO_STRING(normalized),
         stride,
         ptr,
         unboundBuffer,
         ptr == nullptr ? boundBuffer(GL_ARRAY_BUFFER) : 0);
  GLCALL(VertexAttribPointer)(index, size, type, normalized, stride, ptr);
  GLCHECK_ERRORS();
}

void IContext::vertexAttribDivisor(GLuint index, GLuint divisor) {
  if (vertexAttribDivisorProc_ == nullptr) {
    if (deviceFeatureSet_.hasInternalRequirement(InternalRequirement::VertexAttribDivisorExtReq)) {
      if (deviceFeatureSet_.hasExtension(Extensions::VertexAttribDivisor)) {
        vertexAttribDivisorProc_ = iglVertexAttribDivisorNV;
      }
    } else if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::VertexAttribDivisor)) {
      vertexAttribDivisorProc_ = iglVertexAttribDivisor;
    }
    IGL_DEBUG_ASSERT(vertexAttribDivisorProc_, "No supported function for glVertexAttribDivisor\n");
  }

  APILOG("glVertexAttribDivisor(%u, %u) (buffer: %u)\n",
         index,
         divisor,
         index < boundAttributes_.size() ? boundAttributes_[index] : 0);
  GLCALL_PROC(vertexAttribDivisorProc_, index, divisor);
  GLCHECK_ERRORS();
}

Result IContext::getLastError() const {
  const GLenum error = lastError_.load(std::memory_order_relaxed);
  return GL_ERROR_TO_RESULT(error);
}

// NOLINTNEXTLINE(misc-no-recursion)
GLenum IContext::checkForErrors(IGL_MAYBE_UNUSED const char* callerName,
                                IGL_MAYBE_UNUSED size_t lineNum) const {
  // GL error state belongs to the calling thread's current context, so the error is reported from
  // this call's own read rather than from lastError_, which another thread may have replaced.
  const GLenum error = getError();
  lastError_.store(error, std::memory_order_relaxed);
#if IGL_DEBUG && !IGL_API_LOG
  thread_local bool gettingMessageLog = false; // Used to avoid recursive entry
  if (error != GL_NO_ERROR && !gettingMessageLog &&
      deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessageCallback)) {
    GLint numMessages = 0;
    getIntegerv(GL_DEBUG_LOGGED_MESSAGES, &numMessages);
    for (GLint i = 0; i < numMessages; ++i) {
      GLint messageLength = 0;
      getIntegerv(GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH, &messageLength);
      std::vector<GLchar> messageBuffer(messageLength);
      GLenum source = 0, type = 0, severity = 0;
      GLuint id = 0;
      GLsizei length = 0;
      gettingMessageLog = true;
      IGL_SCOPE_EXIT {
        gettingMessageLog = false;
      };
      const GLuint count = getDebugMessageLog(
          1, messageLength, &source, &type, &id, &severity, &length, messageBuffer.data());

      if (IGL_DEBUG_VERIFY(count == 1)) {
        logDebugMessage(source, type, id, severity, length, messageBuffer.data());
      }
    }
  }
#endif //  IGL_DEBUG && !IGL_API_LOG

  GL_ASSERT_ERROR(error == GL_NO_ERROR, callerName, lineNum, error);

  return error;
}

// This function has no effect in release mode because the current thinking
// is there will be no need to call glGetError() after each GL call
void IContext::enableAutomaticErrorCheck(bool enable) {
#if IGL_DEBUG
  alwaysCheckError_ = enable;
#endif
}

/** Returns current `callCounter_` value. Exposed for testing only. */
unsigned int IContext::getCallCount() const {
  return callCounter_.load(std::memory_order_relaxed);
}

unsigned int IContext::getCurrentDrawCount() const {
  return drawCallCount_;
}

unsigned int IContext::getShaderCompilationCount() const {
  return shaderCompilationCount_;
}

void IContext::resetCounters() {
  callCounter_.store(0, std::memory_order_relaxed);
}

bool IContext::addRef() {
  const bool ret = isLikelyValidObject();
  if (ret) {
    refCount_.fetch_add(1, std::memory_order_relaxed);
  }
  return ret;
}

bool IContext::releaseRef() {
  const bool ret = isLikelyValidObject();
  if (ret) {
    // Release ordering, so a destructor that observes zero also observes every object's teardown.
    refCount_.fetch_sub(1, std::memory_order_release);
  }
  return ret;
}

UnbindPolicy IContext::getUnbindPolicy() const {
  return unbindPolicy_;
}

void IContext::setUnbindPolicy(UnbindPolicy newValue) {
  unbindPolicy_ = newValue;
}

/**
 * @brief Initializes the OpenGL context by detecting the GL
 *        version, collecting extensions, and configuring the
 *        device feature set.
 *
 * Called by platform-specific context subclasses after creating
 * the native OpenGL context. Sets the context as current,
 * queries GL_VERSION and GL_EXTENSIONS, initializes the device
 * feature set, and enables seamless cube map if supported.
 *
 * @param[out] result Optional output for reporting errors such
 *        as failure to set the current context or to determine
 *        the GL version.
 */
GLVersion IContext::initializeGLVersion(Result* result) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const char* version = reinterpret_cast<const char*>(getString(GL_VERSION));
  if (version == nullptr) {
    IGL_LOG_ERROR("Unable to get GL version string\n");
    Result::setResult(result, Result::Code::RuntimeError, "Unable to get GL version string\n");
    return DeviceFeatureSet::usesOpenGLES() ? GLVersion::v2_0_ES : GLVersion::v2_0;
  }
  const GLVersion glVersion = ::igl::opengl::getGLVersion(version);
  if (glVersion == GLVersion::NotAvailable) {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    Result::setResult(result, Result::Code::RuntimeError, "Unable to get GL version\n");
  }
  return glVersion;
}

void IContext::loadExtensions(std::string& extensions,
                              std::unordered_set<std::string>& supportedExtensions) {
  IGL_PROFILER_FUNCTION();
  if (!deviceFeatureSet_.hasInternalFeature(InternalFeatures::GetStringi)) {
    const GLubyte* extensionStr = getString(GL_EXTENSIONS);
    // If setCurrent() fails, then extensions may be nullptr.
    if (extensionStr) {
      extensions = std::string(reinterpret_cast<const char*>(extensionStr));
    }
    return;
  }
  GLint n = 0;
  getIntegerv(GL_NUM_EXTENSIONS, &n);
  if (IGL_DEBUG_VERIFY(n >= 0)) {
    for (GLuint i = 0; i < static_cast<GLuint>(n); i++) {
      const auto* ext = reinterpret_cast<const char*>(getStringi(GL_EXTENSIONS, i));
      if (ext) {
        supportedExtensions.insert(ext);
      }
    }
  }
}

void IContext::initialize(Result* result) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  setCurrent();
  if (!isCurrentContext()) {
    Result::setResult(result, Result::Code::ArgumentInvalid, "Invalid context, setCurrent failed.");
    return;
  }

  const GLVersion glVersion = initializeGLVersion(result);
  deviceFeatureSet_.initializeVersion(glVersion);

  std::string extensions;
  std::unordered_set<std::string> supportedExtensions;
  loadExtensions(extensions, supportedExtensions);

#if IGL_LOGGING_ENABLED
  const char* version = reinterpret_cast<const char*>(getString(GL_VERSION));
  IGL_LOG_INFO("GL Context Initialized: %p\n", this);
  IGL_LOG_INFO("GL Version: %s\n", version);
  const char* vendor = reinterpret_cast<const char*>(getString(GL_VENDOR));
  IGL_LOG_INFO("GL Vendor: %s\n", (vendor != nullptr) ? vendor : "(null)");
  const char* renderer = reinterpret_cast<const char*>(getString(GL_RENDERER));
  IGL_LOG_INFO("GL Renderer: %s\n", (renderer != nullptr) ? renderer : "(null)");
  if (!extensions.empty() || supportedExtensions.empty()) {
    IGL_LOG_INFO("GL Extensions: %s\n", extensions.c_str());
  } else {
    std::vector<std::string> sortedExtensions(supportedExtensions.begin(),
                                              supportedExtensions.end());
    std::sort(sortedExtensions.begin(), sortedExtensions.end());
    IGL_LOG_INFO("GL Extensions:\n");
    for (const auto& extension : sortedExtensions) {
      IGL_LOG_INFO("- %s\n", extension.c_str());
    }
  }
#endif

  deviceFeatureSet_.initializeExtensions(std::move(extensions), std::move(supportedExtensions));

  if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::SeamlessCubeMap)) {
    enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
  }

#if IGL_DEBUG || IGL_API_LOG
  if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessageCallback)) {
    enable(GL_DEBUG_OUTPUT);
#if IGL_API_LOG
    debugMessageCallback(logOpenGlDebugMessage, nullptr);
#endif // IGL_API_LOG
  }
#endif // IGL_DEBUG || IGL_API_LOG

#if defined(IGL_WITH_TRACY_GPU)
  [[maybe_unused]] constexpr std::string_view kTracyContextName = "IGL OpenGL";
  TracyGpuContext;
  TracyGpuContextName(kTracyContextName.data(), kTracyContextName.size());
#endif
}

const DeviceFeatureSet& IContext::deviceFeatures() const {
  return deviceFeatureSet_;
}

void IContext::apiLogNextNDraws(const unsigned int n) {
  apiLogDrawsLeft_ = n;
}

void IContext::apiLogStart() {
  apiLogEnabled_ = true;
}

void IContext::apiLogEnd() {
  apiLogEnabled_ = false;
}

void IContext::setShouldValidateShaders(bool shouldValidateShaders) {
  shouldValidateShaders_ = shouldValidateShaders;
}

bool IContext::shouldValidateShaders() const {
  return shouldValidateShaders_;
}

void IContext::SynchronizedDeletionQueues::flushDeletionQueue(IContext& context) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);
  if (IGL_DEBUG_VERIFY(context.isCurrentContext() || context.isCurrentSharegroup())) {
    swapScratchDeletionQueues();

    if (!scratchBuffersQueue_.empty()) {
      context.deleteBuffers(static_cast<GLsizei>(scratchBuffersQueue_.size()),
                            scratchBuffersQueue_.data());
      scratchBuffersQueue_.clear();
    }

    for (auto i : scratchUnbindBuffersQueue_) {
      context.bindBuffer(i, 0);
    }
    scratchUnbindBuffersQueue_.clear();

    if (!scratchFramebuffersQueue_.empty()) {
      context.deleteFramebuffers(static_cast<GLsizei>(scratchFramebuffersQueue_.size()),
                                 scratchFramebuffersQueue_.data());
      scratchFramebuffersQueue_.clear();
    }

    if (!scratchRenderbuffersQueue_.empty()) {
      context.deleteRenderbuffers(static_cast<GLsizei>(scratchRenderbuffersQueue_.size()),
                                  scratchRenderbuffersQueue_.data());
      scratchRenderbuffersQueue_.clear();
    }

    if (!scratchVertexArraysQueue_.empty()) {
      context.deleteVertexArrays(static_cast<GLsizei>(scratchVertexArraysQueue_.size()),
                                 scratchVertexArraysQueue_.data());
      scratchVertexArraysQueue_.clear();
    }

    for (auto i : scratchProgramQueue_) {
      context.deleteProgram(i);
    }
    scratchProgramQueue_.clear();

    for (auto i : scratchShaderQueue_) {
      context.deleteShader(i);
    }
    scratchShaderQueue_.clear();

    if (!scratchTexturesQueue_.empty()) {
      context.deleteTextures(static_cast<GLsizei>(scratchTexturesQueue_.size()),
                             scratchTexturesQueue_.data());
      scratchTexturesQueue_.clear();
    }
  }
}

void IContext::SynchronizedDeletionQueues::swapScratchDeletionQueues() {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);

  std::swap(scratchBuffersQueue_, buffersQueue_);
  std::swap(scratchUnbindBuffersQueue_, unbindBuffersQueue_);
  std::swap(scratchFramebuffersQueue_, framebuffersQueue_);
  std::swap(scratchRenderbuffersQueue_, renderbuffersQueue_);
  std::swap(scratchVertexArraysQueue_, vertexArraysQueue_);
  std::swap(scratchProgramQueue_, programQueue_);
  std::swap(scratchShaderQueue_, shaderQueue_);
  std::swap(scratchTexturesQueue_, texturesQueue_);
}

void IContext::SynchronizedDeletionQueues::queueDeleteBuffers(GLsizei n, const GLuint* buffers) {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);
  for (GLsizei i = 0; i < n; ++i) {
    buffersQueue_.push_back(buffers[i]);
  }
}

void IContext::SynchronizedDeletionQueues::queueUnbindBuffer(GLenum target) {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);
  unbindBuffersQueue_.insert(target);
}

void IContext::SynchronizedDeletionQueues::queueDeleteFramebuffers(GLsizei n,
                                                                   const GLuint* framebuffers) {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);
  for (GLsizei i = 0; i < n; ++i) {
    framebuffersQueue_.push_back(framebuffers[i]);
  }
}

void IContext::SynchronizedDeletionQueues::queueDeleteRenderbuffers(GLsizei n,
                                                                    const GLuint* renderbuffers) {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);
  for (GLsizei i = 0; i < n; ++i) {
    renderbuffersQueue_.push_back(renderbuffers[i]);
  }
}

void IContext::SynchronizedDeletionQueues::queueDeleteVertexArrays(GLsizei n,
                                                                   const GLuint* vertexArrays) {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);
  for (GLsizei i = 0; i < n; ++i) {
    vertexArraysQueue_.push_back(vertexArrays[i]);
  }
}

void IContext::SynchronizedDeletionQueues::queueDeleteProgram(GLuint program) {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);
  programQueue_.push_back(program);
}

void IContext::SynchronizedDeletionQueues::queueDeleteShader(GLuint shaderId) {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);
  shaderQueue_.push_back(shaderId);
}

void IContext::SynchronizedDeletionQueues::queueDeleteTextures(GLsizei n, const GLuint* textures) {
  const std::lock_guard<std::mutex> guard(deletionQueueMutex_);
  for (GLsizei i = 0; i < n; ++i) {
    texturesQueue_.push_back(textures[i]);
  }
}
// NOLINTEND(facebook-hte-NullableDereference)
} // namespace igl::opengl
