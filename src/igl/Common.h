/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef IGL_COMMON_H
#define IGL_COMMON_H

#include <cstddef>
#include <cstdint>
#include <ldrutils/lutils/Handle.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector> // NOLINT(facebook-unused-include-check)
#include <igl/Color.h> // IWYU pragma: export
#include <igl/Core.h>
#include <igl/base/Common.h>

#define IGL_ARRAY_NUM_ELEMENTS(x) (sizeof(x) / sizeof((x)[0]))

namespace igl {

// Re-export types from base namespace for backward compatibility
using base::BackendType;
using base::TextureFormat;
using base::TextureType;

// Callback to delete and/or release a pointer
using Deleter = void (*)(void* IGL_NULLABLE);

/// Device Capabilities or Metal Features
constexpr uint32_t IGL_TEXTURE_SAMPLERS_MAX = 16;

// Maximum number of storage images that can be bound to a shader stage. Sampled images and storage
// images are distinct descriptor types with distinct limits, so this is deliberately not derived
// from IGL_TEXTURE_SAMPLERS_MAX. See maxPerStageDescriptorSampledImages (floor of 16) and
// maxPerStageDescriptorStorageImages (floor of 4) in
// https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceLimits.html
// Spelled like its IGL_*_MAX neighbours, not the kIglStorageImagesMax that clang-tidy asks for:
// every site reads this constant beside IGL_TEXTURE_SAMPLERS_MAX.
// NOLINTNEXTLINE(readability-identifier-naming)
constexpr uint32_t IGL_STORAGE_IMAGES_MAX = 16;

constexpr uint32_t IGL_VERTEX_ATTRIBUTES_MAX = 24;

// maximum number of buffers that can be bound to a shader stage
// See maximum number of entries in the buffer argument table, per graphics or kernel function
// in https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
constexpr uint32_t IGL_BUFFER_BINDINGS_MAX = 31;

constexpr uint32_t IGL_UNIFORM_BLOCKS_BINDING_MAX = 16;

// See GL_MAX_COLOR_ATTACHMENTS in
// https://www.khronos.org/registry/OpenGL-Refpages/es3.0/html/glGet.xhtml
// and see maximum number of color render targets in
// https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
constexpr uint32_t IGL_COLOR_ATTACHMENTS_MAX = 4;

// maximum number of subpasses in RenderPassDesc::subpasses
// NOLINTNEXTLINE(readability-identifier-naming)
constexpr uint32_t IGL_SUBPASSES_MAX = 4;

// See https://www.khronos.org/registry/OpenGL-Refpages/es3.1/html/glDrawElementsIndirect.xhtml,
// https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkDrawIndexedIndirectCommand.html,
// or https://developer.apple.com/documentation/metal/mtldrawindexedprimitivesindirectarguments.
constexpr uint32_t IGL_DRAW_ELEMENTS_INDIRECT_COMMAND_SIZE = 4 * 5;

enum class ResourceStorage {
  Invalid, /// Invalid sharing mode
  Private, /// Memory private to GPU access (fastest)
  Shared, /// Memory shared between CPU and GPU
  Managed, /// Memory pair synchronized between CPU and GPU
  Memoryless /// Memory that only the GPU can access and that exists only during a render pass.
             /// Memoryless textures must be single-mip 2D or 2D array attachments (optionally also
             /// InputAttachment) that are not exportable and use optimal tiling. Load actions must
             /// be Clear or DontCare and store actions DontCare or MsaaResolve. Uploading or
             /// reading back the contents is invalid. Backends without tile memory fall back to
             /// private memory.
};

enum class CullMode : uint8_t { Disabled, Front, Back };
enum class WindingMode : uint8_t { Clockwise, CounterClockwise };
enum class NormalizedZRange : uint8_t { NegOneToOne, ZeroToOne };

enum class PrimitiveType : uint8_t {
  Point,
  Line,
  LineStrip,
  Triangle,
  TriangleStrip,
};

struct Result {
  enum class Code {
    // No error
    Ok,

    /// Bad argument, e.g. invalid buffer/texture/bind type
    ArgumentInvalid,
    /// nullptr input for non-null argument
    ArgumentNull,
    /// Argument out of range, e.g. attachment/mip-level/aniso out of range
    ArgumentOutOfRange,
    /// Cannot execute operation in current state
    InvalidOperation,
    /// Feature is not supported on current hardware or software
    Unsupported,
    /// Feature has not yet been implemented into IGL
    Unimplemented,

    /// Something bad happened internally but we don't know what
    RuntimeError,

    /// GPU device was lost — connection to the hardware was severed (e.g. hardware disconnect).
    /// After this, all GPU operations will fail until the device is recreated.
    DeviceLost,
  };

  Code code = Code::Ok;
  std::string message;
  explicit Result() = default;
  explicit Result(Code code, const char* IGL_NULLABLE message = "") :
    code(code), message(message ? static_cast<const char*>(message) : "") {}
  explicit Result(Code code, std::string message) : code(code), message(std::move(message)) {}

  [[nodiscard]] bool isOk() const {
    return code == Result::Code::Ok;
  }

  static void setResult(Result* IGL_NULLABLE outResult,
                        Code code,
                        const std::string& message = "") {
    if (outResult != nullptr) {
      outResult->code = code;
      outResult->message = message;
    }
  }

  static void setResult(Result* IGL_NULLABLE outResult, const Result& sourceResult) {
    if (outResult != nullptr) {
      *outResult = sourceResult;
    }
  }

  static void setResult(Result* IGL_NULLABLE outResult, Result&& sourceResult) {
    if (outResult != nullptr) {
      *outResult = std::move(sourceResult);
    }
  }

  static void setOk(Result* IGL_NULLABLE outResult) {
    if (outResult != nullptr) {
      outResult->code = Code::Ok;
      outResult->message.clear();
    }
  }
};

enum class BackendFlavor : uint8_t {
  Invalid,
  OpenGL,
  OpenGL_ES,
  Metal,
  Vulkan,
  D3D12,
  // @fb-only
  WebGPU,
};

std::string BackendTypeToString(BackendType backendType);

///--------------------------------------
/// MARK: - ScissorRect

struct ScissorRect {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  [[nodiscard]] bool isNull() const {
    return width == 0 && height == 0;
  }
};

static_assert(sizeof(ScissorRect) == 4 * sizeof(uint32_t));
static_assert(std::is_trivially_copyable_v<ScissorRect>);

///--------------------------------------
/// MARK: - Size

struct Size {
  float width = 0.0f;
  float height = 0.0f;

  Size() = default;
  Size(float width, float height) : width(width), height(height) {}
};

struct Dimensions {
  Dimensions() = default;
  Dimensions(uint32_t w, uint32_t h, uint32_t d) : width(w), height(h), depth(d) {}
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depth = 0;
};

inline bool operator==(const Dimensions& lhs, const Dimensions& rhs) {
  return (lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth);
}

inline bool operator!=(const Dimensions& lhs, const Dimensions& rhs) {
  return !operator==(lhs, rhs);
}

inline bool operator==(const Size& lhs, const Size& rhs) {
  return (lhs.width == rhs.width && lhs.height == rhs.height);
}

inline bool operator!=(const Size& lhs, const Size& rhs) {
  return !operator==(lhs, rhs);
}

///--------------------------------------
/// MARK: - Viewport

/// x, y, width, height in pixels. minDepth, maxDepth are in [0.0, 1.0]
struct Viewport {
  float x = 0.0f;
  float y = 0.0f;
  float width = 1.0f;
  float height = 1.0f;
  float minDepth = 0.0f;
  float maxDepth = 1.0f;
};

static_assert(sizeof(Viewport) == 6 * sizeof(float));
static_assert(std::is_trivially_copyable_v<Viewport>);

inline bool operator==(const Viewport& lhs, const Viewport& rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width && lhs.height == rhs.height &&
         lhs.minDepth == rhs.minDepth && lhs.maxDepth == rhs.maxDepth;
}

inline bool operator!=(const Viewport& lhs, const Viewport& rhs) {
  return !operator==(lhs, rhs);
}

constexpr Viewport kInvalidViewport =
    {.x = -1.0f, .y = -1.0f, .width = -1.0f, .height = -1.0f, .minDepth = -1.0f, .maxDepth = -1.0f};

///--------------------------------------
/// MARK: - Enum utilities

// Get value of enum by stripping enum class type
template<typename E>
constexpr std::underlying_type_t<E> EnumToValue(E enumerator) noexcept {
  return static_cast<std::underlying_type_t<E>>(enumerator);
}

///--------------------------------------
/// MARK: - ScopeGuard

// a very minimalistic version of folly/ScopeGuard.h
namespace detail {

enum class ScopeGuardOnExit {};

template<typename T>
class ScopeGuard {
 public:
  explicit ScopeGuard(const T& fn) : fn_(fn) {}
  ~ScopeGuard() {
    fn_();
  }

 private:
  T fn_;
};

template<typename T>
// NOLINTNEXTLINE(readability-named-parameter)
ScopeGuard<T> operator+(ScopeGuardOnExit /*guard*/, T&& fn) {
  return ScopeGuard<T>(std::forward<T>(fn));
}

} // namespace detail

#define IGL_SCOPE_EXIT \
  auto FB_ANONYMOUS_VARIABLE(SCOPE_EXIT_STATE) = ::igl::detail::ScopeGuardOnExit() + [&]() noexcept

///--------------------------------------
/// MARK: - optimizedMemcpy

// Optimized version of memcpy() that allows to copy smallest uniforms in the most efficient way.
// Other sizes utilize a libc memcpy() implementation. It's not a universal function and expects to
// have a proper alignment for data!
void optimizedMemcpy(void* IGL_NULLABLE dst, const void* IGL_NULLABLE src, size_t size);

///--------------------------------------
/// MARK: - Handle

// specialized with dummy structs for type safety
using BindGroupTextureHandle = ldr::Handle<struct BindGroupTextureTag>;
using BindGroupBufferHandle = ldr::Handle<struct BindGroupBufferTag>;
using TextureHandle = ldr::Handle<struct TextureTag>;
using SamplerHandle = ldr::Handle<struct SamplerTag>;
using DepthStencilStateHandle = ldr::Handle<struct DepthStencilStateTag>;

class IDevice;

// forward declarations to access incomplete type IDevice
void destroy(IDevice* IGL_NULLABLE device, BindGroupTextureHandle handle);
void destroy(IDevice* IGL_NULLABLE device, BindGroupBufferHandle handle);
void destroy(IDevice* IGL_NULLABLE device, TextureHandle handle);
void destroy(IDevice* IGL_NULLABLE device, SamplerHandle handle);
void destroy(IDevice* IGL_NULLABLE device, DepthStencilStateHandle handle);

///--------------------------------------
/// MARK: - Holder

// RAII wrapper around Handle<>; based on:
// https://github.com/corporateshark/lightweightvk/blob/72f3508b52d9e81534fcb0ca64c6a0fc1aabbf78/lvk/LVK.h#L163
template<typename HandleType>
class Holder final {
 public:
  Holder() noexcept = default;
  Holder(IDevice* IGL_NULLABLE device, HandleType handle) noexcept :
    device_(device), handle_(handle) {}
  ~Holder() {
    igl::destroy(device_, handle_);
  }
  Holder(const Holder&) = delete;
  Holder(Holder&& other) noexcept : device_(other.device_), handle_(other.handle_) {
    other.device_ = nullptr;
    other.handle_ = HandleType{};
  }
  Holder& operator=(const Holder&) = delete;
  Holder& operator=(Holder&& other) noexcept {
    std::swap(device_, other.device_);
    std::swap(handle_, other.handle_);
    return *this;
  }
  Holder& operator=(std::nullptr_t) {
    this->reset();
    return *this;
  }

  [[nodiscard]] operator HandleType() const noexcept {
    return handle_;
  }

  [[nodiscard]] bool valid() const noexcept {
    return handle_.valid();
  }

  [[nodiscard]] bool empty() const noexcept {
    return handle_.empty();
  }

  void reset() {
    igl::destroy(device_, handle_);
    device_ = nullptr;
    handle_ = HandleType{};
  }

  [[nodiscard]] HandleType release() noexcept {
    device_ = nullptr;
    return std::exchange(handle_, HandleType{});
  }

  [[nodiscard]] uint32_t gen() const noexcept {
    return handle_.gen();
  }
  [[nodiscard]] uint32_t index() const noexcept {
    return handle_.index();
  }
  [[nodiscard]] void* IGL_NULLABLE indexAsVoid() const noexcept {
    return handle_.indexAsVoid();
  }

 private:
  IDevice* IGL_NULLABLE device_ = nullptr;
  HandleType handle_ = {};
};

} // namespace igl

#endif // IGL_COMMON_H
