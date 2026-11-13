/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <CoreVideo/CVImageBuffer.h>
#import <CoreVideo/CVMetalTextureCache.h>
#import <Metal/Metal.h>
#import <QuartzCore/CALayer.h>
#import <QuartzCore/CAMetalLayer.h>
#include <igl/Common.h>
#include <igl/PlatformDevice.h>
#include <igl/Texture.h>
#include <igl/metal/SamplerState.h>

namespace igl {

struct FramebufferDesc;
struct ShaderStagesDesc;
class IShaderStages;

namespace metal {

class Device;
class DynamicLibrary;
class Framebuffer;

class PlatformDevice final : public IPlatformDevice {
 public:
  static constexpr PlatformDeviceType kType = igl::PlatformDeviceType::Metal;

  PlatformDevice(Device& device);
  ~PlatformDevice() override;

  std::shared_ptr<SamplerState> createSamplerState(const SamplerStateDesc& desc,
                                                   Result* outResult) const;
  std::shared_ptr<Framebuffer> createFramebuffer(const FramebufferDesc& desc,
                                                 Result* outResult) const;

  /// Creates a texture from a native drawable
  /// @param nativeDrawable drawable. For Metal, drawable MUST be CAMetalDrawable
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::unique_ptr<ITexture> createTextureFromNativeDrawable(id<CAMetalDrawable> nativeDrawable,
                                                            Result* outResult);

  /// Creates a texture from a native drawable
  /// @param nativeDrawable drawable. For Metal, drawable MUST be MTLTexture
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::unique_ptr<ITexture> createTextureFromNativeDrawable(id<MTLTexture> nativeDrawable,
                                                            Result* outResult);

  /// Creates a texture from a native drawable surface
  /// @param nativeDrawable drawable surface. For Metal, drawable MUST be CAMetalLayer
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::unique_ptr<ITexture> createTextureFromNativeDrawable(CALayer* nativeDrawable,
                                                            Result* outResult);

  /// Creates a depth texture from a native depth stencil texture
  /// @param depthStencilTexture Native depth stencil texture.
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::unique_ptr<ITexture> createTextureFromNativeDepth(id<MTLTexture> depthStencilTexture,
                                                         Result* outResult);

  /// Creates a texture from a native PixelBuffer. Uses the backing PixelBuffer for width and
  /// height.
  /// @param sourceImage source image
  /// @param format the format of the source texture
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::unique_ptr<ITexture> createTextureFromNativePixelBuffer(CVImageBufferRef sourceImage,
                                                               TextureFormat format,
                                                               size_t planeIndex,
                                                               Result* outResult);

  /// Creates a texture from a native PixelBuffer with auto-detected format.
  /// The texture format is inferred from the CVPixelBuffer's pixel format type and plane index,
  /// matching the OpenGL backend's behavior of automatic format detection.
  /// @param sourceImage source image
  /// @param planeIndex the plane index to generate the texture
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::unique_ptr<ITexture> createTextureFromNativePixelBuffer(CVImageBufferRef sourceImage,
                                                               size_t planeIndex,
                                                               Result* outResult);

  /// Creates a texture from a native PixelBuffer
  /// @param sourceImage source image
  /// @param format the format of the source texture
  /// @param width the width of the texture
  /// @param height the height of the texture
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::unique_ptr<ITexture> createTextureFromNativePixelBufferWithSize(CVImageBufferRef sourceImage,
                                                                       TextureFormat format,
                                                                       size_t width,
                                                                       size_t height,
                                                                       size_t planeIndex,
                                                                       Result* outResult);

  /// Get a size of a given native drawable surface.
  /// @param nativeDrawable drawable surface. For Metal, drawable MUST be CAMetalLayer
  /// @param outResult Optional result.
  /// @return An accurate size that is suitable for rendering or zero if failed to get a size.
  Size getNativeDrawableSize(CALayer* nativeDrawable, Result* outResult);

  /// Get a texture format that is suitable to render a given drawable surface.
  /// @param nativeDrawable drawable surface. For Metal, drawable MUST be CAMetalLayer
  /// @param outResult Optional result.
  /// @return An accurate pixel format that is suitable for rendering or invalid if failed.
  TextureFormat getNativeDrawableTextureFormat(CALayer* nativeDrawable, Result* outResult);

  void flushNativeTextureCache() const;

  /// Destroys the cache; createTextureFromNativePixelBuffer() recreates it lazily.
  /// The caller must serialize cache access, complete GPU work, and release all
  /// textures created through the cache before calling resetNativeTextureCache().
  void resetNativeTextureCache();

  CVMetalTextureCacheRef getTextureCache();

  /**
   * @brief Whether render and compute pipelines on this device can link dynamic libraries: macOS
   * 12 / iOS 15, and a GPU with MTLDevice.supportsDynamicLibraries and
   * supportsRenderDynamicLibraries.
   */
  [[nodiscard]] bool supportsDynamicLibraries() const noexcept;

  /**
   * @brief Creates a dynamic library from a dynamic .metallib, as `metal -dynamiclib` builds one.
   * When Metal links a pipeline, it still looks the library up at its install name, so a dynamic
   * library has to exist at that path, typically a default copy shipped with the app at an
   * `@executable_path/...` install name. A pipeline that links this library uses it instead of
   * that copy.
   * @return The library, or nullptr: with Result::Code::Unsupported when supportsDynamicLibraries()
   * is false, or Result::Code::ArgumentInvalid when `data` is not a dynamic library.
   */
  std::shared_ptr<DynamicLibrary> createDynamicLibrary(const void* data,
                                                       size_t length,
                                                       const std::string& debugName,
                                                       Result* outResult) const;

  /**
   * @brief Creates shader stages whose pipelines link `linkedLibraries` in place of the libraries
   * with the same install names that the stages' functions were built against. A pipeline can so
   * switch to another library without recompiling its functions. The stages go to
   * IDevice::createRenderPipeline() and IDevice::createComputePipeline() as usual; mesh pipelines
   * cannot link dynamic libraries.
   * @return The stages, or nullptr: with Result::Code::Unsupported when supportsDynamicLibraries()
   * is false and `linkedLibraries` is not empty, or Result::Code::ArgumentNull when one of them is
   * null.
   */
  std::unique_ptr<IShaderStages> createShaderStages(
      const ShaderStagesDesc& desc,
      std::vector<std::shared_ptr<DynamicLibrary>> linkedLibraries,
      Result* outResult) const;

 protected:
  [[nodiscard]] bool isType(PlatformDeviceType t) const noexcept override {
    return t == kType;
  }

 private:
  Device& device_;
  CVMetalTextureCacheRef textureCache_ = nullptr;
};

} // namespace metal
} // namespace igl
