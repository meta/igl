/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <EGL/egl.h>
#include <igl/Texture.h>
#include <igl/opengl/PlatformDevice.h>

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
struct AHardwareBuffer;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

namespace igl::opengl {

class ViewTextureTarget;

namespace egl {

class Device;
// NOLINTNEXTLINE(facebook-unused-forward-decls)
class Context;

// @fb-only
class PlatformDevice : public opengl::PlatformDevice {
 public:
  static constexpr PlatformDeviceType kType = PlatformDeviceType::OpenGLEgl;

  explicit PlatformDevice(Device& owner);
  ~PlatformDevice() override = default;

  /// Returns a texture representing the EGL Surface associated with this device's context.
  std::shared_ptr<ITexture> createTextureFromNativeDrawable(TextureFormat colorTextureFormat,
                                                            Result* IGL_NULLABLE outResult);

  /// Returns a texture representing the EGL Surface associated with this device's context.
  std::shared_ptr<ITexture> createTextureFromNativeDrawable(Result* IGL_NULLABLE outResult) {
    return createTextureFromNativeDrawable(TextureFormat::RGBA_UNorm8, outResult);
  }

  std::shared_ptr<ITexture> createTextureFromNativeDrawable(int width,
                                                            int height,
                                                            TextureFormat colorTextureFormat,
                                                            Result* IGL_NULLABLE outResult);

  std::shared_ptr<ITexture> createTextureFromNativeDrawable(int width,
                                                            int height,
                                                            Result* IGL_NULLABLE outResult) {
    return createTextureFromNativeDrawable(width, height, TextureFormat::RGBA_UNorm8, outResult);
  }

  /// Returns a texture representing the EGL depth texture associated with this device's context.
  std::shared_ptr<ITexture> createTextureFromNativeDepth(TextureFormat depthTextureFormat,
                                                         Result* IGL_NULLABLE outResult);

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
  /// returns a android::NativeHWTextureBuffer on platforms supporting it
  /// this texture allows CPU and GPU to both read/write memory
  std::shared_ptr<ITexture> createTextureWithSharedMemory(const TextureDesc& desc,
                                                          Result* IGL_NULLABLE outResult) const;
  std::shared_ptr<ITexture> createTextureWithSharedMemory(AHardwareBuffer* IGL_NONNULL buffer,
                                                          Result* IGL_NULLABLE outResult) const;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

  /// This function must be called every time the currently bound EGL read and/or draw surfaces
  /// change, in order to notify IGL of these changes.
  void updateSurfaces(EGLSurface readSurface,
                      EGLSurface drawSurface,
                      Result* IGL_NULLABLE outResult);

  EGLSurface IGL_NULLABLE createSurface(NativeWindowType nativeWindow,
                                        Result* IGL_NULLABLE outResult);

  EGLSurface IGL_NULLABLE getReadSurface(Result* IGL_NULLABLE outResult);

  void setPresentationTime(long long presentationTimeNs, Result* IGL_NULLABLE outResult);

 protected:
  [[nodiscard]] bool isType(PlatformDeviceType t) const noexcept override;

 private:
  std::shared_ptr<ViewTextureTarget> drawableTexture_;
};

} // namespace egl
} // namespace igl::opengl
