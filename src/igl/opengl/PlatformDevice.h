/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/Common.h>
#include <igl/PlatformDevice.h>
#include <igl/Texture.h>
#include <igl/opengl/GLIncludes.h> // IWYU pragma: keep

namespace igl {

class IFramebuffer;
struct FramebufferDesc;

namespace opengl {

class DestructionGuard;
class Device;
class Framebuffer;
class IContext;
class TextureBufferExternal;

/// opengl::PlatformDevice enables transitioning to IGL from legacy OpenGL code.
///
/// It mimics IDevice's factory methods, but ensures the return types are the most-derived,
/// OpenGL-specific versions of those types. For example, here are the accessors for frame buffer.
/// opengl::PlatformDevice allows us to create the OpenGL-specific resource (opengl::Framebuffer)
/// in a type-safe way (i.e. no unsafe downcasting by the caller):
///
///   class IDevice {
///    public:
///     virtual shared_ptr<IFramebuffer> createFramebuffer();
///   };
///
///   class opengl::PlatformDevice {
///    public:
///     shared_ptr<opengl::Framebuffer> createFramebuffer() override; // Note different return type
///   };
///
/// When transitioning to IGL from OpenGL, we recommend you do so in 2 phases:
///
/// 1. Use OpenGL-flavor of IGL resource, e.g. replace texture id's with igl::opengl::Texture.
/// 2. Transition to pure IGL resources. Once IGL command buffers are adopted, use igl::ITexture
/// instead of the OpenGL version.
///
/// With PlatformDevice, refactoring between phase 1 and 2 becomes easier. For example, imagine
/// we have the following struct that holds onto a device instance. In Phase 1, device_'s type is
/// opengl::PlatformDevice; in Phase 2, the type is IDevice.
///
///   struct RenderSession {
///     // Phase 1
///     opengl::PlatformDevice* device_; // <=== [AA]
///
///     // Phase 2: replace above with the following:
///     // IDevice *device_;             // <=== [BB]
///   }
///
/// We can write code like the following and easily change device_'s type between
/// opengl::PlatformDevice and IDevice:
///
///   auto framebuffer = device_ -> createFramebuffer();
///
/// To go from Phase 1 to Phase 2, we simply have to remove the line marked [AA] and
/// uncomment the line marked [BB] — and that's it!
///
/// In Phase 1, framebuffer's type is shared_ptr<opengl::Framebuffer>, so you can access the
/// raw OpenGL framebuffer id. In Phase 2, framebuffer's type is shared_ptr<IFramebuffer>;
/// you are submitting GPU commands via IGL instead of via OpenGL.
///
class PlatformDevice : public IPlatformDevice {
 public:
  static constexpr PlatformDeviceType kType = PlatformDeviceType::OpenGL;

  explicit PlatformDevice(Device& owner) : owner_(owner) {}

  std::shared_ptr<Framebuffer> createFramebuffer(const FramebufferDesc& desc,
                                                 Result* outResult) const;
  [[nodiscard]] std::shared_ptr<Framebuffer> createCurrentFramebuffer() const;
  [[nodiscard]] std::unique_ptr<TextureBufferExternal> createTextureBufferExternal(
      GLuint textureID,
      GLenum target,
      TextureDesc::TextureUsage usage,
      GLsizei width,
      GLsizei height,
      TextureFormat format,
      GLsizei numLayers = 1) const;
  [[nodiscard]] DestructionGuard getDestructionGuard() const;
  [[nodiscard]] IContext& getContext() const;
  [[nodiscard]] const std::shared_ptr<IContext>& getSharedContext() const;
  void blitFramebuffer(const std::shared_ptr<IFramebuffer>& src,
                       int srcLeft,
                       int srcTop,
                       int srcRight,
                       int srcBottom,
                       const std::shared_ptr<IFramebuffer>& dst,
                       int dstLeft,
                       int dstTop,
                       int dstRight,
                       int dstBottom,
                       GLbitfield mask,
                       Result* outResult) const;
  static void blitFramebuffer(const std::shared_ptr<IFramebuffer>& src,
                              int srcLeft,
                              int srcTop,
                              int srcRight,
                              int srcBottom,
                              const std::shared_ptr<IFramebuffer>& dst,
                              int dstLeft,
                              int dstTop,
                              int dstRight,
                              int dstBottom,
                              GLbitfield mask,
                              IContext& ctx,
                              Result* outResult);

 protected:
  [[nodiscard]] bool isType(PlatformDeviceType t) const noexcept override {
    return t == kType;
  }

  Device& owner_;
};

} // namespace opengl
} // namespace igl
