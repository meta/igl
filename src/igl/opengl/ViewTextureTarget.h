/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/Core.h>
#include <igl/opengl/Texture.h>

namespace igl::opengl {

namespace egl {
class PlatformDevice;
} // namespace egl

namespace wgl {
class PlatformDevice;
} // namespace wgl

namespace webgl {
class PlatformDevice;
} // namespace webgl

// ViewTextureTarget is a special implementation of opengl::Texture that's only available on certain
// platforms. It represents the "texture" associated with the default framebuffer on OpenGL (i.e.
// framebuffer with ID 0), which is not available on some platforms such as iOS.
class ViewTextureTarget final : public Texture {
  friend class egl::PlatformDevice;
  friend class wgl::PlatformDevice;
  friend class webgl::PlatformDevice;

 public:
  using Super = Texture;
  ViewTextureTarget(IContext& context, TextureFormat format) : Super(context, format) {}

  // ITexture overrides
  [[nodiscard]] TextureType getType() const override;
  [[nodiscard]] TextureDesc::TextureUsage getUsage() const override;

  // Texture overrides
  void bind() override;
  void bindImage(size_t /*unit*/) override;
  void unbind() override;
  void attachAsColor(uint32_t index, const AttachmentParams& params) override;
  void detachAsColor(uint32_t index, bool read) override;
  void attachAsDepth(const AttachmentParams& params) override;
  void detachAsDepth(bool read) override;
  void attachAsStencil(const AttachmentParams& params) override;
  void detachAsStencil(bool read) override;

  [[nodiscard]] bool isImplicitStorage() const override;

  // @fb-only
  [[nodiscard]] GLuint getId() const override;
};

[[nodiscard]] inline GLuint ViewTextureTarget::getId() const {
  IGL_DEBUG_ASSERT_NOT_REACHED();
  return 0;
}

} // namespace igl::opengl
