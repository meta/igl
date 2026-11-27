/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>
#include <webgpu/webgpu.h>
#include <igl/Framebuffer.h>

namespace igl::webgpu {

class WebGPUContext;

/// @brief Implements the igl::IFramebuffer interface for WebGPU. copyBytes*() read back through a
/// staging buffer and wait for the copy; rows are returned bottom-up and channels are not swizzled.
class Framebuffer final : public IFramebuffer {
 public:
  Framebuffer(WebGPUContext& ctx, FramebufferDesc desc);

  [[nodiscard]] std::vector<size_t> getColorAttachmentIndices() const override;
  [[nodiscard]] std::shared_ptr<ITexture> getColorAttachment(size_t index) const override;
  [[nodiscard]] std::shared_ptr<ITexture> getResolveColorAttachment(size_t index) const override;
  [[nodiscard]] std::shared_ptr<ITexture> getDepthAttachment() const override;
  [[nodiscard]] std::shared_ptr<ITexture> getResolveDepthAttachment() const override;
  [[nodiscard]] std::shared_ptr<ITexture> getStencilAttachment() const override;
  [[nodiscard]] FramebufferMode getMode() const override;
  [[nodiscard]] bool isSwapchainBound() const override;

  void copyBytesColorAttachment(ICommandQueue& cmdQueue,
                                size_t index,
                                void* IGL_NONNULL pixelBytes,
                                const TextureRangeDesc& range,
                                size_t bytesPerRow = 0) const override;
  void copyBytesDepthAttachment(ICommandQueue& cmdQueue,
                                void* IGL_NONNULL pixelBytes,
                                const TextureRangeDesc& range,
                                size_t bytesPerRow = 0) const override;
  void copyBytesStencilAttachment(ICommandQueue& cmdQueue,
                                  void* IGL_NONNULL pixelBytes,
                                  const TextureRangeDesc& range,
                                  size_t bytesPerRow = 0) const override;
  void copyTextureColorAttachment(ICommandQueue& cmdQueue,
                                  size_t index,
                                  std::shared_ptr<ITexture> destTexture,
                                  const TextureRangeDesc& range) const override;

  void updateDrawable(std::shared_ptr<ITexture> texture) override;
  void updateDrawable(SurfaceTextures surfaceTextures) override;
  void updateResolveAttachment(std::shared_ptr<ITexture> texture) override;

  [[nodiscard]] const FramebufferDesc& getDesc() const noexcept;

 private:
  void copyBytes(const std::shared_ptr<ITexture>& texture,
                 WGPUTextureAspect aspect,
                 void* IGL_NONNULL pixelBytes,
                 const TextureRangeDesc& range,
                 size_t bytesPerRow) const;

  WebGPUContext& ctx_;
  FramebufferDesc desc_;
};

[[nodiscard]] inline const FramebufferDesc& Framebuffer::getDesc() const noexcept {
  return desc_;
}

} // namespace igl::webgpu
