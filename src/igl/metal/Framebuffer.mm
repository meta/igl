/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/metal/Framebuffer.h>

#include <utility>
#include <vector>
#include <igl/Macros.h>
#include <igl/metal/CommandQueue.h>
#include <igl/metal/Device.h>
#include <igl/metal/Texture.h>

namespace igl::metal {

Framebuffer::Framebuffer(FramebufferDesc value) : value_(std::move(value)) {}

std::vector<size_t> Framebuffer::getColorAttachmentIndices() const {
  IGL_PROFILER_FUNCTION();
  std::vector<size_t> indices;

  for (size_t i = 0; i != IGL_COLOR_ATTACHMENTS_MAX; i++) {
    if (value_.colorAttachments[i].texture || value_.colorAttachments[i].resolveTexture) {
      indices.push_back(i);
    }
  }

  return indices;
}

std::shared_ptr<ITexture> Framebuffer::getColorAttachment(size_t index) const {
  IGL_DEBUG_ASSERT(index < IGL_COLOR_ATTACHMENTS_MAX);
  return value_.colorAttachments[index].texture;
}

std::shared_ptr<ITexture> Framebuffer::getResolveColorAttachment(size_t index) const {
  IGL_DEBUG_ASSERT(index < IGL_COLOR_ATTACHMENTS_MAX);
  return value_.colorAttachments[index].resolveTexture;
}

std::shared_ptr<ITexture> Framebuffer::getDepthAttachment() const {
  return value_.depthAttachment.texture;
}

std::shared_ptr<ITexture> Framebuffer::getResolveDepthAttachment() const {
  return value_.depthAttachment.resolveTexture;
}

std::shared_ptr<ITexture> Framebuffer::getStencilAttachment() const {
  return value_.stencilAttachment.texture;
}

bool Framebuffer::isSwapchainBound() const {
  if (auto tex = getColorAttachment(0)) {
    return tex->isSwapchainTexture();
  }
  return false;
}

void Framebuffer::copyBytesColorAttachment(ICommandQueue& cmdQueue,
                                           size_t index,
                                           void* pixelBytes,
                                           const TextureRangeDesc& range,
                                           size_t bytesPerRow) const {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(index < IGL_COLOR_ATTACHMENTS_MAX);
  IGL_DEBUG_ASSERT(range.numFaces == 1, "range.numFaces MUST be 1");
  IGL_DEBUG_ASSERT(range.numLayers == 1, "range.numLayers MUST be 1");
  IGL_DEBUG_ASSERT(range.numMipLevels == 1, "range.numMipLevels MUST be 1");
  const auto& colorAttachment = value_.colorAttachments[index];

  if (IGL_DEBUG_VERIFY(colorAttachment.texture != nullptr)) {
    auto texture = colorAttachment.texture;
    copyBytes(cmdQueue, texture, pixelBytes, range, bytesPerRow);
  }
}

void Framebuffer::copyBytesDepthAttachment(ICommandQueue& cmdQueue,
                                           void* pixelBytes,
                                           const TextureRangeDesc& range,
                                           size_t bytesPerRow) const {
  IGL_PROFILER_FUNCTION();
  auto texture = value_.depthAttachment.texture;
  copyBytes(cmdQueue, texture, pixelBytes, range, bytesPerRow);
}

void Framebuffer::copyBytesStencilAttachment(ICommandQueue& cmdQueue,
                                             void* pixelBytes,
                                             const TextureRangeDesc& range,
                                             size_t bytesPerRow) const {
  IGL_PROFILER_FUNCTION();
  auto texture = value_.stencilAttachment.texture;
  copyBytes(cmdQueue, texture, pixelBytes, range, bytesPerRow);
}

/// copyTextureColorAttachment
///
/// This function does not do any cross-queue synchronization. At the time
/// of writing, we are assuming IGL operates on one queue per context
void Framebuffer::copyTextureColorAttachment(ICommandQueue& cmdQueue,
                                             size_t index,
                                             std::shared_ptr<ITexture> destTexture,
                                             const TextureRangeDesc& range) const {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(index < IGL_COLOR_ATTACHMENTS_MAX);
  const auto& colorAttachment = value_.colorAttachments[index];

  if (IGL_DEBUG_VERIFY(colorAttachment.texture != nullptr)) {
    auto srcTexture = colorAttachment.texture;
    id<MTLTexture> srcMtlTexture = static_cast<Texture&>(*srcTexture).get();
    id<MTLTexture> dstMtlTexture = static_cast<Texture&>(*destTexture).get();
    if (IGL_DEBUG_VERIFY(srcMtlTexture && dstMtlTexture)) {
      auto iglMtlCmdQueue = static_cast<CommandQueue&>(cmdQueue);

      id<MTLCommandBuffer> cmdBuf = [iglMtlCmdQueue.get() commandBuffer];
      id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf blitCommandEncoder];

      // NOLINTNEXTLINE(clang-analyzer-nullability.NullabilityBase)
      [blitEncoder copyFromTexture:srcMtlTexture
                       sourceSlice:range.layer
                       sourceLevel:range.mipLevel
                      sourceOrigin:MTLOriginMake(range.x, range.y, 0)
                        sourceSize:MTLSizeMake(range.width, range.height, 1)
                         toTexture:dstMtlTexture
                  destinationSlice:range.layer
                  destinationLevel:range.mipLevel
                 destinationOrigin:MTLOriginMake(range.x, range.y, 0)];
      [blitEncoder endEncoding];
      [cmdBuf commit];
    }
  }
}

void Framebuffer::copyBytes(ICommandQueue& cmdQueue,
                            const std::shared_ptr<ITexture>& iglTexture,
                            void* pixelBytes,
                            const TextureRangeDesc& range,
                            size_t bytesPerRow) const {
  IGL_PROFILER_FUNCTION();
  auto mtlTexture = std::static_pointer_cast<Texture>(iglTexture);
  if (bytesPerRow == 0) {
    bytesPerRow = iglTexture->getProperties().getBytesPerRow(range);
  }
  if (canCopy(cmdQueue, mtlTexture->get(), range)) {
    mtlTexture->getBytes(range, pixelBytes, bytesPerRow);
  } else {
    auto& mtlCommandQueue = static_cast<CommandQueue&>(cmdQueue);
    auto& mtlDevice = mtlCommandQueue.getDevice();

    // Once Intel Mac support is EOL remove the isAppleGpu() check and also the
    // sychronizeTexture call below.
    Result result;
    const TextureDesc desc{
        .width = iglTexture->getDimensions().width,
        .height = iglTexture->getDimensions().height,
        .type = iglTexture->getType(),
        .format = iglTexture->getProperties().format,
        .storage = mtlDevice.isAppleGpu() ? ResourceStorage::Shared : ResourceStorage::Managed,
        .debugName = "stageTexture",
    };

    // 1. Create a shared stage texture
    auto stageTexture = mtlDevice.createTexture(desc, &result);

    if (!IGL_DEBUG_VERIFY(stageTexture && result.isOk())) {
      return;
    }

    // 2. Copy data from the private texture to the stage texture by MTLBlitCommandEncoder
    id<MTLTexture> srcMtlTexture = static_cast<Texture&>(*iglTexture).get();
    id<MTLTexture> dstMtlTexture = static_cast<Texture&>(*stageTexture).get();
    if (IGL_DEBUG_VERIFY(srcMtlTexture && dstMtlTexture)) {
      auto iglMtlCmdQueue = static_cast<CommandQueue&>(cmdQueue);

      id<MTLCommandBuffer> cmdBuf = [iglMtlCmdQueue.get() commandBuffer];
      id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf blitCommandEncoder];

      // NOLINTNEXTLINE(clang-analyzer-nullability.NullabilityBase)
      [blitEncoder copyFromTexture:srcMtlTexture
                       sourceSlice:range.layer
                       sourceLevel:range.mipLevel
                      sourceOrigin:MTLOriginMake(range.x, range.y, 0)
                        sourceSize:MTLSizeMake(range.width, range.height, 1)
                         toTexture:dstMtlTexture
                  destinationSlice:range.layer
                  destinationLevel:range.mipLevel
                 destinationOrigin:MTLOriginMake(range.x, range.y, 0)];

#if IGL_PLATFORM_MACOSX
      if (desc.storage == ResourceStorage::Managed) {
        [blitEncoder synchronizeTexture:dstMtlTexture slice:range.layer level:range.mipLevel];
      }
#endif

      [blitEncoder endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // 3. Read data from the shared stage texture
    auto& mtlStageTexture = static_cast<Texture&>(*stageTexture);
    mtlStageTexture.getBytes(range, pixelBytes, bytesPerRow);
  }
}

void Framebuffer::updateDrawable(std::shared_ptr<ITexture> texture) {
  IGL_PROFILER_FUNCTION();
  if (getColorAttachment(0) != texture) {
    if (!texture) {
      value_.colorAttachments[0] = {};
    } else {
      value_.colorAttachments[0].texture = std::move(texture);
    }
  }
}

void Framebuffer::updateDrawable(SurfaceTextures surfaceTextures) {
  IGL_PROFILER_FUNCTION();
  updateDrawable(std::move(surfaceTextures.color));
  if (surfaceTextures.depth && surfaceTextures.depth->getProperties().hasStencil()) {
    if (getStencilAttachment() != surfaceTextures.depth) {
      value_.stencilAttachment.texture = surfaceTextures.depth;
    }
  } else {
    value_.stencilAttachment.texture = nullptr;
  }
  if (getDepthAttachment() != surfaceTextures.depth) {
    value_.depthAttachment.texture = std::move(surfaceTextures.depth);
  }
}

void Framebuffer::updateResolveAttachment(std::shared_ptr<ITexture> texture) {
  if (getResolveColorAttachment(0) != texture) {
    value_.colorAttachments[0].resolveTexture = std::move(texture);
  }
}

FramebufferMode Framebuffer::getMode() const {
  return value_.mode;
}

} // namespace igl::metal
