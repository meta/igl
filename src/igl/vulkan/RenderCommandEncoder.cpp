/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/RenderCommandEncoder.h>

#include <algorithm>
#include <array>
#include <igl/IGLSafeC.h>
#include <igl/RenderPass.h>
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Framebuffer.h>
#include <igl/vulkan/RenderPipelineState.h>
#include <igl/vulkan/SamplerState.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/TimestampQueries.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanRenderPassBuilder.h>
#include <igl/vulkan/VulkanTexture.h>
#include <igl/vulkan/util/SpvReflection.h>

// The encoders of a reusable command buffer have no Tracy context, and record no GPU zone
#if defined(IGL_WITH_TRACY_GPU)
#define IGL_ENCODER_ZONE_GPU(name) \
  TracyVkNamedZone(tracyCtx_, ___tracy_gpu_zone, cmdBuffer_, name, tracyCtx_ != nullptr)
#define IGL_ENCODER_ZONE_GPU_COLOR(name, color) \
  TracyVkNamedZoneC(tracyCtx_, ___tracy_gpu_zone, cmdBuffer_, name, color, tracyCtx_ != nullptr)
#else
#define IGL_ENCODER_ZONE_GPU(name)
#define IGL_ENCODER_ZONE_GPU_COLOR(name, color)
#endif // IGL_WITH_TRACY_GPU

namespace {

VkAttachmentLoadOp loadActionToVkAttachmentLoadOp(igl::LoadAction a) {
  using igl::LoadAction;
  switch (a) {
  case LoadAction::DontCare:
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  case LoadAction::Load:
    return VK_ATTACHMENT_LOAD_OP_LOAD;
  case LoadAction::Clear:
    return VK_ATTACHMENT_LOAD_OP_CLEAR;
  }
  IGL_DEBUG_ASSERT_NOT_REACHED();
  return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
}

VkAttachmentStoreOp storeActionToVkAttachmentStoreOp(igl::StoreAction a) {
  using igl::StoreAction;
  switch (a) {
  case StoreAction::DontCare:
    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
  case StoreAction::Store:
    return VK_ATTACHMENT_STORE_OP_STORE;
  case StoreAction::MsaaResolve:
    // for MSAA resolve, we have to store data into a special "resolve" attachment
    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
  }
  IGL_DEBUG_ASSERT_NOT_REACHED();
  return VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

// Sets the layout in which the render pass leaves `texture`, after the previous layout is journaled
// for a reusable command buffer that records into `cmdBuf`
void setLayoutAfterRenderPass(const igl::vulkan::VulkanContext& ctx,
                              VkCommandBuffer cmdBuf,
                              igl::ITexture* IGL_NULLABLE texture,
                              VkImageLayout layout) {
  if (!texture) {
    return;
  }
  const igl::vulkan::VulkanImage& image =
      static_cast<igl::vulkan::Texture&>(*texture).getVulkanTexture().image;
  ctx.journalImageLayout(cmdBuf, image);
  image.imageLayout_ = layout;
}

} // namespace

namespace igl::vulkan {

RenderCommandEncoder::RenderCommandEncoder(const std::shared_ptr<CommandBuffer>& commandBuffer,
                                           VulkanContext& ctx) :
  IRenderCommandEncoder::IRenderCommandEncoder(commandBuffer),
  ctx_(ctx),
  cmdBuffer_(commandBuffer ? commandBuffer->getVkCommandBuffer() : VK_NULL_HANDLE),
  binder_(commandBuffer.get(), ctx, VK_PIPELINE_BIND_POINT_GRAPHICS) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(commandBuffer);
  IGL_DEBUG_ASSERT(cmdBuffer_ != VK_NULL_HANDLE);
#if defined(IGL_WITH_TRACY_GPU)
  // every run of a reusable command buffer would write the timestamps of its GPU zones again into
  // queries that nothing resets
  if (commandBuffer && !commandBuffer->isReusable()) {
    tracyCtx_ = ctx.tracyCtx_;
  }
#endif
}

void RenderCommandEncoder::initialize(const RenderPassDesc& renderPass,
                                      const std::shared_ptr<IFramebuffer>& framebuffer,
                                      const Dependencies& dependencies,
                                      Result& outResult) {
  IGL_PROFILER_FUNCTION();

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx_);

  if (!IGL_DEBUG_VERIFY(cmdBuffer_)) {
    Result::setResult(&outResult, Result::Code::ArgumentNull);
    return;
  }

  processDependencies(dependencies);

  framebuffer_ = framebuffer;
  dependencies_ = dependencies;

  Result::setOk(&outResult);

  if (!IGL_DEBUG_VERIFY(framebuffer)) {
    Result::setResult(&outResult, Result::Code::ArgumentNull);
    return;
  }

  const FramebufferDesc& desc = static_cast<const Framebuffer&>((*framebuffer)).getDesc();

  std::array<VkClearValue, 2 * IGL_COLOR_ATTACHMENTS_MAX + 2> clearValues{};
  uint32_t numClearValues = 0;
  uint32_t mipLevel = 0;
  uint32_t layer = 0;

  VulkanRenderPassBuilder builder;

  const bool isStereo = desc.mode == FramebufferMode::Stereo;
  if (desc.mode != FramebufferMode::Mono) {
    if (isStereo) {
      builder.setMultiviewMasks(0x00000003, 0x00000003);
    } else {
      IGL_DEBUG_ABORT("FramebufferMode::Multiview is not implemented.");
    }
  }

  for (size_t i = 0; i != IGL_COLOR_ATTACHMENTS_MAX; i++) {
    const auto& attachment = desc.colorAttachments[i];
    if (!attachment.texture) {
      continue;
    }

    const auto& colorTexture = static_cast<Texture&>(*attachment.texture);
    numColorAttachments_++;

    // Specifically using renderPass.colorAttachments.size() in case we somehow
    // get into this loop even when renderPass.colorAttachments.empty() == true
    if (i >= renderPass.colorAttachments.size()) {
      Result::setResult(
          &outResult,
          Result::Code::ArgumentInvalid,
          "Framebuffer color attachment count larger than renderPass color attachment count");
      IGL_DEBUG_ABORT(outResult.message.c_str());
      return;
    }

    const auto& descColor = renderPass.colorAttachments[i];
    clearValues[numClearValues++] = VkClearValue{.color = {.float32 = {
                                                               descColor.clearColor.r,
                                                               descColor.clearColor.g,
                                                               descColor.clearColor.b,
                                                               descColor.clearColor.a,
                                                           }}};
    const uint32_t colorLayer = getVkLayer(colorTexture.getType(), descColor.face, descColor.layer);
    if (mipLevel) {
      IGL_DEBUG_ASSERT(descColor.mipLevel == mipLevel,
                       "All color attachments should have the same mip-level");
    }
    if (layer) {
      IGL_DEBUG_ASSERT(colorLayer == layer,
                       "All color attachments should have the same face or layer");
    }
    mipLevel = descColor.mipLevel;
    layer = colorLayer;
    const auto initialLayout = descColor.loadAction == igl::LoadAction::Load
                                   ? colorTexture.getVulkanTexture().image.imageLayout_
                                   : VK_IMAGE_LAYOUT_UNDEFINED;
    builder.addColor(textureFormatToVkFormat(colorTexture.getFormat()),
                     loadActionToVkAttachmentLoadOp(descColor.loadAction),
                     storeActionToVkAttachmentStoreOp(descColor.storeAction),
                     initialLayout,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     colorTexture.getVulkanTexture().image.samples_);
    // handle MSAA
    if (descColor.storeAction == StoreAction::MsaaResolve) {
      IGL_DEBUG_ASSERT(attachment.resolveTexture,
                       "Framebuffer attachment should contain a resolve texture");
      const auto& colorResolveTexture = static_cast<Texture&>(*attachment.resolveTexture);
      builder.addColorResolve(textureFormatToVkFormat(colorResolveTexture.getFormat()),
                              VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                              VK_ATTACHMENT_STORE_OP_STORE);
      clearValues[numClearValues++] = VkClearValue{.color = {.float32 = {
                                                                 descColor.clearColor.r,
                                                                 descColor.clearColor.g,
                                                                 descColor.clearColor.b,
                                                                 descColor.clearColor.a,
                                                             }}};
    }
  }

  // Process depth attachment
  const RenderPassDesc::AttachmentDesc descDepth = renderPass.depthAttachment;
  const RenderPassDesc::AttachmentDesc descStencil = renderPass.stencilAttachment;
  hasDepthAttachment_ = false;

  if (framebuffer->getDepthAttachment()) {
    const auto& depthTexture = static_cast<Texture&>(*(framebuffer->getDepthAttachment()));
    hasDepthAttachment_ = true;
    IGL_DEBUG_ASSERT(descDepth.mipLevel == mipLevel,
                     "Depth attachment should have the same mip-level as color attachments");
    IGL_DEBUG_ASSERT(getVkLayer(depthTexture.getType(), descDepth.face, descDepth.layer) == layer,
                     "Depth attachment should have the same face or layer as color attachments");
    clearValues[numClearValues++] = VkClearValue{.depthStencil = {
                                                     .depth = descDepth.clearDepth,
                                                     .stencil = descStencil.clearStencil,
                                                 }};
    const auto initialLayout = descDepth.loadAction == igl::LoadAction::Load
                                   ? depthTexture.getVulkanTexture().image.imageLayout_
                                   : VK_IMAGE_LAYOUT_UNDEFINED;
    builder.addDepthStencil(depthTexture.getVkFormat(),
                            loadActionToVkAttachmentLoadOp(descDepth.loadAction),
                            storeActionToVkAttachmentStoreOp(descDepth.storeAction),
                            loadActionToVkAttachmentLoadOp(descStencil.loadAction),
                            storeActionToVkAttachmentStoreOp(descStencil.storeAction),
                            initialLayout,
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            depthTexture.getVulkanTexture().image.samples_);

    // handle MSAA
    if (renderPass.depthAttachment.storeAction == StoreAction::MsaaResolve) {
      IGL_DEBUG_ASSERT(framebuffer->getResolveDepthAttachment(),
                       "Framebuffer attachment should contain a resolve depth texture");
      const auto& depthResolveTexture =
          static_cast<Texture&>(*framebuffer->getResolveDepthAttachment());
      builder.addDepthStencilResolve(depthResolveTexture.getVkFormat(),
                                     VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                     VK_ATTACHMENT_STORE_OP_STORE,
                                     VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                     VK_ATTACHMENT_STORE_OP_STORE,
                                     initialLayout,
                                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
      clearValues[numClearValues++] = VkClearValue{.depthStencil = {
                                                       .depth = descDepth.clearDepth,
                                                       .stencil = descStencil.clearStencil,
                                                   }};
    }
  }

  const auto& fb = static_cast<Framebuffer&>(*framebuffer);

  // CommandBuffer has validated the subpasses
  builder.setSubpasses(renderPass.subpasses.data(),
                       static_cast<uint32_t>(renderPass.subpasses.size()));

  const auto renderPassHandle = ctx_.findRenderPass(builder);
  if (renderPassHandle.pass == VK_NULL_HANDLE) {
    Result::setResult(
        &outResult, Result::Code::RuntimeError, "Failed to create a Vulkan render pass");
    return;
  }

  numSubpasses_ = renderPassHandle.numSubpasses;
  if (numSubpasses_ > 1) {
    std::copy_n(renderPass.subpasses.begin(), numSubpasses_, subpasses_.begin());
  }
  framebufferMipLevel_ = mipLevel;
  framebufferLayer_ = layer;
  framebufferMode_ = desc.mode;
  if (isStereo && numSubpasses_ > 1) {
    replay_ = std::make_unique<ReplayState>();
    replay_->pushConstants.resize(ctx_.getVkPhysicalDeviceProperties().limits.maxPushConstantsSize);
  }

  dynamicState_.renderPassIndex = renderPassHandle.index;
  dynamicState_.subpassIndex = 0;
  dynamicState_.multiSubpass = numSubpasses_ > 1;
  dynamicState_.renderPassCompatIndex = renderPassHandle.compatIndex;
  dynamicState_.depthBiasEnable = false;

  // The framebuffers of the render passes with subpasses are not shared with the ones of the render
  // passes without subpasses, nor with the ones of the render passes with other subpasses
  const uint32_t framebufferCompatKey = numSubpasses_ > 1 ? renderPassHandle.compatIndex + 1u : 0u;
  const VkRenderPassBeginInfo bi = fb.getRenderPassBeginInfo(renderPassHandle.pass,
                                                             mipLevel,
                                                             layer,
                                                             numClearValues,
                                                             clearValues.data(),
                                                             framebufferCompatKey);

  const uint32_t width = std::max(fb.getWidth() >> mipLevel, 1u);
  const uint32_t height = std::max(fb.getHeight() >> mipLevel, 1u);
  const Viewport viewport = {.x = 0.0f,
                             .y = 0.0f,
                             .width = static_cast<float>(width),
                             .height = static_cast<float>(height),
                             .minDepth = 0.0f,
                             .maxDepth = +1.0f};
  const ScissorRect scissor = {.x = 0, .y = 0, .width = width, .height = height};

  const VkResult vkResult = ctx_.checkAndUpdateDescriptorSets();
  if (vkResult != VK_SUCCESS) {
    IGL_LOG_ERROR("checkAndUpdateDescriptorSets returned a non-successful result: %d", vkResult);
    Result::setResult(&outResult, Result::Code::RuntimeError, "Failed to update descriptor sets");
    return;
  }

  if (renderPass.timestampQuery.queries) {
    const uint32_t slot =
        static_cast<TimestampQueries&>(*renderPass.timestampQuery.queries)
            .beginElapsedQuery(cmdBuffer_, renderPass.timestampQuery.slotIndex, nullptr);
    if (slot != TimestampQueries::kInvalidSlot) {
      timestampQueries_ = renderPass.timestampQuery.queries;
      timestampQuerySlotIndex_ = slot;
    }
  }

  if (numSubpasses_ > 1) {
    bindSubpassInputAttachments();
  }

  // A reusable command buffer that runs from another command buffer records the commands of the
  // render pass into a secondary command buffer
  cmdBuffer_ = static_cast<CommandBuffer&>(getCommandBuffer()).beginRenderPass(bi);
  if (cmdBuffer_ == VK_NULL_HANDLE) {
    Result::setResult(&outResult,
                      Result::Code::RuntimeError,
                      "Failed to record the render pass into a secondary command buffer");
    return;
  }
  binder_.cmdBuffer_ = cmdBuffer_;

  isEncoding_ = true;

  // A multiview render pass leaves the dynamic state undefined when it begins
  bindViewport(viewport);
  bindScissorRect(scissor);

  Result::setOk(&outResult);
}

std::unique_ptr<RenderCommandEncoder> RenderCommandEncoder::create(
    const std::shared_ptr<CommandBuffer>& commandBuffer,
    VulkanContext& ctx,
    const RenderPassDesc& renderPass,
    const std::shared_ptr<IFramebuffer>& framebuffer,
    const Dependencies& dependencies,
    Result* outResult) {
  IGL_PROFILER_FUNCTION();

  Result ret;

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  std::unique_ptr<RenderCommandEncoder> encoder(new RenderCommandEncoder(commandBuffer, ctx));
  encoder->initialize(renderPass, framebuffer, dependencies, ret);

  Result::setResult(outResult, ret);
  return ret.isOk() ? std::move(encoder) : nullptr;
}

void RenderCommandEncoder::endEncoding() {
  IGL_PROFILER_FUNCTION();

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx_);

  if (!isEncoding_) {
    return;
  }

  isEncoding_ = false;

  IGL_DEBUG_ASSERT(currentSubpass_ + 1 == numSubpasses_,
                   "nextSubpass() must be called RenderPassDesc::subpasses.size() - 1 times before "
                   "endEncoding()");
  // VUID-vkCmdEndRenderPass-None-00910: the last subpass must be the current one
  while (currentSubpass_ + 1 < numSubpasses_) {
    ctx_.vf_.vkCmdNextSubpass(cmdBuffer_, VK_SUBPASS_CONTENTS_INLINE);
    ++currentSubpass_;
  }

  cmdBuffer_ = static_cast<CommandBuffer&>(getCommandBuffer()).endRenderPass();
  if (cmdBuffer_ == VK_NULL_HANDLE) {
    // the reusable command buffer does not run the slot whose recording failed
    dependencies_ = {};
    return;
  }

  for (ITexture* IGL_NULLABLE tex : dependencies_.textures) {
    // TODO: at some point we might want to know in which layout a dependent texture wants to be. We
    // can implement that by adding a notion of image layouts to IGL.
    if (!tex) {
      continue;
    }

    // Retrieve the VulkanImage to check its usage
    const auto& vkTex = static_cast<Texture&>(*tex);
    const VulkanImage& img = vkTex.getVulkanTexture().image;

    if (tex->getProperties().isDepthOrStencil()) {
      // If the texture has not been marked as a depth/stencil attachment
      // (TextureDesc::TextureUsageBits::Attachment), don't transition it to a depth/stencil
      // attchment
      if ((img.usageFlags_ & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
        transitionToDepthStencilAttachment(cmdBuffer_, tex);
      }
    } else {
      // If the texture has not been marked as a color attachment
      // (TextureDesc::TextureUsageBits::Attachment), don't transition it to a color attchment
      if ((img.usageFlags_ & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0) {
        transitionToColorAttachment(cmdBuffer_, tex);
      }
    }
  }
  dependencies_ = {};

  // set image layouts after the render pass
  const FramebufferDesc& desc = static_cast<const Framebuffer&>((*framebuffer_)).getDesc();

  for (const auto& attachment : desc.colorAttachments) {
    // the image layouts of color attachments must match the final layout of the render pass, which
    // is always VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL (check VulkanRenderPassBuilder.cpp)
    setLayoutAfterRenderPass(
        ctx_, cmdBuffer_, attachment.texture.get(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    setLayoutAfterRenderPass(ctx_,
                             cmdBuffer_,
                             attachment.resolveTexture.get(),
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transitionToShaderReadOnly(cmdBuffer_, attachment.texture.get());
    transitionToShaderReadOnly(cmdBuffer_, attachment.resolveTexture.get());
  }

  // this must match the final layout of the render pass, which is always
  // VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL (check VulkanRenderPassBuilder.cpp)
  setLayoutAfterRenderPass(ctx_,
                           cmdBuffer_,
                           desc.depthAttachment.texture.get(),
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  setLayoutAfterRenderPass(ctx_,
                           cmdBuffer_,
                           desc.depthAttachment.resolveTexture.get(),
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  transitionToShaderReadOnly(cmdBuffer_, desc.depthAttachment.texture.get());
  transitionToShaderReadOnly(cmdBuffer_, desc.depthAttachment.resolveTexture.get());

  if (timestampQueries_) {
    static_cast<TimestampQueries&>(*timestampQueries_)
        .endElapsedQuery(cmdBuffer_, timestampQuerySlotIndex_);
    timestampQueries_ = nullptr;
  }

#if defined(IGL_WITH_TRACY_GPU)
  if (tracyCtx_) {
    TracyVkCollect(tracyCtx_, cmdBuffer_);
  }
#endif
}

void RenderCommandEncoder::nextSubpass() {
  IGL_PROFILER_FUNCTION();

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx_);

  if (!IGL_DEBUG_VERIFY(isEncoding_)) {
    return;
  }
  if (currentSubpass_ + 1 >= numSubpasses_) {
    IGL_DEBUG_ABORT(
        "nextSubpass() was called more than RenderPassDesc::subpasses.size() - 1 times");
    return;
  }

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p vkCmdNextSubpass(%u)\n", cmdBuffer_, currentSubpass_ + 1);
#endif // IGL_VULKAN_PRINT_COMMANDS
  ctx_.vf_.vkCmdNextSubpass(cmdBuffer_, VK_SUBPASS_CONTENTS_INLINE);

  ++currentSubpass_;
  dynamicState_.subpassIndex = currentSubpass_ & 0x3u;
  if (replay_) {
    invalidateStateForSubpass();
  } else {
    // the next draw call gets the pipeline for the new subpass from the render pipeline state, and
    // binds it. The descriptor sets that are bound stay valid, because the shaders do not change.
    binder_.invalidateBoundPipeline();
  }
  bindSubpassInputAttachments();
}

void RenderCommandEncoder::invalidateStateForSubpass() {
  IGL_PROFILER_FUNCTION();

  const uint32_t dirtyFlags = binder_.isDirtyFlags_;
  binder_.invalidateAll();

  // A bind group keeps providing its descriptor set, unless the binder wrote the descriptor set
  // of its type after the bind group was bound. Then the binder is the one that provides it.
  if (replay_->boundTextureGroup.dset != VK_NULL_HANDLE) {
    if ((dirtyFlags & ResourcesBinder::DirtyFlagBits_Textures) != 0) {
      replay_->boundTextureGroup = {};
    } else {
      binder_.isDirtyFlags_ &= ~ResourcesBinder::DirtyFlagBits_Textures;
    }
  }
  if (replay_->boundBufferGroup.dset != VK_NULL_HANDLE) {
    if ((dirtyFlags & ResourcesBinder::DirtyFlagBits_Buffers) != 0) {
      replay_->boundBufferGroup = {};
    } else {
      binder_.isDirtyFlags_ &= ~ResourcesBinder::DirtyFlagBits_Buffers;
    }
  }

  replay_->replayedPushConstantStages = 0;
  isReplayPending_ = true;
}

void RenderCommandEncoder::replayState() {
  IGL_PROFILER_FUNCTION();

  const ReplayState& state = *replay_;

  ctx_.vf_.vkCmdSetViewport(cmdBuffer_, 0, 1, &state.viewport);
  ctx_.vf_.vkCmdSetScissor(cmdBuffer_, 0, 1, &state.scissor);
  if (state.hasDepthBias) {
    ctx_.vf_.vkCmdSetDepthBias(
        cmdBuffer_, state.depthBias, state.depthBiasClamp, state.depthBiasSlopeFactor);
  }
  if (state.hasBlendConstants) {
    ctx_.vf_.vkCmdSetBlendConstants(cmdBuffer_, state.blendConstants.data());
  }
  if (state.hasStencilMasks) {
    constexpr std::array<VkStencilFaceFlagBits, 2> kFaces = {VK_STENCIL_FACE_FRONT_BIT,
                                                             VK_STENCIL_FACE_BACK_BIT};
    for (size_t i = 0; i != kFaces.size(); ++i) {
      ctx_.vf_.vkCmdSetStencilCompareMask(cmdBuffer_, kFaces[i], state.stencilCompareMasks[i]);
      ctx_.vf_.vkCmdSetStencilWriteMask(cmdBuffer_, kFaces[i], state.stencilWriteMasks[i]);
    }
  }
  if (state.hasStencilReference) {
    ctx_.vf_.vkCmdSetStencilReference(
        cmdBuffer_, VK_STENCIL_FACE_FRONT_BIT | VK_STENCIL_FACE_BACK_BIT, state.stencilReference);
  }
  if (state.hasCullMode) {
    ctx_.vf_.vkCmdSetCullMode(cmdBuffer_, state.cullMode);
  }
  if (state.hasFrontFace) {
    ctx_.vf_.vkCmdSetFrontFace(cmdBuffer_, state.frontFace);
  }

  replayPushConstants();

  for (uint32_t i = 0; i != IGL_BUFFER_BINDINGS_MAX; ++i) {
    const ReplayState::VertexBuffer& vertexBuffer = state.vertexBuffers[i];
    if (vertexBuffer.buffer == VK_NULL_HANDLE) {
      continue;
    }
    if (vertexBuffer.stride == 0) {
      ctx_.vf_.vkCmdBindVertexBuffers(cmdBuffer_, i, 1, &vertexBuffer.buffer, &vertexBuffer.offset);
    } else {
      ctx_.vf_.vkCmdBindVertexBuffers2(cmdBuffer_,
                                       i,
                                       1,
                                       &vertexBuffer.buffer,
                                       &vertexBuffer.offset,
                                       &vertexBuffer.size,
                                       &vertexBuffer.stride);
    }
  }
  if (state.indexBuffer != VK_NULL_HANDLE) {
    ctx_.vf_.vkCmdBindIndexBuffer(
        cmdBuffer_, state.indexBuffer, state.indexBufferOffset, state.indexType);
  }
}

void RenderCommandEncoder::replayPushConstants() {
  IGL_PROFILER_FUNCTION();

  ReplayState& state = *replay_;
  const VkPushConstantRange& range = rps_->pushConstantRange;
  if (range.stageFlags == 0 || range.stageFlags == state.replayedPushConstantStages) {
    return;
  }
  state.replayedPushConstantStages = range.stageFlags;

  // The pipeline is bound at this point, so its layout exists
  const uint32_t pushBegin = std::max(state.pushConstantsBegin, range.offset);
  const uint32_t pushEnd = std::min(state.pushConstantsEnd, range.offset + range.size);
  if (pushBegin < pushEnd) {
    ctx_.vf_.vkCmdPushConstants(cmdBuffer_,
                                rps_->getVkPipelineLayout(),
                                range.stageFlags,
                                pushBegin,
                                pushEnd - pushBegin,
                                state.pushConstants.data() + pushBegin);
  }
}

void RenderCommandEncoder::replayBindGroups() {
  IGL_PROFILER_FUNCTION();

  const util::SpvModuleInfo& info = rps_->getSpvModuleInfo();
  const VkPipelineLayout layout = rps_->getVkPipelineLayout();

  const ReplayState::BoundBindGroup& textureGroup = replay_->boundTextureGroup;
  if (pendingBindGroupTexture_.empty() && textureGroup.dset != VK_NULL_HANDLE &&
      (binder_.isDirtyFlags_ & ResourcesBinder::DirtyFlagBits_Textures) == 0) {
    if (textureGroup.usageMask == info.usageMaskTextures) {
      ctx_.vf_.vkCmdBindDescriptorSets(cmdBuffer_,
                                       VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       layout,
                                       kBindPoint_CombinedImageSamplers,
                                       1,
                                       &textureGroup.dset,
                                       0,
                                       nullptr);
    } else {
      binder_.isDirtyFlags_ |= ResourcesBinder::DirtyFlagBits_Textures;
    }
  }

  const ReplayState::BoundBindGroup& bufferGroup = replay_->boundBufferGroup;
  if (pendingBindGroupBuffer_.empty() && bufferGroup.dset != VK_NULL_HANDLE &&
      (binder_.isDirtyFlags_ & ResourcesBinder::DirtyFlagBits_Buffers) == 0) {
    if (bufferGroup.usageMask == info.usageMaskBuffers) {
      ctx_.vf_.vkCmdBindDescriptorSets(cmdBuffer_,
                                       VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       layout,
                                       kBindPoint_Buffers,
                                       1,
                                       &bufferGroup.dset,
                                       replay_->numBoundDynamicOffsets,
                                       replay_->boundDynamicOffsets.data());
    } else {
      binder_.isDirtyFlags_ |= ResourcesBinder::DirtyFlagBits_Buffers;
    }
  }
}

void RenderCommandEncoder::bindSubpassInputAttachments() {
  IGL_PROFILER_FUNCTION();

  const uint32_t inputMask = subpasses_[currentSubpass_].inputAttachmentMask;
  for (uint32_t i = 0; i != numColorAttachments_; ++i) {
    Texture* texture = ((inputMask >> i) & 1u) != 0
                           ? static_cast<Texture*>(framebuffer_->getColorAttachment(i).get())
                           : nullptr;
    binder_.bindInputAttachment(
        i, texture, framebufferMipLevel_, framebufferLayer_, framebufferMode_);
  }
}

void RenderCommandEncoder::pushDebugGroupLabel(const char* label, const Color& color) const {
  IGL_DEBUG_ASSERT(label && *label);
  ivkCmdBeginDebugUtilsLabel(&ctx_.vf_, cmdBuffer_, label, color.toFloatPtr());
}

void RenderCommandEncoder::insertDebugEventLabel(const char* label, const Color& color) const {
  IGL_DEBUG_ASSERT(label && *label);
  ivkCmdInsertDebugUtilsLabel(&ctx_.vf_, cmdBuffer_, label, color.toFloatPtr());
}

void RenderCommandEncoder::popDebugGroupLabel() const {
  ivkCmdEndDebugUtilsLabel(&ctx_.vf_, cmdBuffer_);
}

void RenderCommandEncoder::bindViewport(const Viewport& viewport) {
  IGL_PROFILER_FUNCTION();
  IGL_ENCODER_ZONE_GPU("bindViewport()");

  /**
    Using the negative viewport height Vulkan feature, we make the Vulkan "top-left" coordinate
  system to be "bottom-left" as in OpenGL. This way VK_FRONT_FACE_COUNTER_CLOCKWISE and
  VK_FRONT_FACE_CLOCKWISE use the same winding as in OpenGL. Part of VK_KHR_maintenance1 which is
  promoted to Vulkan 1.1.

  More details: https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
  **/
  const VkViewport vp = {
      .x = viewport.x,
      .y = viewport.height - viewport.y,
      .width = viewport.width,
      .height = -viewport.height,
      .minDepth = viewport.minDepth,
      .maxDepth = viewport.maxDepth,
  };
  ctx_.vf_.vkCmdSetViewport(cmdBuffer_, 0, 1, &vp);
  if (replay_) {
    replay_->viewport = vp;
  }
}

void RenderCommandEncoder::bindScissorRect(const ScissorRect& rect) {
  const VkRect2D scissor = {
      .offset = {.x = static_cast<int32_t>(rect.x), .y = static_cast<int32_t>(rect.y)},
      .extent = {.width = rect.width, .height = rect.height},
  };
  ctx_.vf_.vkCmdSetScissor(cmdBuffer_, 0, 1, &scissor);
  if (replay_) {
    replay_->scissor = scissor;
  }
}

void RenderCommandEncoder::bindRenderPipelineState(
    const std::shared_ptr<IRenderPipelineState>& pipelineState) {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(pipelineState)) {
    return;
  }

  rps_ = static_cast<RenderPipelineState*>(pipelineState.get());

  IGL_DEBUG_ASSERT(rps_);

  const RenderPipelineDesc& desc = rps_->getRenderPipelineDesc();

  if (desc.shaderStages->getType() == igl::ShaderStagesType::Render) {
    ensureShaderModule(desc.shaderStages->getVertexModule().get());
  } else if (desc.shaderStages->getType() == igl::ShaderStagesType::RenderMeshShader) {
    ensureShaderModule(desc.shaderStages->getMeshModule().get());
  }
  ensureShaderModule(desc.shaderStages->getFragmentModule().get());

  const bool hasDepthAttachment = desc.targetDesc.depthAttachmentFormat != TextureFormat::Invalid;

  if (IGL_DEBUG_VERIFY_NOT(hasDepthAttachment != hasDepthAttachment_)) {
    IGL_LOG_ERROR(
        "Make sure your render pass and render pipeline both have matching depth attachments");
  }

  binder_.bindPipeline(VK_NULL_HANDLE, nullptr);

  applyPipelineRasterizationDynamicState();
}

void RenderCommandEncoder::applyPipelineRasterizationDynamicState() {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(rps_)) {
    return;
  }

  const bool isVulkan13 = ctx_.getVkPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_3;

  if ((isVulkan13 || ctx_.features().has_VK_EXT_extended_dynamic_state) &&
      ctx_.vf_.vkCmdSetCullMode) {
    const RenderPipelineDesc& desc = rps_->getRenderPipelineDesc();
    const VkCullModeFlags cullMode = cullModeToVkCullMode(desc.cullMode);
    const VkFrontFace frontFace = windingModeToVkFrontFace(desc.frontFaceWinding);
    ctx_.vf_.vkCmdSetCullMode(cmdBuffer_, cullMode);
    ctx_.vf_.vkCmdSetFrontFace(cmdBuffer_, frontFace);
    if (replay_) {
      replay_->hasCullMode = true;
      replay_->cullMode = cullMode;
      replay_->hasFrontFace = true;
      replay_->frontFace = frontFace;
    }
  }
}

void RenderCommandEncoder::bindDepthStencilState(
    const std::shared_ptr<IDepthStencilState>& depthStencilState) {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(depthStencilState)) {
    return;
  }
  const DepthStencilState* state = static_cast<DepthStencilState*>(depthStencilState.get());

  const DepthStencilStateDesc& desc = state->desc;

  dynamicState_.depthWriteEnable = desc.isDepthWriteEnabled;
  dynamicState_.setDepthCompareOp(compareFunctionToVkCompareOp(desc.compareFunction));
  dynamicState_.stencilTestEnable = false;

  auto setStencilState = [this](VkStencilFaceFlagBits faceMask, const StencilStateDesc& desc) {
    const VkStencilOp failOp = stencilOperationToVkStencilOp(desc.stencilFailureOperation);
    const VkStencilOp passOp = stencilOperationToVkStencilOp(desc.depthStencilPassOperation);
    const VkStencilOp depthFailOp = stencilOperationToVkStencilOp(desc.depthFailureOperation);
    const VkCompareOp compareOp = compareFunctionToVkCompareOp(desc.stencilCompareFunction);
    dynamicState_.setStencilStateOps(
        faceMask == VK_STENCIL_FACE_FRONT_BIT, failOp, passOp, depthFailOp, compareOp);
    dynamicState_.stencilTestEnable =
        dynamicState_.stencilTestEnable || failOp != VK_STENCIL_OP_KEEP ||
        passOp != VK_STENCIL_OP_KEEP || depthFailOp != VK_STENCIL_OP_KEEP ||
        compareOp != VK_COMPARE_OP_ALWAYS;
    ctx_.vf_.vkCmdSetStencilCompareMask(cmdBuffer_, faceMask, desc.readMask);
    ctx_.vf_.vkCmdSetStencilWriteMask(cmdBuffer_, faceMask, desc.writeMask);
    if (replay_) {
      const size_t face = faceMask == VK_STENCIL_FACE_FRONT_BIT ? 0 : 1;
      replay_->hasStencilMasks = true;
      replay_->stencilCompareMasks[face] = desc.readMask;
      replay_->stencilWriteMasks[face] = desc.writeMask;
    }
  };

  setStencilState(VK_STENCIL_FACE_FRONT_BIT, desc.frontFaceStencil);
  setStencilState(VK_STENCIL_FACE_BACK_BIT, desc.backFaceStencil);
}

void RenderCommandEncoder::bindBuffer(uint32_t index,
                                      uint8_t target,
                                      IBuffer* buffer,
                                      size_t bufferOffset,
                                      size_t bufferSize) {
  (void)target;
  bindBuffer(index, buffer, bufferOffset, bufferSize);
}

void RenderCommandEncoder::bindBuffer(uint32_t index,
                                      IBuffer* buffer,
                                      size_t bufferOffset,
                                      size_t bufferSize) {
  IGL_PROFILER_FUNCTION();
  IGL_ENCODER_ZONE_GPU("bindBuffer()");

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p  bindBuffer(%u, %u)\n", cmdBuffer_, index, static_cast<uint32_t>(bufferOffset));
#endif // IGL_VULKAN_PRINT_COMMANDS

  if (!IGL_DEBUG_VERIFY(buffer)) {
    return;
  }

  auto* buf = static_cast<Buffer*>(buffer);

  const bool isUniformBuffer = (buf->getBufferType() & BufferDesc::BufferTypeBits::Uniform) > 0;
  const bool isStorageBuffer = (buf->getBufferType() & BufferDesc::BufferTypeBits::Storage) > 0;
  const bool isUniformOrStorageBuffer = isUniformBuffer || isStorageBuffer;

  if (!IGL_DEBUG_VERIFY(isUniformOrStorageBuffer, "Must be a uniform or a storage buffer")) {
    return;
  }
  binder_.bindBuffer(index, buf, bufferOffset, bufferSize);
}

void RenderCommandEncoder::bindVertexBuffer(uint32_t index,
                                            IBuffer& buffer,
                                            size_t bufferOffset,
                                            size_t attributeStride) {
  IGL_PROFILER_FUNCTION();
  IGL_ENCODER_ZONE_GPU("bindVertexBuffer()");

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p  bindVertexBuffer(%u, %p, %u)\n",
               cmdBuffer_,
               index,
               &buffer,
               static_cast<uint32_t>(bufferOffset));
#endif // IGL_VULKAN_PRINT_COMMANDS

  const bool isVertexBuffer = (buffer.getBufferType() & BufferDesc::BufferTypeBits::Vertex) != 0;

  if (!IGL_DEBUG_VERIFY(isVertexBuffer)) {
    return;
  }

  if (IGL_DEBUG_VERIFY(index < IGL_ARRAY_NUM_ELEMENTS(isVertexBufferBound_))) {
    isVertexBufferBound_[index] = true;
  }
  const VkBuffer vkBuf = static_cast<Buffer&>(buffer).getVkBuffer();
  const VkDeviceSize offset = bufferOffset;

  if (attributeStride == 0) {
    ctx_.vf_.vkCmdBindVertexBuffers(cmdBuffer_, index, 1, &vkBuf, &offset);
    if (replay_ && index < replay_->vertexBuffers.size()) {
      replay_->vertexBuffers[index] = {.buffer = vkBuf, .offset = offset};
    }
  } else {
    if (!ctx_.vf_.vkCmdBindVertexBuffers2) {
      IGL_LOG_ERROR_ONCE(
          "RenderCommandEncoder::bindVertexBuffer(attributeStride): vkCmdBindVertexBuffers2(EXT) "
          "not available; "
          "attributeStride ignored. Enable VK_EXT_extended_dynamic_state or Vulkan 1.3.");
      return;
    }

    IGL_DEBUG_ASSERT(ctx_.config_.enableDynamicVertexBufferStride,
                     "Must enable DynamicVertexBufferStride first!");

    const VkDeviceSize size = buffer.getSizeInBytes() - bufferOffset;
    const VkDeviceSize stride = attributeStride;
    ctx_.vf_.vkCmdBindVertexBuffers2(cmdBuffer_, index, 1, &vkBuf, &offset, &size, &stride);
    if (replay_ && index < replay_->vertexBuffers.size()) {
      replay_->vertexBuffers[index] = {
          .buffer = vkBuf, .offset = offset, .size = size, .stride = stride};
    }
  }
}

void RenderCommandEncoder::bindIndexBuffer(IBuffer& buffer,
                                           IndexFormat format,
                                           size_t bufferOffset) {
  const auto& buf = static_cast<Buffer&>(buffer);

  IGL_DEBUG_ASSERT(buf.getBufferUsageFlags() & VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                   "Did you forget to specify BufferTypeBits::Index on your buffer?");

  auto indexFormatToVkIndexType = [](IndexFormat fmt, bool has8BitIndices) -> VkIndexType {
    switch (fmt) {
    case igl::IndexFormat::UInt8:
      return IGL_DEBUG_VERIFY(has8BitIndices) ? VK_INDEX_TYPE_UINT8_EXT : VK_INDEX_TYPE_UINT16;
    case igl::IndexFormat::UInt16:
      return VK_INDEX_TYPE_UINT16;
    case igl::IndexFormat::UInt32:
      return VK_INDEX_TYPE_UINT32;
    };
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return VK_INDEX_TYPE_UINT16;
  };

  const VkIndexType type =
      indexFormatToVkIndexType(format, ctx_.features_.has_VK_EXT_index_type_uint8);

  ctx_.vf_.vkCmdBindIndexBuffer(cmdBuffer_, buf.getVkBuffer(), bufferOffset, type);
  if (replay_) {
    replay_->indexBuffer = buf.getVkBuffer();
    replay_->indexBufferOffset = bufferOffset;
    replay_->indexType = type;
  }
}

void RenderCommandEncoder::bindBytes(size_t /*index*/,
                                     uint8_t /*target*/,
                                     const void* /*data*/,
                                     size_t /*length*/) {
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
}

void RenderCommandEncoder::bindPushConstants(const void* data, size_t length, size_t offset) {
  IGL_PROFILER_FUNCTION();
  IGL_ENCODER_ZONE_GPU("bindPushConstants()");

  IGL_DEBUG_ASSERT(length % 4 == 0); // VUID-vkCmdPushConstants-size-00369: size must be a multiple
                                     // of 4

  IGL_DEBUG_ASSERT(rps_, "Did you forget to call bindRenderPipelineState()?");
  IGL_DEBUG_ASSERT(rps_->pushConstantRange.size,
                   "Currently bound render pipeline state has no push constants");
  IGL_DEBUG_ASSERT(offset + length <= rps_->pushConstantRange.offset + rps_->pushConstantRange.size,
                   "Push constants size exceeded");

  if (!rps_->pipelineLayout) {
    // bring a pipeline layout into existence - we don't really care about the dynamic state here
    (void)rps_->getVkPipeline(dynamicState_);
  }

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p vkCmdPushConstants(%u) - GRAPHICS\n", cmdBuffer_, length);
#endif // IGL_VULKAN_PRINT_COMMANDS
  ctx_.vf_.vkCmdPushConstants(cmdBuffer_,
                              rps_->getVkPipelineLayout(),
                              rps_->pushConstantRange.stageFlags,
                              static_cast<uint32_t>(offset),
                              static_cast<uint32_t>(length),
                              data);

  if (replay_ && length != 0 &&
      IGL_DEBUG_VERIFY(offset + length <= replay_->pushConstants.size())) {
    checked_memcpy_offset(
        replay_->pushConstants.data(), replay_->pushConstants.size(), offset, data, length);
    const auto begin = static_cast<uint32_t>(offset);
    const auto end = static_cast<uint32_t>(offset + length);
    const bool isEmpty = replay_->pushConstantsBegin == replay_->pushConstantsEnd;
    replay_->pushConstantsBegin = isEmpty ? begin : std::min(replay_->pushConstantsBegin, begin);
    replay_->pushConstantsEnd = isEmpty ? end : std::max(replay_->pushConstantsEnd, end);
  }
}

void RenderCommandEncoder::bindSamplerState(size_t index,
                                            uint8_t target,
                                            ISamplerState* samplerState) {
  IGL_PROFILER_FUNCTION();

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p  bindSamplerState(%u, %u)\n",
               cmdBuffer_,
               static_cast<uint32_t>(index),
               static_cast<uint32_t>(target));
#endif // IGL_VULKAN_PRINT_COMMANDS

  if (!IGL_DEBUG_VERIFY(target == igl::BindTarget::kFragment ||
                            target == igl::BindTarget::kVertex ||
                            target == igl::BindTarget::kAllGraphics,
                        "Invalid sampler target")) {
    return;
  }

  binder_.bindSamplerState(static_cast<uint32_t>(index), static_cast<SamplerState*>(samplerState));
}

void RenderCommandEncoder::bindTexture(size_t index, uint8_t target, ITexture* texture) {
  IGL_PROFILER_FUNCTION();

  IGL_DEBUG_ASSERT(pendingBindGroupTexture_.empty(),
                   "A texture BindGroup was already bound to this command encoder. You can bind "
                   "individual textures again only after a draw call.");

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p  bindTexture(%u, %u)\n",
               cmdBuffer_,
               static_cast<uint32_t>(index),
               static_cast<uint32_t>(target));
#endif // IGL_VULKAN_PRINT_COMMANDS

  if (!IGL_DEBUG_VERIFY(target == igl::BindTarget::kFragment ||
                            target == igl::BindTarget::kVertex ||
                            target == igl::BindTarget::kAllGraphics,
                        "Invalid texture target")) {
    return;
  }

  binder_.bindTexture(static_cast<uint32_t>(index), static_cast<Texture*>(texture));
}

void RenderCommandEncoder::bindTexture(size_t index, ITexture* texture) {
  bindTexture(index, igl::BindTarget::kFragment, texture);
}

void RenderCommandEncoder::bindUniform(const UniformDesc& /*uniformDesc*/, const void* /*data*/) {
  // DO NOT IMPLEMENT!
  // This is only for backends that MUST use single uniforms in some situations.
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
}

void RenderCommandEncoder::draw(size_t vertexCount,
                                uint32_t instanceCount,
                                uint32_t firstVertex,
                                uint32_t baseInstance) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DRAW);
  IGL_ENCODER_ZONE_GPU_COLOR("draw()", IGL_PROFILER_COLOR_DRAW);

  ctx_.drawCallCount_ += drawCallCountEnabled_;

  if (vertexCount == 0) {
    // IGL/OpenGL tests rely on this behavior due to how state caching is organized over there.
    // If we do not return here, Validation Layers will complain.
    return;
  }

  IGL_DEBUG_ASSERT(rps_, "Did you forget to call bindRenderPipelineState()?");

  ensureVertexBuffers();

  if (!flushDynamicState()) {
    return;
  }

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p vkCmdDraw(%u, %u, %u, %u)\n",
               cmdBuffer_,
               static_cast<uint32_t>(vertexCount),
               instanceCount,
               firstVertex,
               baseInstance);
#endif // IGL_VULKAN_PRINT_COMMANDS

  ctx_.vf_.vkCmdDraw(
      cmdBuffer_, static_cast<uint32_t>(vertexCount), instanceCount, firstVertex, baseInstance);
}

void RenderCommandEncoder::drawIndexed(size_t indexCount,
                                       uint32_t instanceCount,
                                       uint32_t firstIndex,
                                       int32_t vertexOffset,
                                       uint32_t baseInstance) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DRAW);
  IGL_ENCODER_ZONE_GPU_COLOR("drawIndexed()", IGL_PROFILER_COLOR_DRAW);

  ctx_.drawCallCount_ += drawCallCountEnabled_;

  if (indexCount == 0) {
    // IGL/OpenGL tests rely on this behavior due to how state caching is organized over there.
    // If we do not return here, Validation Layers will complain.
    return;
  }

  IGL_DEBUG_ASSERT(rps_, "Did you forget to call bindRenderPipelineState()?");

  ensureVertexBuffers();

  if (!flushDynamicState()) {
    return;
  }

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p vkCmdDrawIndexed(%u, %u, %u, %i, %u)\n",
               cmdBuffer_,
               static_cast<uint32_t>(indexCount),
               instanceCount,
               firstIndex,
               vertexOffset,
               baseInstance);
#endif // IGL_VULKAN_PRINT_COMMANDS
  ctx_.vf_.vkCmdDrawIndexed(cmdBuffer_,
                            static_cast<uint32_t>(indexCount),
                            instanceCount,
                            firstIndex,
                            vertexOffset,
                            baseInstance);
}

void RenderCommandEncoder::drawMeshTasks(const Dimensions& threadgroupsPerGrid,
                                         const Dimensions& /*threadsPerTaskThreadgroup*/,
                                         const Dimensions& /*threadsPerMeshThreadgroup*/) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DRAW);
  IGL_ENCODER_ZONE_GPU_COLOR("drawMeshTasks()", IGL_PROFILER_COLOR_DRAW);

  if (!ctx_.features().has_VK_EXT_mesh_shader) {
    IGL_DEBUG_ASSERT(false, "Mesh shaders require VK_EXT_mesh_shader extension.");
    return;
  }

  ctx_.drawCallCount_ += drawCallCountEnabled_;

  IGL_DEBUG_ASSERT(rps_, "Did you forget to call bindRenderPipelineState()?");

  if (!flushDynamicState()) {
    return;
  }

  ctx_.vf_.vkCmdDrawMeshTasksEXT(
      cmdBuffer_, threadgroupsPerGrid.width, threadgroupsPerGrid.height, threadgroupsPerGrid.depth);
}

void RenderCommandEncoder::multiDrawIndirect(IBuffer& indirectBuffer,
                                             size_t indirectBufferOffset,
                                             uint32_t drawCount,
                                             uint32_t stride) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DRAW);
  IGL_ENCODER_ZONE_GPU_COLOR("multiDrawIndirect()", IGL_PROFILER_COLOR_DRAW);

  IGL_DEBUG_ASSERT(rps_, "Did you forget to call bindRenderPipelineState()?");

  ensureVertexBuffers();

  if (!flushDynamicState()) {
    return;
  }

  ctx_.drawCallCount_ += drawCallCountEnabled_;

  const Buffer* bufIndirect = static_cast<Buffer*>(&indirectBuffer);

  ctx_.vf_.vkCmdDrawIndirect(cmdBuffer_,
                             bufIndirect->getVkBuffer(),
                             indirectBufferOffset,
                             drawCount,
                             stride ? stride : sizeof(VkDrawIndirectCommand));
}

void RenderCommandEncoder::multiDrawIndexedIndirect(IBuffer& indirectBuffer,
                                                    size_t indirectBufferOffset,
                                                    uint32_t drawCount,
                                                    uint32_t stride) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DRAW);
  IGL_ENCODER_ZONE_GPU_COLOR("multiDrawIndexedIndirect()", IGL_PROFILER_COLOR_DRAW);

  IGL_DEBUG_ASSERT(rps_, "Did you forget to call bindRenderPipelineState()?");

  ensureVertexBuffers();

  if (!flushDynamicState()) {
    return;
  }

  ctx_.drawCallCount_ += drawCallCountEnabled_;

  const Buffer* bufIndirect = static_cast<Buffer*>(&indirectBuffer);

  ctx_.vf_.vkCmdDrawIndexedIndirect(cmdBuffer_,
                                    bufIndirect->getVkBuffer(),
                                    indirectBufferOffset,
                                    drawCount,
                                    stride ? stride : sizeof(VkDrawIndexedIndirectCommand));
}

void RenderCommandEncoder::setStencilReferenceValue(uint32_t value) {
  IGL_PROFILER_FUNCTION();

  ctx_.vf_.vkCmdSetStencilReference(
      cmdBuffer_, VK_STENCIL_FACE_FRONT_BIT | VK_STENCIL_FACE_BACK_BIT, value);
  if (replay_) {
    replay_->hasStencilReference = true;
    replay_->stencilReference = value;
  }
}

void RenderCommandEncoder::setBlendColor(const Color& color) {
  IGL_PROFILER_FUNCTION();

  ctx_.vf_.vkCmdSetBlendConstants(cmdBuffer_, color.toFloatPtr());
  if (replay_) {
    replay_->hasBlendConstants = true;
    replay_->blendConstants = {color.r, color.g, color.b, color.a};
  }
}

void RenderCommandEncoder::setCullMode(CullMode cullMode) {
  IGL_PROFILER_FUNCTION();

  if (!ctx_.vf_.vkCmdSetCullMode) {
    IGL_LOG_ERROR_ONCE(
        "RenderCommandEncoder::setCullMode: vkCmdSetCullMode(EXT) not available; "
        "cull mode change ignored. Enable VK_EXT_extended_dynamic_state or Vulkan 1.3.");
    return;
  }

  const VkCullModeFlags vkCullMode = cullModeToVkCullMode(cullMode);
  ctx_.vf_.vkCmdSetCullMode(cmdBuffer_, vkCullMode);
  if (replay_) {
    replay_->hasCullMode = true;
    replay_->cullMode = vkCullMode;
  }
}

void RenderCommandEncoder::setDepthBias(float depthBias, float slopeScale, float clamp) {
  IGL_PROFILER_FUNCTION();

  dynamicState_.depthBiasEnable = true;
  ctx_.vf_.vkCmdSetDepthBias(cmdBuffer_, depthBias, clamp, slopeScale);
  if (replay_) {
    replay_->hasDepthBias = true;
    replay_->depthBias = depthBias;
    replay_->depthBiasClamp = clamp;
    replay_->depthBiasSlopeFactor = slopeScale;
  }
}

void RenderCommandEncoder::setFrontFacingWinding(WindingMode frontFaceWinding) {
  IGL_PROFILER_FUNCTION();

  if (!ctx_.vf_.vkCmdSetFrontFace) {
    IGL_LOG_ERROR_ONCE(
        "RenderCommandEncoder::setFrontFacingWinding: vkCmdSetFrontFace(EXT) not available; "
        "front face change ignored. Enable VK_EXT_extended_dynamic_state or Vulkan 1.3.");
    return;
  }

  const VkFrontFace vkFrontFace = windingModeToVkFrontFace(frontFaceWinding);
  ctx_.vf_.vkCmdSetFrontFace(cmdBuffer_, vkFrontFace);
  if (replay_) {
    replay_->hasFrontFace = true;
    replay_->frontFace = vkFrontFace;
  }
}

bool RenderCommandEncoder::setDrawCallCountEnabled(bool value) {
  IGL_PROFILER_FUNCTION();

  const bool returnVal = drawCallCountEnabled_ > 0;
  drawCallCountEnabled_ = static_cast<uint32_t>(value);
  return returnVal;
}

void RenderCommandEncoder::flushDynamicDepthStencilState() {
  IGL_PROFILER_FUNCTION();

  const bool isVulkan13 = ctx_.getVkPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_3;

  if ((isVulkan13 || ctx_.features().has_VK_EXT_extended_dynamic_state) &&
      ctx_.vf_.vkCmdSetDepthTestEnable) {
    const bool depthTestEnable = dynamicState_.getDepthCompareOp() != VK_COMPARE_OP_ALWAYS ||
                                 dynamicState_.depthWriteEnable;
    ctx_.vf_.vkCmdSetDepthTestEnable(cmdBuffer_, depthTestEnable ? VK_TRUE : VK_FALSE);
    ctx_.vf_.vkCmdSetDepthWriteEnable(cmdBuffer_,
                                      dynamicState_.depthWriteEnable ? VK_TRUE : VK_FALSE);
    ctx_.vf_.vkCmdSetDepthCompareOp(cmdBuffer_, dynamicState_.getDepthCompareOp());

    ctx_.vf_.vkCmdSetStencilTestEnable(cmdBuffer_,
                                       dynamicState_.stencilTestEnable ? VK_TRUE : VK_FALSE);
    ctx_.vf_.vkCmdSetStencilOp(cmdBuffer_,
                               VK_STENCIL_FACE_FRONT_BIT,
                               dynamicState_.getStencilStateFailOp(true),
                               dynamicState_.getStencilStatePassOp(true),
                               dynamicState_.getStencilStateDepthFailOp(true),
                               dynamicState_.getStencilStateCompareOp(true));
    ctx_.vf_.vkCmdSetStencilOp(cmdBuffer_,
                               VK_STENCIL_FACE_BACK_BIT,
                               dynamicState_.getStencilStateFailOp(false),
                               dynamicState_.getStencilStatePassOp(false),
                               dynamicState_.getStencilStateDepthFailOp(false),
                               dynamicState_.getStencilStateCompareOp(false));
  }

  if (ctx_.vf_.vkCmdSetDepthBiasEnable) {
    ctx_.vf_.vkCmdSetDepthBiasEnable(cmdBuffer_,
                                     dynamicState_.depthBiasEnable ? VK_TRUE : VK_FALSE);
  }
}

#if IGL_DEBUG_ABORT_ENABLED
void RenderCommandEncoder::validatePipelineForSubpass() const {
  if (!rps_) {
    return;
  }

  const char* const debugName = rps_->getRenderPipelineDesc().debugName.c_str();

  const uint32_t inputMask = subpasses_[currentSubpass_].inputAttachmentMask;
  for (const util::InputAttachmentDescription& a : rps_->getSpvModuleInfo().inputAttachments) {
    IGL_DEBUG_ASSERT(a.inputAttachmentIndex < IGL_COLOR_ATTACHMENTS_MAX &&
                         ((inputMask >> a.inputAttachmentIndex) & 1u) != 0,
                     "Render pipeline state '%s' reads input attachment %u, which subpass %u does "
                     "not read (input attachment mask: 0x%x)",
                     debugName,
                     a.inputAttachmentIndex,
                     currentSubpass_,
                     inputMask);
  }

  if (numSubpasses_ > 1) {
    // every subpass has an entry for every color attachment, and so must the pipeline
    const auto& colorAttachments = rps_->getRenderPipelineDesc().targetDesc.colorAttachments;
    bool declaresExactlyTheAttachments = colorAttachments.size() >= numColorAttachments_;
    for (size_t i = 0; i != colorAttachments.size(); ++i) {
      const bool hasFormat = colorAttachments[i].textureFormat != TextureFormat::Invalid;
      if (hasFormat != (i < numColorAttachments_)) {
        declaresExactlyTheAttachments = false;
      }
    }
    IGL_DEBUG_ASSERT(declaresExactlyTheAttachments,
                     "Render pipeline state '%s' is used in a render pass with subpasses, so its "
                     "targetDesc.colorAttachments must declare a format for each of the %u color "
                     "attachments of the framebuffer, and for no other attachment",
                     debugName,
                     numColorAttachments_);
  }
}
#endif // IGL_DEBUG_ABORT_ENABLED

bool RenderCommandEncoder::flushDynamicState() {
  IGL_PROFILER_FUNCTION();

#if IGL_DEBUG_ABORT_ENABLED
  validatePipelineForSubpass();
#endif // IGL_DEBUG_ABORT_ENABLED

  VkPipeline pipeline = rps_->getVkPipeline(dynamicState_);
  if (!pipeline) {
    return false;
  }

  binder_.bindPipeline(pipeline, &rps_->getSpvModuleInfo());

  flushDynamicDepthStencilState();

  if (isReplayPending_) {
    isReplayPending_ = false;
    replayState();
    replayBindGroups();
  } else if (replay_ && currentSubpass_ != 0) {
    // A pipeline bound after the first draw call of the subpass can have a layout that is not
    // compatible for push constants with the one that the replay pushed with
    replayPushConstants();
  }

  const VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

  if (!pendingBindGroupTexture_.empty()) {
    VkDescriptorSet dset = ctx_.getBindGroupDescriptorSet(pendingBindGroupTexture_);
    VkPipelineLayout layout = rps_->getVkPipelineLayout();

    const uint32_t usageMaskPipeline = rps_->getSpvModuleInfo().usageMaskTextures;
    const uint32_t usageMaskBindGroup = ctx_.getBindGroupUsageMask(pendingBindGroupTexture_);

    if (!IGL_DEBUG_VERIFY(usageMaskPipeline == usageMaskBindGroup)) {
      IGL_LOG_ERROR(
          "Texture bind group is not compatible with the current IRenderPipelineState '%s'\n",
          rps_->getRenderPipelineDesc().debugName.c_str());
      IGL_LOG_INFO(IGL_FORMAT("Bind group textures mask: {:b}\n", usageMaskBindGroup).c_str());
      IGL_LOG_INFO(IGL_FORMAT("Pipeline expects        : {:b}\n", usageMaskPipeline).c_str());
      return false;
    }

#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkCmdBindDescriptorSets(%u) - textures bind group\n", cmdBuffer_);
#endif // IGL_VULKAN_PRINT_COMMANDS
    ctx_.vf_.vkCmdBindDescriptorSets(
        cmdBuffer_, bindPoint, layout, kBindPoint_CombinedImageSamplers, 1, &dset, 0, nullptr);
    // This is necessary to support a mix of BindGroups and bindTexture() calls in the same command
    // encoder. A typical use case for that is running ImGui rendering etc.
    binder_.isDirtyFlags_ &= ~igl::vulkan::ResourcesBinder::DirtyFlagBits_Textures;
    if (replay_) {
      replay_->boundTextureGroup = {.dset = dset, .usageMask = usageMaskBindGroup};
    }
    pendingBindGroupTexture_ = {}; // reset
  }

  if (!pendingBindGroupBuffer_.empty()) {
    VkDescriptorSet dset = ctx_.getBindGroupDescriptorSet(pendingBindGroupBuffer_);
    VkPipelineLayout layout = rps_->getVkPipelineLayout();

    const uint32_t usageMaskPipeline = rps_->getSpvModuleInfo().usageMaskBuffers;
    const uint32_t usageMaskBindGroup = ctx_.getBindGroupUsageMask(pendingBindGroupBuffer_);

    if (!IGL_DEBUG_VERIFY(usageMaskPipeline == usageMaskBindGroup)) {
      IGL_LOG_ERROR(
          "Buffer bind group is not compatible with the current IRenderPipelineState '%s'\n",
          rps_->getRenderPipelineDesc().debugName.c_str());
      IGL_LOG_INFO(IGL_FORMAT("Bind group buffers mask: {:b}\n", usageMaskBindGroup).c_str());
      IGL_LOG_INFO(IGL_FORMAT("Pipeline expects       : {:b}\n", usageMaskPipeline).c_str());
      return false;
    }

#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkCmdBindDescriptorSets(%u) - buffers bind group\n", cmdBuffer_);
#endif // IGL_VULKAN_PRINT_COMMANDS
    ctx_.vf_.vkCmdBindDescriptorSets(cmdBuffer_,
                                     bindPoint,
                                     layout,
                                     kBindPoint_Buffers,
                                     1,
                                     &dset,
                                     numDynamicOffsets_,
                                     dynamicOffsets_);
    // This is necessary to support a mix of BindGroups and bindBuffer() calls in the same command
    // encoder.
    binder_.isDirtyFlags_ &= ~igl::vulkan::ResourcesBinder::DirtyFlagBits_Buffers;
    if (replay_) {
      replay_->boundBufferGroup = {.dset = dset, .usageMask = usageMaskBindGroup};
      replay_->numBoundDynamicOffsets = numDynamicOffsets_;
      std::copy_n(dynamicOffsets_, numDynamicOffsets_, replay_->boundDynamicOffsets.begin());
    }
    pendingBindGroupBuffer_ = {}; // reset
  }

  if (replay_) {
    // the binder provides the descriptor sets that it writes now, instead of the bind groups
    if ((binder_.isDirtyFlags_ & ResourcesBinder::DirtyFlagBits_Textures) != 0) {
      replay_->boundTextureGroup = {};
    }
    if ((binder_.isDirtyFlags_ & ResourcesBinder::DirtyFlagBits_Buffers) != 0) {
      replay_->boundBufferGroup = {};
    }
  }

  binder_.updateBindings(rps_->getVkPipelineLayout(), *rps_);

  if (ctx_.config_.enableDescriptorIndexing) {
    VkDescriptorSet dset = ctx_.getBindlessVkDescriptorSet();

#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkCmdBindDescriptorSets(GRAPHICS) - bindless\n", cmdBuffer_);
#endif // IGL_VULKAN_PRINT_COMMANDS
    ctx_.vf_.vkCmdBindDescriptorSets(cmdBuffer_,
                                     VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     rps_->getVkPipelineLayout(),
                                     kBindPoint_Bindless,
                                     1,
                                     &dset,
                                     0,
                                     nullptr);
  }
  return true;
}

void RenderCommandEncoder::ensureVertexBuffers() {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(rps_)) {
    return;
  }

  const VertexInputState* vi =
      static_cast<VertexInputState*>(rps_->getRenderPipelineDesc().vertexInputState.get());

  if (!vi) {
    // no vertex input is perfectly valid
    return;
  }

  const VertexInputStateDesc& desc = vi->desc;

  IGL_DEBUG_ASSERT(desc.numInputBindings <= IGL_ARRAY_NUM_ELEMENTS(isVertexBufferBound_));

  for (size_t i = 0; i != desc.numAttributes; i++) {
    [[maybe_unused]] const size_t index = desc.attributes[i].bufferIndex;
    IGL_DEBUG_ASSERT(index < IGL_ARRAY_NUM_ELEMENTS(isVertexBufferBound_));
    if (!isVertexBufferBound_[index]) {
      // TODO: fix client apps and uncomment
      // IGL_DEBUG_ABORT(
      //                "Did you forget to call bindBuffer() for one of your vertex input
      //                buffers?");
      IGL_LOG_ERROR_ONCE(
          "Did you forget to call bindBuffer() for one of your vertex input buffers?");
    }
  }
}

void RenderCommandEncoder::blitColorImage(const VulkanImage& srcImage,
                                          const VulkanImage& destImage,
                                          const TextureRangeDesc& srcRange,
                                          const TextureRangeDesc& destRange) {
  const VkImageSubresourceRange srcResourceRange = {
      .aspectMask = srcImage.getImageAspectFlags(),
      .baseMipLevel = srcRange.mipLevel,
      .levelCount = srcRange.numMipLevels,
      .baseArrayLayer = srcRange.layer,
      .layerCount = srcRange.numLayers,
  };
  const VkImageSubresourceRange destSubresourceRange = {
      .aspectMask = destImage.getImageAspectFlags(),
      .baseMipLevel = destRange.mipLevel,
      .levelCount = destRange.numMipLevels,
      .baseArrayLayer = destRange.layer,
      .layerCount = destRange.numLayers,
  };
  srcImage.transitionLayout(cmdBuffer_,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            srcResourceRange);

  destImage.transitionLayout(cmdBuffer_,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             destSubresourceRange);

  const std::array<VkOffset3D, 2> srcOffsets = {{
      {.x = static_cast<int32_t>(srcRange.x), .y = static_cast<int32_t>(srcRange.y), .z = 0},
      {.x = static_cast<int32_t>(srcRange.width + srcRange.x),
       .y = static_cast<int32_t>(srcRange.height + srcRange.y),
       .z = 1},
  }};
  const std::array<VkOffset3D, 2> dstOffsets = {{
      {.x = static_cast<int32_t>(destRange.x), .y = static_cast<int32_t>(destRange.y), .z = 0},
      {.x = static_cast<int32_t>(destRange.width + destRange.x),
       .y = static_cast<int32_t>(destRange.height + destRange.y),
       .z = 1},
  }};
  ivkCmdBlitImage(&ctx_.vf_,
                  cmdBuffer_,
                  srcImage.getVkImage(),
                  destImage.getVkImage(),
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  srcOffsets.data(),
                  dstOffsets.data(),
                  VkImageSubresourceLayers{
                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                      .mipLevel = srcRange.mipLevel,
                      .baseArrayLayer = 0,
                      .layerCount = 1,
                  },
                  VkImageSubresourceLayers{
                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                      .mipLevel = 0,
                      .baseArrayLayer = 0,
                      .layerCount = 1,
                  },
                  VK_FILTER_LINEAR);

  const bool isSampled = (destImage.getVkImageUsageFlags() & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;
  const bool isStorage = (destImage.getVkImageUsageFlags() & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
  const bool isColorAttachment =
      (destImage.getVkImageUsageFlags() & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0;
  const bool isDepthStencilAttachment =
      (destImage.getVkImageUsageFlags() & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;

  // a ternary cascade...
  const VkImageLayout targetLayout =
      isSampled ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : (isStorage ? VK_IMAGE_LAYOUT_GENERAL
                             : (isColorAttachment
                                    ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                    : (isDepthStencilAttachment
                                           ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                           : VK_IMAGE_LAYOUT_UNDEFINED)));

  IGL_DEBUG_ASSERT(targetLayout != VK_IMAGE_LAYOUT_UNDEFINED, "Missing usage flags");

  // 3. Transition TRANSFER_DST_OPTIMAL into `targetLayout`
  destImage.transitionLayout(cmdBuffer_,
                             targetLayout,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             destSubresourceRange);

  destImage.imageLayout_ = targetLayout;
}

void RenderCommandEncoder::processDependencies(const Dependencies& dependencies) {
  // 1. Process all textures
  {
    const Dependencies* deps = &dependencies;

    while (deps) {
      for (ITexture* IGL_NULLABLE tex : deps->textures) {
        if (!tex) {
          break;
        }
        transitionToShaderReadOnly(cmdBuffer_, tex);
      }
      deps = deps->next;
    }
  }

  // 2. Process all buffers
  {
    const Dependencies* deps = &dependencies;

    while (deps) {
      for (uint32_t index = 0; index < Dependencies::kIglMaxBufferDependencies; ++index) {
        IBuffer* IGL_NULLABLE const buf = deps->buffers[index];
        if (!buf) {
          break;
        }
        VkPipelineStageFlags dstStageFlags =
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        const auto* vkBuf = static_cast<const Buffer*>(buf);
        const VkBufferUsageFlags flags = vkBuf->getBufferUsageFlags();
        if ((flags & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0 ||
            (flags & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0) {
          dstStageFlags |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        }
        if ((flags & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0) {
          dstStageFlags |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        }
        const VkPipelineStageFlags srcStageFlags =
            (deps->hostWriteBufferMask & (uint32_t{1} << index)) != 0
                ? VK_PIPELINE_STAGE_HOST_BIT
                : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        ivkBufferBarrier(&ctx_.vf_,
                         cmdBuffer_,
                         vkBuf->getVkBuffer(),
                         vkBuf->getBufferUsageFlags(),
                         srcStageFlags,
                         dstStageFlags);
      }
      deps = deps->next;
    }
  }
}

void RenderCommandEncoder::bindBindGroup(BindGroupTextureHandle handle) {
  IGL_DEBUG_ASSERT(!handle.empty());

  pendingBindGroupTexture_ = handle;
}

void RenderCommandEncoder::bindBindGroup(BindGroupBufferHandle handle,
                                         uint32_t numDynamicOffsets,
                                         const uint32_t* dynamicOffsets) {
  IGL_DEBUG_ASSERT(!handle.empty());

  pendingBindGroupBuffer_ = handle;
  numDynamicOffsets_ = numDynamicOffsets;

  if (numDynamicOffsets) {
    IGL_DEBUG_ASSERT(dynamicOffsets);
    IGL_DEBUG_ASSERT(numDynamicOffsets <= IGL_UNIFORM_BLOCKS_BINDING_MAX);

    checked_memcpy(dynamicOffsets_,
                   sizeof(dynamicOffsets_),
                   dynamicOffsets,
                   numDynamicOffsets * sizeof(uint32_t));
  }
}

} // namespace igl::vulkan
