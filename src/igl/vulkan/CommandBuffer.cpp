/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/CommandBuffer.h>

#include <algorithm>
#include <igl/Framebuffer.h>
#include <igl/RenderPass.h>
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/ComputeCommandEncoder.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/RenderCommandEncoder.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/Timer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanTexture.h>

namespace igl::vulkan {

namespace {

constexpr const char* kRecordingFailedMessage =
    "Recording the slot of the reusable command buffer failed";

} // namespace

CommandBuffer::CommandBuffer(VulkanContext& ctx, CommandBufferDesc desc) :
  ICommandBuffer(std::move(desc)),
  ctx_(ctx),
  wrapper_(&ctx_.immediate_->acquire()),
  cmdBuf_(wrapper_->cmdBuf) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(cmdBuf_ != VK_NULL_HANDLE);

  // The `desc` parameter was moved into the base class, so use the member
  if (this->desc.timer) {
    auto& timer = static_cast<Timer&>(*this->desc.timer);
    IGL_DEBUG_ASSERT(timer.belongsTo(ctx_), "The timer was created by another Vulkan device");
    timerSlot_ = timer.begin(cmdBuf_);
  }
}

CommandBuffer::Reusable::Reusable(VulkanContext& ctx,
                                  VkCommandPool commandPool,
                                  bool executeInCommandBuffer) :
  ctx(ctx),
  commandPool(commandPool),
  executeInCommandBuffer(executeInCommandBuffer),
  slots(ctx.config_.maxResourceCount),
  arenas(ctx.config_.maxResourceCount) {
  for (std::shared_ptr<DescriptorPoolsArenas>& slotArenas : arenas) {
    slotArenas = ctx.createDescriptorPoolsArenas();
  }
}

CommandBuffer::Reusable::~Reusable() {
  ctx.destroyRecordings(commandPool, arenas, lastSubmission);
}

CommandBuffer::CommandBuffer(VulkanContext& ctx,
                             const ReusableCommandBufferDesc& desc,
                             VkCommandPool commandPool) :
  ICommandBuffer(CommandBufferDesc{.debugName = desc.debugName}),
  ctx_(ctx),
  handle_(uint64_t{1} << 32),
  reusable_(std::make_shared<Reusable>(ctx, commandPool, desc.executeInCommandBuffer)) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
}

CommandBuffer::~CommandBuffer() {
  // A command buffer that executes slots of reusable command buffers and is destroyed before it is
  // submitted runs nothing: it gives back the slots, its Vulkan command buffer and its count in the
  // queue
  if (executesSlots_ && !isSubmitted_) {
    releaseExecutedSlots({});
    ctx_.immediate_->release(*wrapper_);
    if (numBuffersLeftToSubmit_) {
      --*numBuffersLeftToSubmit_;
    }
  }
  if (!reusable_) {
    return;
  }
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  // the slot that is being recorded never runs
  if (reusable_->recordingSlot != kNoSlot) {
    failRecording(Result(Result::Code::RuntimeError,
                         "The reusable command buffer was destroyed while it recorded a slot"));
  }
}

VkCommandBuffer IGL_NULLABLE CommandBuffer::getReusableVkCommandBuffer() const {
  if (reusable_->recordingSlot != kNoSlot) {
    // the recording of a slot runs only while its slot is the current one
    if (cmdBuf_ != VK_NULL_HANDLE && reusable_->recordingSlot != ctx_.currentSyncIndex()) {
      failRecording(Result(Result::Code::InvalidOperation,
                           "Another submission moved the reusable command buffer to its next slot "
                           "while it recorded a slot"));
    }
    return cmdBuf_;
  }
  const uint32_t slot = ctx_.currentSyncIndex();
  if (!IGL_DEBUG_VERIFY(!reusable_->slots[slot].isRecorded,
                        "Encode into a reusable command buffer only while isRecorded() is false")) {
    return VK_NULL_HANDLE;
  }
  reusable_->recordingSlot = slot;
  reusable_->recordingDrawCount = getCurrentDrawCount();
  cmdBuf_ = beginSlotCommandBuffer(VK_NULL_HANDLE, VK_NULL_HANDLE);
  if (cmdBuf_ == VK_NULL_HANDLE) {
    failRecording(Result(Result::Code::RuntimeError, kRecordingFailedMessage));
  } else {
    reusable_->slots[slot].commands.push_back(cmdBuf_);
    reusable_->journal.cmdBuf = cmdBuf_;
    ctx_.imageLayoutJournals_.push_back(&reusable_->journal);
  }
  return cmdBuf_;
}

VkCommandBuffer IGL_NULLABLE
CommandBuffer::beginSlotCommandBuffer(VkRenderPass renderPass, VkFramebuffer framebuffer) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  const bool isSecondary = reusable_->executeInCommandBuffer;
  const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = reusable_->commandPool,
      .level = isSecondary ? VK_COMMAND_BUFFER_LEVEL_SECONDARY : VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
  const VkResult allocated = ctx_.vf_.vkAllocateCommandBuffers(ctx_.getVkDevice(), &ai, &cmdBuf);
  VK_ASSERT(allocated);
  if (allocated != VK_SUCCESS || cmdBuf == VK_NULL_HANDLE) {
    return VK_NULL_HANDLE;
  }
  ivkSetDebugObjectName(
      &ctx_.vf_,
      ctx_.getVkDevice(),
      VK_OBJECT_TYPE_COMMAND_BUFFER,
      (uint64_t)cmdBuf,
      IGL_FORMAT("Command Buffer: {} (slot {})", desc.debugName, reusable_->recordingSlot).c_str());

  const VkCommandBufferInheritanceInfo inheritanceInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
      .renderPass = renderPass,
      .subpass = 0,
      .framebuffer = framebuffer,
  };
  // Without VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, every frame can run the recording again
  const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = renderPass != VK_NULL_HANDLE ? VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
                                            : VkCommandBufferUsageFlags{0},
      .pInheritanceInfo = isSecondary ? &inheritanceInfo : nullptr,
  };
  const VkResult result = ctx_.vf_.vkBeginCommandBuffer(cmdBuf, &bi);
  VK_ASSERT(result);
  if (result != VK_SUCCESS) {
    ctx_.vf_.vkFreeCommandBuffers(ctx_.getVkDevice(), reusable_->commandPool, 1, &cmdBuf);
    return VK_NULL_HANDLE;
  }
  return cmdBuf;
}

VkCommandBuffer IGL_NULLABLE CommandBuffer::nextSlotCommandBuffer(VkRenderPass renderPass,
                                                                  VkFramebuffer framebuffer) {
  const VkResult result = ctx_.vf_.vkEndCommandBuffer(cmdBuf_);
  VK_ASSERT(result);
  const VkCommandBuffer cmdBuf =
      result == VK_SUCCESS ? beginSlotCommandBuffer(renderPass, framebuffer) : VK_NULL_HANDLE;
  if (cmdBuf == VK_NULL_HANDLE) {
    failRecording(Result(Result::Code::RuntimeError, kRecordingFailedMessage));
    return VK_NULL_HANDLE;
  }
  cmdBuf_ = cmdBuf;
  reusable_->journal.cmdBuf = cmdBuf_;
  return cmdBuf_;
}

void CommandBuffer::failRecording(Result failure) const {
  Result& recordingFailure = reusable_->slots[reusable_->recordingSlot].recordingFailure;
  if (!recordingFailure.isOk()) {
    return;
  }
  recordingFailure = std::move(failure);
  // the layout transitions of the recording never run
  for (const ImageLayoutJournal::Layout& layout : reusable_->journal.layouts) {
    if (!layout.isChangedElsewhere) {
      layout.image->imageLayout_ = layout.layout;
    }
  }
  stopImageLayoutJournal();
  cmdBuf_ = VK_NULL_HANDLE;
}

void CommandBuffer::stopImageLayoutJournal() const {
  std::vector<ImageLayoutJournal*>& journals = ctx_.imageLayoutJournals_;
  journals.erase(std::remove(journals.begin(), journals.end(), &reusable_->journal),
                 journals.end());
  reusable_->journal = {};
}

void CommandBuffer::discardSlot(uint32_t slotIndex) const {
  ReusableSlot& slot = reusable_->slots[slotIndex];
  std::vector<VkCommandBuffer> cmdBufs = std::move(slot.commands);
  for (const ReusableRenderPass& renderPass : slot.renderPasses) {
    cmdBufs.push_back(renderPass.commands);
  }
  if (!cmdBufs.empty()) {
    ctx_.vf_.vkFreeCommandBuffers(ctx_.getVkDevice(),
                                  reusable_->commandPool,
                                  static_cast<uint32_t>(cmdBufs.size()),
                                  cmdBufs.data());
  }
  // no submission ran the freed command buffers, the only ones that used the descriptor sets
  ctx_.destroyRecordings(VK_NULL_HANDLE, {reusable_->arenas[slotIndex]}, {});
  slot = ReusableSlot();
}

bool CommandBuffer::isRecorded() const {
  return reusable_ && reusable_->slots[ctx_.currentSyncIndex()].isRecorded;
}

CommandBuffer::ReusableSlot* IGL_NULLABLE CommandBuffer::prepareRun(Result& outResult) {
  IGL_PROFILER_FUNCTION();

  outResult = Result();
  uint32_t slotIndex = reusable_->recordingSlot;
  if (slotIndex != kNoSlot && reusable_->isInRenderPass) {
    outResult = Result(Result::Code::InvalidOperation,
                       "A render pass of the reusable command buffer is being encoded");
    return nullptr;
  }
  if (slotIndex != kNoSlot) {
    // The first run of a slot ends its recording. A failed recording, which includes one whose slot
    // is no longer the current one, has no command buffer to end.
    ReusableSlot& slot = reusable_->slots[slotIndex];
    const VkCommandBuffer cmdBuf = getVkCommandBuffer();
    if (cmdBuf != VK_NULL_HANDLE) {
      const VkResult result = ctx_.vf_.vkEndCommandBuffer(cmdBuf);
      VK_ASSERT(result);
      if (result != VK_SUCCESS) {
        failRecording(Result(Result::Code::RuntimeError, kRecordingFailedMessage));
      }
    }
    stopImageLayoutJournal();
    slot.numDraws = getCurrentDrawCount() - reusable_->recordingDrawCount;
    reusable_->recordingSlot = kNoSlot;
    cmdBuf_ = VK_NULL_HANDLE;
    if (!slot.recordingFailure.isOk()) {
      outResult = std::move(slot.recordingFailure);
      discardSlot(slotIndex);
      return nullptr;
    }
    slot.isRecorded = true;
  } else {
    slotIndex = ctx_.currentSyncIndex();
  }
  ReusableSlot& slot = reusable_->slots[slotIndex];
  if (!slot.isRecorded) {
    return nullptr;
  }
  // its secondary command buffers can be in one command buffer that is not pending yet
  if (slot.executor) {
    outResult = Result(Result::Code::InvalidOperation,
                       "A command buffer that has not been submitted executes the current slot of "
                       "the reusable command buffer");
    return nullptr;
  }
  // a command buffer recorded without VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT cannot run while
  // it is pending
  const VkResult result = ctx_.immediate_->wait(slot.lastRun, ctx_.config_.fenceTimeoutNanoseconds);
  if (result != VK_SUCCESS) {
    outResult = Result(Result::Code::RuntimeError,
                       IGL_FORMAT("The last run of the slot of the reusable command buffer did not "
                                  "complete: {}",
                                  ivkGetVulkanResultString(result)));
    return nullptr;
  }
  return &slot;
}

VkCommandBuffer IGL_NULLABLE CommandBuffer::beginRenderPass(const VkRenderPassBeginInfo& bi) {
  if (!reusable_) {
    ctx_.vf_.vkCmdBeginRenderPass(cmdBuf_, &bi, VK_SUBPASS_CONTENTS_INLINE);
    return cmdBuf_;
  }
  IGL_PROFILER_FUNCTION();

  if (!reusable_->executeInCommandBuffer) {
    ctx_.vf_.vkCmdBeginRenderPass(cmdBuf_, &bi, VK_SUBPASS_CONTENTS_INLINE);
    reusable_->isInRenderPass = true;
    return cmdBuf_;
  }
  // vkCmdBeginRenderPass() records only into primary command buffers: the command buffer that
  // executes the slot begins the render pass
  if (nextSlotCommandBuffer(bi.renderPass, bi.framebuffer) == VK_NULL_HANDLE) {
    return VK_NULL_HANDLE;
  }
  reusable_->isInRenderPass = true;
  ReusableRenderPass& renderPass =
      reusable_->slots[reusable_->recordingSlot].renderPasses.emplace_back();
  IGL_DEBUG_ASSERT(bi.clearValueCount <= renderPass.clearValues.size());
  renderPass.beginInfo = bi;
  std::copy_n(bi.pClearValues,
              std::min<size_t>(bi.clearValueCount, renderPass.clearValues.size()),
              renderPass.clearValues.begin());
  renderPass.commands = cmdBuf_;
  return cmdBuf_;
}

VkCommandBuffer IGL_NULLABLE CommandBuffer::endRenderPass() {
  if (!reusable_) {
    ctx_.vf_.vkCmdEndRenderPass(cmdBuf_);
    return cmdBuf_;
  }
  IGL_PROFILER_FUNCTION();

  reusable_->isInRenderPass = false;
  // the recording failed, or the slot ran, while the render pass was being recorded
  if (reusable_->recordingSlot == kNoSlot || getVkCommandBuffer() == VK_NULL_HANDLE) {
    return VK_NULL_HANDLE;
  }
  if (!reusable_->executeInCommandBuffer) {
    ctx_.vf_.vkCmdEndRenderPass(cmdBuf_);
    return cmdBuf_;
  }
  const VkCommandBuffer cmdBuf = nextSlotCommandBuffer(VK_NULL_HANDLE, VK_NULL_HANDLE);
  if (cmdBuf != VK_NULL_HANDLE) {
    reusable_->slots[reusable_->recordingSlot].commands.push_back(cmdBuf);
  }
  return cmdBuf;
}

void CommandBuffer::executeCommands(ICommandBuffer& reusable, Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx_);

  if (!ctx_.supportsReusableCommandBuffers()) {
    Result::setResult(
        outResult, Result::Code::Unsupported, "Reusable command buffers are not supported");
    return;
  }
  auto& vkReusable = static_cast<CommandBuffer&>(reusable);
  // before anything of another device reaches this command buffer
  if (!vkReusable.belongsTo(ctx_)) {
    Result::setResult(outResult,
                      Result::Code::ArgumentInvalid,
                      "The reusable command buffer belongs to another device");
    return;
  }
  if (!IGL_DEBUG_VERIFY(wrapper_ && vkReusable.reusable_ &&
                        vkReusable.reusable_->executeInCommandBuffer)) {
    Result::setResult(outResult,
                      Result::Code::ArgumentInvalid,
                      "Only a command buffer that is not reusable executes a reusable one created "
                      "with ReusableCommandBufferDesc::executeInCommandBuffer");
    return;
  }

  Result result;
  ReusableSlot* slot = vkReusable.prepareRun(result);
  if (!result.isOk()) {
    Result::setResult(outResult, std::move(result));
    return;
  }
  if (slot) {
    // No Tracy GPU zone: this command buffer can be destroyed without being submitted, which would
    // leave queries that Tracy waits for and cannot take back
    for (size_t i = 0; i != slot->commands.size(); ++i) {
      if (i != 0) {
        // the render pass between slot->commands[i - 1] and slot->commands[i]
        const ReusableRenderPass& renderPass = slot->renderPasses[i - 1];
        VkRenderPassBeginInfo bi = renderPass.beginInfo;
        bi.pClearValues = renderPass.clearValues.data();
        ctx_.vf_.vkCmdBeginRenderPass(cmdBuf_, &bi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
        ctx_.vf_.vkCmdExecuteCommands(cmdBuf_, 1, &renderPass.commands);
        ctx_.vf_.vkCmdEndRenderPass(cmdBuf_);
      }
      ctx_.vf_.vkCmdExecuteCommands(cmdBuf_, 1, &slot->commands[i]);
    }
    // the slot runs once this command buffer is submitted
    slot->executor = this;
    executedSlots_.push_back({
        .reusable = vkReusable.reusable_,
        .slot = static_cast<uint32_t>(slot - vkReusable.reusable_->slots.data()),
    });
    executedDrawCount_ += slot->numDraws;
  }
  // Submitting this command buffer moves the reusable one to its next slot, as it does after GPU
  // work reads a Ring buffer
  ctx_.markRingBufferRead();
  executesSlots_ = true;
  Result::setOk(outResult);
}

void CommandBuffer::releaseExecutedSlots(VulkanImmediateCommands::SubmitHandle submission) {
  for (const ExecutedSlot& executed : executedSlots_) {
    ReusableSlot& slot = executed.reusable->slots[executed.slot];
    slot.executor = nullptr;
    if (!submission.empty()) {
      slot.lastRun = submission;
      executed.reusable->lastSubmission = submission;
    }
  }
  executedSlots_.clear();
}

std::unique_ptr<IComputeCommandEncoder> CommandBuffer::createComputeCommandEncoder() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (getVkCommandBuffer() == VK_NULL_HANDLE) {
    return nullptr;
  }
  return std::make_unique<ComputeCommandEncoder>(shared_from_this(), ctx_);
}

std::unique_ptr<IRenderCommandEncoder> CommandBuffer::createRenderCommandEncoder(
    const RenderPassDesc& renderPass,
    const std::shared_ptr<IFramebuffer>& framebuffer,
    const Dependencies& dependencies,
    Result* outResult) {
  IGL_PROFILER_FUNCTION();
  // every run would write the timestamps again into queries that nothing resets
  if (reusable_ && renderPass.timestampQuery.queries) {
    Result::setResult(outResult,
                      Result::Code::Unsupported,
                      "A reusable command buffer does not support timestamp queries");
    return nullptr;
  }
  if (isRecorded()) {
    Result::setResult(outResult,
                      Result::Code::InvalidOperation,
                      "The current slot of the reusable command buffer holds a recording");
    return nullptr;
  }
  const bool hasSubpasses = !renderPass.subpasses.empty();
  if (hasSubpasses) {
    if (!ctx_.supportsSubpasses()) {
      Result::setResult(outResult, Result::Code::Unsupported, "Subpasses are not supported");
      return nullptr;
    }
    // every subpass would need secondary command buffers of its own
    if (reusable_ && reusable_->executeInCommandBuffer) {
      Result::setResult(outResult,
                        Result::Code::Unsupported,
                        "A reusable command buffer that runs from another command buffer does not "
                        "support subpasses");
      return nullptr;
    }
    if (!framebuffer) {
      Result::setResult(outResult, Result::Code::ArgumentNull, "The framebuffer is null");
      return nullptr;
    }
    if (framebuffer->getMode() == FramebufferMode::Stereo && !ctx_.supportsMultiviewSubpasses()) {
      Result::setResult(outResult, Result::Code::Unsupported, "Stereo subpasses are not supported");
      return nullptr;
    }
    const Result validation = renderPass.validate(*framebuffer);
    if (!validation.isOk()) {
      Result::setResult(outResult, validation);
      return nullptr;
    }
  }
#if IGL_DEBUG_ABORT_ENABLED
  if (!hasSubpasses && framebuffer) {
    const Result validation = renderPass.validate(*framebuffer);
    if (!validation.isOk()) {
      IGL_LOG_ERROR_ONCE("Invalid RenderPassDesc: %s\n", validation.message.c_str());
    }
  }
#endif
  IGL_DEBUG_ASSERT(framebuffer);

  const VkCommandBuffer cmdBuf = getVkCommandBuffer();
  if (cmdBuf == VK_NULL_HANDLE) {
    // isRecorded() is false: the recording of the current slot failed
    Result::setResult(outResult, reusable_->slots[reusable_->recordingSlot].recordingFailure);
    return nullptr;
  }

  framebuffer_ = framebuffer;

  // prepare all the color attachments
  for (const auto i : framebuffer->getColorAttachmentIndices()) {
    ITexture* colorTex = framebuffer->getColorAttachment(i).get();
    transitionToColorAttachment(cmdBuf, colorTex);
    // handle MSAA
    ITexture* colorResolveTex = framebuffer->getResolveColorAttachment(i).get();
    transitionToColorAttachment(cmdBuf, colorResolveTex);
  }

  // prepare depth attachment
  const auto depthTex = framebuffer->getDepthAttachment();
  if (depthTex) {
    const auto& vkDepthTex = static_cast<Texture&>(*depthTex);
    const VulkanImage& depthImg = vkDepthTex.getVulkanTexture().image;
    IGL_DEBUG_ASSERT(depthImg.imageFormat_ != VK_FORMAT_UNDEFINED,
                     "Invalid depth attachment format");
    const VkImageAspectFlags flags = vkDepthTex.getVulkanTexture().image.getImageAspectFlags();
    depthImg.transitionLayout(cmdBuf,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                              VkImageSubresourceRange{.aspectMask = flags,
                                                      .baseMipLevel = 0,
                                                      .levelCount = VK_REMAINING_MIP_LEVELS,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = VK_REMAINING_ARRAY_LAYERS});
  }

  auto encoder = RenderCommandEncoder::create(
      shared_from_this(), ctx_, renderPass, framebuffer, dependencies, outResult);

  return encoder;
}

void CommandBuffer::present(const std::shared_ptr<ITexture>& surface) const {
  IGL_PROFILER_FUNCTION();

  IGL_DEBUG_ASSERT(surface);

  const auto& vkTex = static_cast<Texture&>(*surface);
  // every run of a recording would present the same swapchain image
  if (!IGL_DEBUG_VERIFY(!reusable_ || !vkTex.isSwapchainTexture(),
                        "A reusable command buffer cannot present a swapchain image")) {
    return;
  }
  const VkCommandBuffer cmdBuf = getVkCommandBuffer();
  if (cmdBuf == VK_NULL_HANDLE) {
    return;
  }

  presentedSurface_ = surface;

  const VulkanTexture& tex = vkTex.getVulkanTexture();
  const VulkanImage& img = tex.image;

  // prepare image for presentation
  if (vkTex.isSwapchainTexture()) {
    isFromSwapchain_ = true;
    // the image might be coming from a compute shader
    const VkPipelineStageFlagBits srcStage = (img.imageLayout_ == VK_IMAGE_LAYOUT_GENERAL)
                                                 ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                                 : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    img.transitionLayout(cmdBuf,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         srcStage,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // wait for all subsequent operations
                         VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                 .baseMipLevel = 0,
                                                 .levelCount = VK_REMAINING_MIP_LEVELS,
                                                 .baseArrayLayer = 0,
                                                 .layerCount = VK_REMAINING_ARRAY_LAYERS});
    return;
  }

  isFromSwapchain_ = false;

  // transition only non-multisampled images - MSAA images cannot be accessed from shaders
  if (img.samples_ == VK_SAMPLE_COUNT_1_BIT) {
    const VkImageAspectFlags flags = vkTex.getVulkanTexture().image.getImageAspectFlags();
    const VkPipelineStageFlags srcStage = vkTex.getProperties().isDepthOrStencil()
                                              ? VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                                              : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // set the result of the previous render pass
    img.transitionLayout(cmdBuf,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         srcStage,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // wait for subsequent
                                                                   // fragment/compute shaders
                         VkImageSubresourceRange{.aspectMask = flags,
                                                 .baseMipLevel = 0,
                                                 .levelCount = VK_REMAINING_MIP_LEVELS,
                                                 .baseArrayLayer = 0,
                                                 .layerCount = VK_REMAINING_ARRAY_LAYERS});
  }
}

void CommandBuffer::pushDebugGroupLabel(const char* label, const Color& color) const {
  IGL_DEBUG_ASSERT(label && *label);
  // a label cannot span the secondary command buffers of a slot
  const VkCommandBuffer cmdBuf =
      reusable_ && reusable_->executeInCommandBuffer ? VK_NULL_HANDLE : getVkCommandBuffer();
  if (cmdBuf != VK_NULL_HANDLE) {
    ivkCmdBeginDebugUtilsLabel(&ctx_.vf_, cmdBuf, label, color.toFloatPtr());
  }
}

void CommandBuffer::popDebugGroupLabel() const {
  const VkCommandBuffer cmdBuf =
      reusable_ && reusable_->executeInCommandBuffer ? VK_NULL_HANDLE : getVkCommandBuffer();
  if (cmdBuf != VK_NULL_HANDLE) {
    ivkCmdEndDebugUtilsLabel(&ctx_.vf_, cmdBuf);
  }
}

void CommandBuffer::copyBuffer(IBuffer& src,
                               IBuffer& dst,
                               uint64_t srcOffset,
                               uint64_t dstOffset,
                               uint64_t size) {
  IGL_PROFILER_FUNCTION();

  const VkCommandBuffer cmdBuf = getVkCommandBuffer();
  if (cmdBuf == VK_NULL_HANDLE) {
    return;
  }

  const auto& bufSrc = static_cast<Buffer&>(src);
  const auto& bufDst = static_cast<Buffer&>(dst);

  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   bufSrc.getVkBuffer(),
                   bufSrc.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   bufDst.getVkBuffer(),
                   bufDst.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);

  const VkBufferCopy region = {
      .srcOffset = srcOffset,
      .dstOffset = dstOffset,
      .size = size,
  };

  ctx_.vf_.vkCmdCopyBuffer(cmdBuf, bufSrc.getVkBuffer(), bufDst.getVkBuffer(), 1, &region);

  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   bufSrc.getVkBuffer(),
                   bufSrc.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   bufDst.getVkBuffer(),
                   bufDst.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}

void CommandBuffer::fillBuffer(IBuffer& buffer, const BufferRange& range, uint8_t value) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(range.offset % 4u == 0u && range.size % 4u == 0u);
  IGL_DEBUG_ASSERT(range.offset + range.size <= buffer.getSizeInBytes());

  const auto& vulkanBuffer = static_cast<Buffer&>(buffer);
  IGL_DEBUG_ASSERT((vulkanBuffer.getBufferUsageFlags() & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0u);

  const VkCommandBuffer cmdBuf = getVkCommandBuffer();
  if (cmdBuf == VK_NULL_HANDLE) {
    return;
  }

  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   vulkanBuffer.getVkBuffer(),
                   vulkanBuffer.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);

  const uint32_t fillValue = static_cast<uint32_t>(value) * 0x01010101u;
  ctx_.vf_.vkCmdFillBuffer(cmdBuf, vulkanBuffer.getVkBuffer(), range.offset, range.size, fillValue);

  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   vulkanBuffer.getVkBuffer(),
                   vulkanBuffer.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}

void CommandBuffer::updateBuffer(IBuffer& buffer, const BufferRange& range, const void* data) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(data != nullptr);
  IGL_DEBUG_ASSERT(range.offset % 4u == 0u && range.size % 4u == 0u);
  IGL_DEBUG_ASSERT(range.offset + range.size <= buffer.getSizeInBytes());

  const auto& vulkanBuffer = static_cast<Buffer&>(buffer);
  IGL_DEBUG_ASSERT((vulkanBuffer.getBufferUsageFlags() & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0u);
  // A Ring buffer has one VkBuffer per in-flight index; the write would land in only one of them.
  IGL_DEBUG_ASSERT(!vulkanBuffer.isRingBuffer(), "updateBuffer() cannot write into a Ring buffer");

  const VkCommandBuffer cmdBuf = getVkCommandBuffer();
  if (data == nullptr || range.size == 0 || cmdBuf == VK_NULL_HANDLE) {
    return;
  }

  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   vulkanBuffer.getVkBuffer(),
                   vulkanBuffer.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);

  // vkCmdUpdateBuffer() takes at most 65536 bytes per call (VUID-vkCmdUpdateBuffer-dataSize-00037).
  constexpr size_t kMaxUpdateSize = 65536;
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t done = 0; done < range.size; done += kMaxUpdateSize) {
    const size_t size = std::min(kMaxUpdateSize, range.size - done);
    ctx_.vf_.vkCmdUpdateBuffer(cmdBuf,
                               vulkanBuffer.getVkBuffer(),
                               static_cast<VkDeviceSize>(range.offset) + done,
                               static_cast<VkDeviceSize>(size),
                               bytes + done);
  }

  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   vulkanBuffer.getVkBuffer(),
                   vulkanBuffer.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}

void CommandBuffer::copyTextureToBuffer(ITexture& src,
                                        IBuffer& dst,
                                        uint64_t dstOffset,
                                        uint32_t level,
                                        uint32_t layer) {
  IGL_PROFILER_FUNCTION();

  const VkCommandBuffer cmdBuf = getVkCommandBuffer();
  if (cmdBuf == VK_NULL_HANDLE) {
    return;
  }

  const auto& texSrc = static_cast<Texture&>(src);
  const auto& bufDst = static_cast<Buffer&>(dst);

  VulkanImage& image = texSrc.getVulkanTexture().image;

  const VkImageLayout oldLayout = image.imageLayout_;

  IGL_DEBUG_ASSERT(oldLayout != VK_IMAGE_LAYOUT_UNDEFINED);

  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   bufDst.getVkBuffer(),
                   bufDst.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);

  const VkImageAspectFlags aspectMask =
      image.isDepthFormat_
          ? VK_IMAGE_ASPECT_DEPTH_BIT
          : (image.isStencilFormat_ ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);

  const VkImageSubresourceRange range = {
      .aspectMask = aspectMask,
      .baseMipLevel = level,
      .levelCount = 1u,
      .baseArrayLayer = layer,
      .layerCount = texSrc.getNumFaces() == 6 ? 6u : 1u,
  };

  image.transitionLayout(cmdBuf,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         range);

  const VkBufferImageCopy region = {
      .bufferOffset = dstOffset,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource =
          {
              .aspectMask = aspectMask,
              .mipLevel = level,
              .baseArrayLayer = layer,
              .layerCount = texSrc.getNumFaces() == 6 ? 6u : 1u,
          },
      .imageOffset = {},
      .imageExtent = image.extent_,
  };

  ctx_.vf_.vkCmdCopyImageToBuffer(cmdBuf,
                                  texSrc.getVkImage(),
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  bufDst.getVkBuffer(),
                                  1u,
                                  &region);

  ivkBufferBarrier(&ctx_.vf_,
                   cmdBuf,
                   bufDst.getVkBuffer(),
                   bufDst.getBufferUsageFlags(),
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  image.transitionLayout(
      cmdBuf, oldLayout, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, range);
}

void CommandBuffer::waitUntilCompleted() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  ctx_.immediate_->wait(reusable_ ? reusable_->lastSubmission : lastSubmitHandle_,
                        ctx_.config_.fenceTimeoutNanoseconds);

  lastSubmitHandle_ = VulkanImmediateCommands::SubmitHandle();
}

void CommandBuffer::waitUntilScheduled() {}

const std::shared_ptr<IFramebuffer>& CommandBuffer::getFramebuffer() const {
  return framebuffer_;
}

const std::shared_ptr<ITexture>& CommandBuffer::getPresentedSurface() const {
  return presentedSurface_;
}

} // namespace igl::vulkan
