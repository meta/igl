/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/CommandQueue.h>

#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/Timer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanSwapchain.h>

#define IGL_COMMAND_QUEUE_DEBUG_FENCES (IGL_DEBUG && 0)

#if IGL_COMMAND_QUEUE_DEBUG_FENCES
#include <cinttypes> // PRIx3
#include <igl/vulkan/VulkanFence.h>
#endif // IGL_COMMAND_QUEUE_DEBUG_FENCES

namespace igl::vulkan {

CommandQueue::CommandQueue(Device& device, const CommandQueueDesc& /*desc*/) : device_(device) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
}

std::shared_ptr<ICommandBuffer> CommandQueue::createCommandBuffer(const CommandBufferDesc& desc,
                                                                  Result* IGL_NULLABLE
                                                                  /*outResult*/) {
  IGL_PROFILER_FUNCTION();

  VulkanContext& ctx = device_.getVulkanContext();
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx);

  ++*numBuffersLeftToSubmit_;

  auto cmdBuffer = std::make_shared<CommandBuffer>(ctx, desc);
  cmdBuffer->numBuffersLeftToSubmit_ = numBuffersLeftToSubmit_;
  return cmdBuffer;
}

SubmitHandle CommandQueue::submit(const ICommandBuffer& cmdBuffer, bool /* endOfFrame */) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_SUBMIT);

  VulkanContext& ctx = device_.getVulkanContext();
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx);

  auto* vkCmdBuffer = const_cast<CommandBuffer*>(static_cast<const CommandBuffer*>(&cmdBuffer));

  // A reusable command buffer is not counted as left to submit, cannot present a swapchain image,
  // and each run counts the draws of its slot
  if (vkCmdBuffer->reusable_) {
    if (!IGL_DEBUG_VERIFY(
            !vkCmdBuffer->reusable_->executeInCommandBuffer,
            "Run this reusable command buffer with PlatformDevice::executeCommands()")) {
      return 0;
    }
    return endCommandBuffer(ctx, vkCmdBuffer, false);
  }

  incrementDrawCount(cmdBuffer.getCurrentDrawCount());

  --*numBuffersLeftToSubmit_;

#if IGL_COMMAND_QUEUE_DEBUG_FENCES
  // Create label with Fence handle and Fence FD, if available
  // A string such as "Submit command buffer (hex: 0x149b90a10, fd: 12345)" has 55 characters
  char labelName[60] = {};
  if (vkCmdBuffer->wrapper_.fence_.exportable()) {
    std::snprintf(labelName,
                  sizeof(labelName),
                  "Submit command buffer (hex: %#" PRIx64 ", fd: %d)",
                  reinterpret_cast<uint64_t>(vkCmdBuffer->wrapper_.fence_.vkFence_),
                  ctx.immediate_->cachedFDFromSubmitHandle(vkCmdBuffer->wrapper_.handle_));
  } else {
    std::snprintf(labelName,
                  sizeof(labelName),
                  "Submit command buffer (hex: %#" PRIx64 ")",
                  reinterpret_cast<uint64_t>(vkCmdBuffer->wrapper_.fence_.vkFence_));
  }

  ivkCmdInsertDebugUtilsLabel(&ctx.vf_,
                              vkCmdBuffer->getVkCommandBuffer(),
                              labelName,
                              K_COLOR_COMMAND_BUFFER_SUBMISSION_WITH_FENCE.toFloatPtr());
#endif // IGL_COMMAND_QUEUE_DEBUG_FENCES

  const auto submitHandle = endCommandBuffer(ctx, vkCmdBuffer, true);

  return submitHandle;
}

SubmitHandle CommandQueue::endCommandBuffer(VulkanContext& ctx,
                                            CommandBuffer* cmdBuffer,
                                            bool present) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_SUBMIT);
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx);

  auto* const timer = static_cast<Timer*>(cmdBuffer->desc.timer.get());
  cmdBuffer->isSubmitted_ = true;

  // Submit to the graphics queue.
  const bool shouldPresent = ctx.hasSwapchain() && cmdBuffer->isFromSwapchain() && present;
  const auto finishCommandBuffer = [&](VulkanImmediateCommands::SubmitHandle submitHandle) {
    cmdBuffer->lastSubmitHandle_ = submitHandle;
    ctx.syncMarkSubmitted(submitHandle);
    ctx.processDeferredTasks();
    ctx.stagingDevice_->mergeRegionsAndFreeBuffers();
    return submitHandle.handle();
  };
  // discard() takes only the current command buffer. One that executes slots of reusable command
  // buffers gives back its Vulkan command buffer even after a later one was acquired, as it does
  // when it is destroyed without being submitted, and gives back its slots.
  const auto discard = [&ctx, cmdBuffer]() {
    if (!cmdBuffer->executesSlots_) {
      ctx.immediate_->discard(*cmdBuffer->wrapper_);
      return;
    }
    if (cmdBuffer->wrapper_->handle == ctx.immediate_->getNextSubmitHandle()) {
      ctx.immediate_->discard(*cmdBuffer->wrapper_);
    } else {
      ctx.immediate_->release(*cmdBuffer->wrapper_);
    }
    cmdBuffer->releaseExecutedSlots({});
  };
  if (shouldPresent) {
    // Injected-semaphore overflow below discards the command buffer and skips present(), leaving
    // the swapchain image acquired this frame unpresented; sustained failures can perturb swapchain
    // state and starve acquireNextImage(). Callers must stay within kMaxInjectedSemaphores.
    if (ctx.timelineSemaphore_) {
      // if we are presenting a swapchain image, signal our timeline semaphore
      const uint64_t signalValue =
          ctx.swapchain_->getFrameNumber() + ctx.swapchain_->getNumSwapchainImages();
      // we wait for this value next time we want to acquire this swapchain image
      if (!ctx.immediate_->signalSemaphore(ctx.timelineSemaphore_->getVkSemaphore(), signalValue)) {
        discard();
        return finishCommandBuffer({});
      }
      ctx.swapchain_->timelineWaitValues[ctx.swapchain_->getCurrentImageIndex()] = signalValue;
    } else {
      // this can be removed once we switch to timeline semaphores
      if (!ctx.immediate_->waitSemaphore(ctx.swapchain_->getSemaphore())) {
        discard();
        return finishCommandBuffer({});
      }
    }
  }

  if (timer) {
    timer->end(cmdBuffer->wrapper_->cmdBuf, cmdBuffer->timerSlot_);
  }
  // Presentation can outlive the submit fence. Its semaphore is reusable only after the same
  // swapchain image is reacquired and does not consume the caller's injected-signal capacity.
  const VkSemaphore presentationSemaphore = shouldPresent ? ctx.swapchain_->getPresentSemaphore()
                                                          : VK_NULL_HANDLE;
  CommandBuffer::ReusableSlot* slot = nullptr;
  if (cmdBuffer->reusable_) {
    Result result;
    slot = cmdBuffer->prepareRun(result);
    if (!result.isOk()) {
      IGL_LOG_ERROR("CommandQueue::submit(): %s\n", result.message.c_str());
      return 0;
    }
  }
  const auto submitHandle =
      ctx.immediate_->submit(cmdBuffer->wrapper_ ? *cmdBuffer->wrapper_ : ctx.immediate_->acquire(),
                             presentationSemaphore,
                             slot ? slot->commands.front() : VK_NULL_HANDLE);
  // the draw statistics count the draws of the slots that run
  if (slot) {
    slot->lastRun = submitHandle;
    cmdBuffer->reusable_->lastSubmission = submitHandle;
    incrementDrawCount(slot->numDraws);
  } else {
    incrementDrawCount(cmdBuffer->executedDrawCount_);
  }
  if (cmdBuffer->reusable_) {
    // every run moves to the next slot, as a submit that reads a Ring buffer does
    ctx.markRingBufferRead();
  }
  // the slots that the command buffer executes run in this submission
  cmdBuffer->releaseExecutedSlots(submitHandle);
  if (timer) {
    timer->submitted(cmdBuffer->timerSlot_, submitHandle);
  }

  if (shouldPresent) {
    ctx.present();
  }
  // Without descriptor indexing no descriptor set refers to texture slots, so textures that only
  // this context still references can be released here. When no other command buffer is being
  // recorded, no unsubmitted work can use them, and their deferred destruction is tied to this
  // submission, whose fence also covers every earlier submission. A later submit-and-wait then
  // frees every texture the app released.
  if (!ctx.config_.enableDescriptorIndexing && !ctx.immediate_->isAnyCommandBufferEncoding()) {
    ctx.pruneTextures();
  }
  return finishCommandBuffer(submitHandle);
}

} // namespace igl::vulkan
