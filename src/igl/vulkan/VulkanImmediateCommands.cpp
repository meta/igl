/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanImmediateCommands.h"

#include <algorithm>
#include <array>
#include <type_traits>
#include <igl/vulkan/Common.h>

namespace igl::vulkan {

static_assert(std::is_trivially_copyable_v<VulkanImmediateCommands::SubmitHandle>);

VulkanImmediateCommands::VulkanImmediateCommands(const VulkanFunctionTable& vf,
                                                 VkDevice device,
                                                 uint32_t queueFamilyIndex,
                                                 bool exportableFences,
                                                 bool useTimelineSemaphore,
                                                 bool useSynchronization2,
                                                 const char* debugName,
                                                 bool chainSubmitsWithSemaphores) :
  vf_(vf),
  device_(device),
  debugName_(debugName),
  lastSubmitSemaphore_({
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = VK_NULL_HANDLE,
      .value = 0ULL,
      .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      .deviceIndex = 0UL,
  }),
  useTimelineSemaphore_(useTimelineSemaphore),
  useTimelineSemaphoreAndSynchronization2_(useTimelineSemaphore && useSynchronization2),
  chainSubmitsWithSemaphores_(chainSubmitsWithSemaphores) {
  IGL_PROFILER_FUNCTION();

  vf_.vkGetDeviceQueue(device_, queueFamilyIndex, 0, &queue_);

  VK_ASSERT(ivkCreateCommandPool(&vf_,
                                 device_,
                                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                                     VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                 queueFamilyIndex,
                                 &commandPool_));

  ivkSetDebugObjectName(&vf_,
                        device,
                        VK_OBJECT_TYPE_COMMAND_POOL,
                        (uint64_t)commandPool_,
                        IGL_FORMAT("Command Pool: {}", debugName).c_str());

  buffers_.reserve(kMaxCommandBuffers);

  for (uint32_t i = 0; i != kMaxCommandBuffers; i++) {
    buffers_.emplace_back(
        VulkanFence(vf_,
                    device_,
                    VkFenceCreateFlagBits{},
                    exportableFences,
                    IGL_FORMAT("Fence: commandBuffer #{}", i).c_str()),
        VulkanSemaphore(
            vf_, device_, false, IGL_FORMAT("Semaphore: {} ({})", debugName, i).c_str()));
    VK_ASSERT(ivkAllocateCommandBuffer(&vf_, device_, commandPool_, &buffers_[i].cmdBufAllocated));
    ivkSetDebugObjectName(&vf_,
                          device_,
                          VK_OBJECT_TYPE_COMMAND_BUFFER,
                          (uint64_t)buffers_[i].cmdBufAllocated,
                          IGL_FORMAT("Command Buffer: {} ({})", debugName, i).c_str());
    buffers_[i].handle.bufferIndex = i;
  }
}

VulkanImmediateCommands::~VulkanImmediateCommands() {
  waitAll();

  vf_.vkDestroyCommandPool(device_, commandPool_, nullptr);
}

void VulkanImmediateCommands::purge() {
  IGL_PROFILER_FUNCTION();

  for (auto& buf : buffers_) {
    if (buf.cmdBuf == VK_NULL_HANDLE || buf.isEncoding) {
      continue;
    }

    const VkResult result = vf_.vkWaitForFences(device_, 1, &buf.fence.vkFence_, VK_TRUE, 0);

    if (result == VK_SUCCESS) {
      VK_ASSERT(vf_.vkResetCommandBuffer(buf.cmdBuf, VkCommandBufferResetFlags{0}));
      VK_ASSERT(vf_.vkResetFences(device_, 1, &buf.fence.vkFence_));
      buf.cmdBuf = VK_NULL_HANDLE;
      numAvailableCommandBuffers_++;
    } else if (result != VK_TIMEOUT && !checkDeviceLost(result)) {
      VK_ASSERT(result);
    }
  }
}

const VulkanImmediateCommands::CommandBufferWrapper& VulkanImmediateCommands::acquire() {
  IGL_PROFILER_FUNCTION();
  if (!numAvailableCommandBuffers_) {
    purge();
  }

  while (!numAvailableCommandBuffers_) {
    IGL_LOG_INFO("Waiting for command buffers...\n");
    IGL_PROFILER_ZONE("Waiting for command buffers...", IGL_PROFILER_COLOR_WAIT);
    purge();
    IGL_PROFILER_ZONE_END();
  }

  VulkanImmediateCommands::CommandBufferWrapper* current = nullptr;

  // we are ok with any available buffer
  for (auto& buf : buffers_) {
    if (buf.cmdBuf == VK_NULL_HANDLE) {
      current = &buf;
      break;
    }
  }

  // make clang happy
  assert(current);

  IGL_DEBUG_ASSERT(numAvailableCommandBuffers_, "No available command buffers");
  IGL_DEBUG_ASSERT(current, "No available command buffers");
  IGL_DEBUG_ASSERT(current->cmdBufAllocated != VK_NULL_HANDLE);

  current->handle.submitId = submitCounter_;
  numAvailableCommandBuffers_--;

  current->cmdBuf = current->cmdBufAllocated;
  current->isEncoding = true;
  current->fd = -1;

  const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  VK_ASSERT(vf_.vkBeginCommandBuffer(current->cmdBuf, &bi));

  if (!chainSubmitsWithSemaphores_) {
    // Orders this command buffer after all work submitted earlier to the queue, which is what the
    // submit-semaphore chain provides otherwise.
    const VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    vf_.vkCmdPipelineBarrier(current->cmdBuf,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);
  }

  nextSubmitHandle_ = current->handle;
  return *current;
}

VkResult VulkanImmediateCommands::wait(const SubmitHandle handle, uint64_t timeoutNanoseconds) {
  if (isReady(handle)) {
    return VK_SUCCESS;
  }

  if (!IGL_DEBUG_VERIFY(!buffers_[handle.bufferIndex].isEncoding)) {
    // we are waiting for a buffer which has not been submitted - this is probably a logic error
    // somewhere in the calling code
    return VK_ERROR_UNKNOWN;
  }

  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  const VkResult fenceResult = vf_.vkWaitForFences(
      device_, 1, &buffers_[handle.bufferIndex].fence.vkFence_, VK_TRUE, timeoutNanoseconds);

  if (fenceResult == VK_TIMEOUT) {
    return VK_TIMEOUT;
  }

  if (fenceResult != VK_SUCCESS) {
    checkDeviceLost(fenceResult);
    IGL_LOG_ERROR_ONCE(
        "VulkanImmediateCommands::wait - Waiting for command buffer fence failed with error %i",
        static_cast<int>(fenceResult));
    // Intentional fallthrough: we must purge so that we can release command buffers.
  }

  purge();

  return fenceResult;
}

void VulkanImmediateCommands::waitAll() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  VkFence fences[kMaxCommandBuffers] = {};

  uint32_t numFences = 0;

  for (const auto& buf : buffers_) {
    if (buf.cmdBuf != VK_NULL_HANDLE && !buf.isEncoding) {
      fences[numFences++] = buf.fence.vkFence_;
    }
  }

  if (numFences) {
    const VkResult result = vf_.vkWaitForFences(device_, numFences, fences, VK_TRUE, UINT64_MAX);
    if (!checkDeviceLost(result)) {
      VK_ASSERT(result);
    }
  }

  purge();
}

bool VulkanImmediateCommands::isRecycled(SubmitHandle handle) const {
  IGL_DEBUG_ASSERT(handle.bufferIndex < kMaxCommandBuffers);

  if (handle.empty()) {
    // a null handle
    return true;
  }

  // already recycled and reused by another command buffer
  return buffers_[handle.bufferIndex].handle.submitId != handle.submitId;
}

bool VulkanImmediateCommands::isReady(const SubmitHandle handle) const {
  IGL_DEBUG_ASSERT(handle.bufferIndex < kMaxCommandBuffers);

  if (handle.empty()) {
    // a null handle
    return true;
  }

  const CommandBufferWrapper& buf = buffers_[handle.bufferIndex];

  if (buf.cmdBuf == VK_NULL_HANDLE) {
    // already recycled and not yet reused
    return true;
  }

  if (buf.handle.submitId != handle.submitId) {
    // already recycled and reused by another command buffer
    return true;
  }

  const VkResult result = vf_.vkWaitForFences(device_, 1, &buf.fence.vkFence_, VK_TRUE, 0);
  checkDeviceLost(result);
  return result == VK_SUCCESS;
}

bool VulkanImmediateCommands::checkDeviceLost(VkResult result) const noexcept {
  if (result != VK_ERROR_DEVICE_LOST) {
    return false;
  }
  if (!deviceLost_.exchange(true)) {
    IGL_LOG_ERROR("VulkanImmediateCommands: the Vulkan device was lost (%s)\n", debugName_.c_str());
  }
  return true;
}

VulkanImmediateCommands::SubmitHandle VulkanImmediateCommands::submit(
    const CommandBufferWrapper& wrapper) {
  return submit(wrapper, VK_NULL_HANDLE);
}

VulkanImmediateCommands::SubmitHandle VulkanImmediateCommands::submit(
    const CommandBufferWrapper& wrapper,
    VkSemaphore presentationSemaphore) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_SUBMIT);

  IGL_DEBUG_ASSERT(wrapper.isEncoding);
  VK_ASSERT(vf_.vkEndCommandBuffer(wrapper.cmdBuf));

  // The submission chain and presentation signals have capacity separate from injected signals.
  constexpr size_t kSubmitSemaphoreCapacity = kMaxInjectedSemaphores + 2;
  // Without the full chain, a submission that waits on injected semaphores still signals its chain
  // semaphore so that the next submission is ordered after those waits.
  const bool signalChainSemaphore = chainSubmitsWithSemaphores_ || numWaitSemaphores_ != 0;
  if (useTimelineSemaphoreAndSynchronization2_) {
    // @lint-ignore CLANGTIDY
    VkSemaphoreSubmitInfo waitSemaphores[kSubmitSemaphoreCapacity] = {};
    for (uint32_t i = 0; i < numWaitSemaphores_; ++i) {
      waitSemaphores[i] = waitSemaphores_[i];
    }
    uint32_t numWaitSemaphores = numWaitSemaphores_;
    if (lastSubmitSemaphore_.semaphore) {
      waitSemaphores[numWaitSemaphores++] = lastSubmitSemaphore_;
    }
    // @lint-ignore CLANGTIDY
    VkSemaphoreSubmitInfo signalSemaphores[kSubmitSemaphoreCapacity] = {};
    for (uint32_t i = 0; i < numSignalSemaphores_; ++i) {
      signalSemaphores[i] = signalSemaphores_[i];
    }
    uint32_t numSignalSemaphores = numSignalSemaphores_;
    if (signalChainSemaphore) {
      signalSemaphores[numSignalSemaphores++] = VkSemaphoreSubmitInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = wrapper.semaphore.getVkSemaphore(),
          .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      };
    }
    if (presentationSemaphore != VK_NULL_HANDLE) {
      signalSemaphores[numSignalSemaphores++] = VkSemaphoreSubmitInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = presentationSemaphore,
          .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      };
    }

    const VkCommandBufferSubmitInfo bufferSI = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = wrapper.cmdBuf,
    };
    const VkSubmitInfo2 si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = numWaitSemaphores,
        .pWaitSemaphoreInfos = waitSemaphores,
        .commandBufferInfoCount = 1u,
        .pCommandBufferInfos = &bufferSI,
        .signalSemaphoreInfoCount = numSignalSemaphores,
        .pSignalSemaphoreInfos = signalSemaphores,
    };

    IGL_PROFILER_ZONE("vkQueueSubmit2KHR()", IGL_PROFILER_COLOR_SUBMIT);
#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkQueueSubmit2KHR()\n\n", wrapper.cmdBuf);
#endif // IGL_VULKAN_PRINT_COMMANDS
    const VkResult result = vf_.vkQueueSubmit2KHR(queue_, 1u, &si, wrapper.fence.vkFence_);
    if (!checkDeviceLost(result)) {
      VK_ASSERT(result);
    }
    IGL_PROFILER_ZONE_END();
  } else {
    // @lint-ignore CLANGTIDY
    VkPipelineStageFlags waitStageMasks[kSubmitSemaphoreCapacity] = {};
    // @lint-ignore CLANGTIDY
    VkSemaphore waitSemaphores[kSubmitSemaphoreCapacity] = {};
    std::array<uint64_t, kSubmitSemaphoreCapacity> waitValues = {};
    std::array<uint64_t, kSubmitSemaphoreCapacity> signalValues = {};
    uint32_t numWaitSemaphores = numWaitSemaphores_;
    for (uint32_t i = 0; i < numWaitSemaphores_; ++i) {
      waitStageMasks[i] = static_cast<VkPipelineStageFlags>(waitSemaphores_[i].stageMask);
      waitSemaphores[i] = waitSemaphores_[i].semaphore;
      waitValues[i] = waitSemaphores_[i].value;
    }
    if (lastSubmitSemaphore_.semaphore) {
      waitStageMasks[numWaitSemaphores] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      waitSemaphores[numWaitSemaphores++] = lastSubmitSemaphore_.semaphore;
    }

    // @lint-ignore CLANGTIDY
    VkSemaphore signalSemaphores[kSubmitSemaphoreCapacity] = {};
    for (uint32_t i = 0; i < numSignalSemaphores_; ++i) {
      signalSemaphores[i] = signalSemaphores_[i].semaphore;
      signalValues[i] = signalSemaphores_[i].value;
    }
    uint32_t numSignalSemaphores = numSignalSemaphores_;
    if (signalChainSemaphore) {
      signalSemaphores[numSignalSemaphores++] = wrapper.semaphore.vkSemaphore_;
    }
    if (presentationSemaphore != VK_NULL_HANDLE) {
      signalSemaphores[numSignalSemaphores++] = presentationSemaphore;
    }
    // Values align with all semaphore entries, including zero-valued binary chain/presentation.
    const VkTimelineSemaphoreSubmitInfo timelineInfo = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = numWaitSemaphores,
        .pWaitSemaphoreValues = waitValues.data(),
        .signalSemaphoreValueCount = numSignalSemaphores,
        .pSignalSemaphoreValues = signalValues.data(),
    };
    const VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = useTimelineSemaphore_ ? &timelineInfo : nullptr,
        .waitSemaphoreCount = numWaitSemaphores,
        .pWaitSemaphores = numWaitSemaphores ? waitSemaphores : nullptr,
        .pWaitDstStageMask = waitStageMasks,
        .commandBufferCount = 1,
        .pCommandBuffers = &wrapper.cmdBuf,
        .signalSemaphoreCount = numSignalSemaphores,
        .pSignalSemaphores = signalSemaphores,
    };
    // @lint-ignore CLANGTIDY
    const VkFence vkFence = wrapper.fence.vkFence_;
    IGL_PROFILER_ZONE("vkQueueSubmit()", IGL_PROFILER_COLOR_SUBMIT);
#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkQueueSubmit()\n\n", wrapper.cmdBuf);
#endif // IGL_VULKAN_PRINT_COMMANDS
    const VkResult result = vf_.vkQueueSubmit(queue_, 1u, &si, vkFence);
    if (!checkDeviceLost(result)) {
      VK_ASSERT(result);
    }
    IGL_PROFILER_ZONE_END();
  }

  lastSubmitSemaphore_.semaphore = signalChainSemaphore ? wrapper.semaphore.vkSemaphore_
                                                        : VK_NULL_HANDLE;
  lastSubmitHandle_ = wrapper.handle;
  numWaitSemaphores_ = 0;
  numSignalSemaphores_ = 0;

  // reset
  const_cast<CommandBufferWrapper&>(wrapper).isEncoding = false;
  submitCounter_++;

  if (!submitCounter_) {
    // skip the 0 value - when uint32_t wraps around (null SubmitHandle)
    submitCounter_++;
  }

  nextSubmitHandle_ = {};

  return lastSubmitHandle_;
}

void VulkanImmediateCommands::discard(const CommandBufferWrapper& wrapper) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_SUBMIT);
  IGL_DEBUG_ASSERT(wrapper.isEncoding);
  if (!IGL_DEBUG_VERIFY(wrapper.handle == nextSubmitHandle_,
                        "Only the current command buffer can be discarded")) {
    return;
  }
  auto& mutableWrapper = const_cast<CommandBufferWrapper&>(wrapper);
  VK_ASSERT(vf_.vkResetCommandBuffer(mutableWrapper.cmdBuf, VkCommandBufferResetFlags{0}));
  mutableWrapper.cmdBuf = VK_NULL_HANDLE;
  mutableWrapper.isEncoding = false;
  numAvailableCommandBuffers_++;
  numWaitSemaphores_ = 0;
  numSignalSemaphores_ = 0;
  nextSubmitHandle_ = {};
}

bool VulkanImmediateCommands::waitSemaphore(VkSemaphore semaphore, uint64_t waitValue) {
  return waitSemaphore(semaphore, waitValue, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}

bool VulkanImmediateCommands::waitSemaphore(VkSemaphore semaphore,
                                            uint64_t waitValue,
                                            VkPipelineStageFlags waitStageMask) {
  if (numWaitSemaphores_ >= kMaxInjectedSemaphores) {
    IGL_LOG_ERROR("VulkanImmediateCommands::waitSemaphore(): too many injected semaphores\n");
    return false;
  }
  if (waitStageMask == 0) {
    IGL_LOG_ERROR("VulkanImmediateCommands::waitSemaphore(): destination stage mask is empty\n");
    return false;
  }
  waitSemaphores_[numWaitSemaphores_++] = VkSemaphoreSubmitInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = semaphore,
      .value = waitValue,
      .stageMask = waitStageMask,
  };
  return true;
}

bool VulkanImmediateCommands::signalSemaphore(VkSemaphore semaphore, uint64_t signalValue) {
  if (numSignalSemaphores_ >= kMaxInjectedSemaphores) {
    IGL_LOG_ERROR("VulkanImmediateCommands::signalSemaphore(): too many injected semaphores\n");
    return false;
  }
  signalSemaphores_[numSignalSemaphores_++] = VkSemaphoreSubmitInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = semaphore,
      .value = signalValue,
      .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
  };
  return true;
}

VulkanImmediateCommands::SubmitHandle VulkanImmediateCommands::getLastSubmitHandle() const {
  return lastSubmitHandle_;
}

bool VulkanImmediateCommands::isAnyCommandBufferEncoding() const noexcept {
  return std::any_of(buffers_.begin(), buffers_.end(), [](const CommandBufferWrapper& buf) {
    return buf.isEncoding;
  });
}

VulkanImmediateCommands::SubmitHandle VulkanImmediateCommands::getNextSubmitHandle() const {
  return nextSubmitHandle_.empty() ? lastSubmitHandle_ : nextSubmitHandle_;
}

// @fb-only
VkFence VulkanImmediateCommands::getVkFenceFromSubmitHandle(SubmitHandle handle) {
  IGL_DEBUG_ASSERT(handle.bufferIndex < buffers_.size());

  if (isRecycled(handle)) {
    return VK_NULL_HANDLE;
  }

  return buffers_[handle.bufferIndex].fence.vkFence_;
}

void VulkanImmediateCommands::storeFDInSubmitHandle(SubmitHandle handle, int fd) noexcept {
  IGL_DEBUG_ASSERT(handle.bufferIndex < buffers_.size());
  buffers_[handle.bufferIndex].fd = fd;
}

int VulkanImmediateCommands::cachedFDFromSubmitHandle(SubmitHandle handle) const noexcept {
  IGL_DEBUG_ASSERT(handle.bufferIndex < buffers_.size());
  return buffers_[handle.bufferIndex].fd;
}

} // namespace igl::vulkan
