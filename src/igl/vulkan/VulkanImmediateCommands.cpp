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

namespace {

/// The submit id of an acquisition ordinal.
[[nodiscard]] uint32_t lowBits(uint64_t ordinal) noexcept {
  return static_cast<uint32_t>(ordinal & 0xffffffffu);
}

} // namespace

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

  initResult_ = ivkCreateCommandPool(&vf_,
                                     device_,
                                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                                         VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                     queueFamilyIndex,
                                     &commandPool_);
  if (initResult_ != VK_SUCCESS) {
    // The handle a failed `vkCreateCommandPool()` writes is undefined: the destructor must not use
    // it.
    commandPool_ = VK_NULL_HANDLE;
    IGL_LOG_ERROR("VulkanImmediateCommands (%s): cannot create the command pool (VkResult %d)\n",
                  debugName,
                  static_cast<int>(initResult_));
    return;
  }

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
    // A row that cannot be created ends the construction: the rows made so far, and the pool, are
    // destroyed by the destructor, and `valid()` is false.
    initResult_ = buffers_[i].fence.initResult();
    if (initResult_ == VK_SUCCESS) {
      initResult_ = buffers_[i].semaphore.initResult();
    }
    if (initResult_ == VK_SUCCESS) {
      initResult_ =
          ivkAllocateCommandBuffer(&vf_, device_, commandPool_, &buffers_[i].cmdBufAllocated);
    }
    if (initResult_ != VK_SUCCESS) {
      IGL_LOG_ERROR(
          "VulkanImmediateCommands (%s): cannot create command buffer row %u (VkResult %d)\n",
          debugName,
          i,
          static_cast<int>(initResult_));
      return;
    }
    ivkSetDebugObjectName(&vf_,
                          device_,
                          VK_OBJECT_TYPE_COMMAND_BUFFER,
                          (uint64_t)buffers_[i].cmdBufAllocated,
                          IGL_FORMAT("Command Buffer: {} ({})", debugName, i).c_str());
    buffers_[i].handle.bufferIndex = i;
  }
  IGL_DEBUG_ASSERT(buffers_.size() == kMaxCommandBuffers);
}

VulkanImmediateCommands::~VulkanImmediateCommands() {
  waitAll();

  vf_.vkDestroyCommandPool(device_, commandPool_, nullptr);
}

void VulkanImmediateCommands::purge() {
  IGL_PROFILER_FUNCTION();

  for (auto& buf : buffers_) {
    IGL_DEBUG_ASSERT(rowAgreesWithWrapper(buf));
    if (buf.state != RowState::Submitted) {
      continue;
    }

    const VkResult result = vf_.vkWaitForFences(device_, 1, &buf.fence.vkFence_, VK_TRUE, 0);

    if (result == VK_SUCCESS) {
      certify(buf.retirement_.prefix);
      VK_ASSERT(vf_.vkResetCommandBuffer(buf.cmdBuf, VkCommandBufferResetFlags{0}));
      VK_ASSERT(vf_.vkResetFences(device_, 1, &buf.fence.vkFence_));
      setRowState(buf, RowState::Free);
      numAvailableCommandBuffers_++;
    } else if (checkDeviceLost(result)) {
      // A lost device runs nothing any more: the submission is finished, and the row is kept.
      certify(buf.retirement_.prefix, true);
    } else if (result != VK_TIMEOUT) {
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
    IGL_DEBUG_ASSERT(rowAgreesWithWrapper(buf));
    if (buf.state == RowState::Free) {
      current = &buf;
      break;
    }
  }

  // make clang happy
  assert(current);

  IGL_DEBUG_ASSERT(numAvailableCommandBuffers_, "No available command buffers");
  IGL_DEBUG_ASSERT(current, "No available command buffers");
  IGL_DEBUG_ASSERT(current->cmdBufAllocated != VK_NULL_HANDLE);

  ++acquisitionCounter_;
  if (lowBits(acquisitionCounter_) == 0) {
    // skip the ordinals whose submit id would be 0 (a null SubmitHandle)
    ++acquisitionCounter_;
  }
  RowRetirement& row = current->retirement_;
  // A recording closed without a submission, with no `submit()` call from its acquisition until the
  // row is taken again, had the same handle as the recording that takes the row: keep answering for
  // it as that one. One that saw a `submit()` call while it was open had another handle, and
  // answers for its own prefix only.
  const bool aliasesPrevious = row.acquisition != 0 && row.closedUnsubmitted &&
                               row.submitAttemptsAtAcquisition == submitAttempts_;
  row = RowRetirement{
      .acquisition = acquisitionCounter_,
      .submitAttemptsAtAcquisition = submitAttempts_,
      .prefix = 0,
      .previousPrefix = row.acquisition != 0 ? row.prefix : 0,
      .aliasFloor = aliasesPrevious ? row.aliasFloor : acquisitionCounter_,
  };
  current->handle.submitId = lowBits(acquisitionCounter_);
  numAvailableCommandBuffers_--;

  setRowState(*current, RowState::Encoding);
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
  const Ticket ticket = ticketOf(handle);
  if (ticket.kind == Ticket::Kind::Settled) {
    return VK_SUCCESS;
  }
  if (ticket.kind == Ticket::Kind::Prefix) {
    // The zero-time query first, as `isReady()` does.
    pollFor(ticket);
    if (retiredThrough_ >= ticket.prefix) {
      return certifiedResult(ticket.prefix);
    }
  }

  if (!IGL_DEBUG_VERIFY(ticket.kind != Ticket::Kind::Open)) {
    // we are waiting for a buffer which has not been submitted - this is probably a logic error
    // somewhere in the calling code
    return VK_ERROR_UNKNOWN;
  }

  if (timeoutNanoseconds == 0) {
    // A zero-time wait only polls, which `isReady()` did.
    return VK_TIMEOUT;
  }

  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  // The submission that the ticket waits for is still in its row (a row is freed only once its
  // fence signaled), so some row fence covers it.
  uint64_t ordinal = 0;
  const VkFence* fence = lowestSourceFor(ticket.prefix, ticket.own, ordinal);
  if (fence == nullptr) {
    IGL_DEBUG_ABORT("No fence covers a submission that has not finished");
    return VK_ERROR_UNKNOWN;
  }
  const VkResult result = waitFence(fence, ordinal, timeoutNanoseconds);

  if (result == VK_TIMEOUT) {
    return VK_TIMEOUT;
  }

  if (result != VK_SUCCESS) {
    IGL_LOG_ERROR_ONCE(
        "VulkanImmediateCommands::wait - Waiting for command buffer fence failed with error %i",
        static_cast<int>(result));
    // Intentional fallthrough: we must purge so that we can release command buffers.
  }

  purge();

  return result;
}

void VulkanImmediateCommands::waitAll() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  VkFence fences[kMaxCommandBuffers] = {};

  uint32_t numFences = 0;

  for (const auto& buf : buffers_) {
    IGL_DEBUG_ASSERT(rowAgreesWithWrapper(buf));
    if (buf.state == RowState::Submitted) {
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
  return buffers_[handle.bufferIndex].handle.submitId != handle.submitId &&
         !answersAsCurrentRecording(handle);
}

bool VulkanImmediateCommands::answersAsCurrentRecording(SubmitHandle handle) const noexcept {
  const uint32_t currentId = buffers_[handle.bufferIndex].handle.submitId;
  const uint32_t floorId = lowBits(buffers_[handle.bufferIndex].retirement_.aliasFloor);
  // Submit ids skip 0 and order acquisitions modulo 2^32, so `handle` lies in
  // [aliasFloor, current) exactly when its distance from the floor is the smaller.
  return handle.submitId != 0 && floorId != currentId &&
         static_cast<uint32_t>(handle.submitId - floorId) <
             static_cast<uint32_t>(currentId - floorId);
}

VulkanImmediateCommands::Ticket VulkanImmediateCommands::ticketOfRow(
    const CommandBufferWrapper& wrapper) const noexcept {
  if (wrapper.state == RowState::Encoding) {
    return {.kind = Ticket::Kind::Open};
  }
  return {.kind = Ticket::Kind::Prefix, .prefix = wrapper.retirement_.prefix, .own = &wrapper};
}

VulkanImmediateCommands::Ticket VulkanImmediateCommands::ticketOf(
    SubmitHandle handle) const noexcept {
  IGL_DEBUG_ASSERT(handle.bufferIndex < kMaxCommandBuffers);

  if (handle.empty()) {
    // a null handle
    return {.kind = Ticket::Kind::Settled};
  }

  const CommandBufferWrapper& buf = buffers_[handle.bufferIndex];
  IGL_DEBUG_ASSERT(rowAgreesWithWrapper(buf));
  if (buf.handle.submitId == handle.submitId || answersAsCurrentRecording(handle)) {
    return ticketOfRow(buf);
  }
  // An older recording of this row: it closed before the row was acquired again, so its prefix is
  // at most the previous recording's.
  return {.kind = Ticket::Kind::Prefix, .prefix = buf.retirement_.previousPrefix};
}

const VkFence* IGL_NULLABLE
VulkanImmediateCommands::lowestSourceFor(uint64_t target,
                                         const CommandBufferWrapper* IGL_NULLABLE own,
                                         uint64_t& ordinal) const noexcept {
  if (own && own->state == RowState::Submitted && own->retirement_.prefix == target) {
    // The common case: the handle's own submission, whose fence no other source precedes.
    ordinal = target;
    return &own->fence.vkFence_;
  }
  const VkFence* fence = nullptr;
  ordinal = UINT64_MAX;
  for (const auto& buf : buffers_) {
    if (buf.state == RowState::Submitted && buf.retirement_.prefix >= target &&
        buf.retirement_.prefix < ordinal) {
      ordinal = buf.retirement_.prefix;
      fence = &buf.fence.vkFence_;
    }
  }
  return fence;
}

void VulkanImmediateCommands::certify(uint64_t ordinal, bool lost) const noexcept {
  retiredThrough_ = std::max(retiredThrough_, ordinal);
  if (!lost) {
    succeededThrough_ = std::max(succeededThrough_, ordinal);
  }
}

VkResult VulkanImmediateCommands::certifiedResult(uint64_t target) const noexcept {
  return target <= succeededThrough_ ? VK_SUCCESS : VK_ERROR_DEVICE_LOST;
}

void VulkanImmediateCommands::pollFor(const Ticket& ticket) const {
  if (retiredThrough_ >= ticket.prefix) {
    return;
  }
  uint64_t ordinal = 0;
  const VkFence* fence = lowestSourceFor(ticket.prefix, ticket.own, ordinal);
  if (!fence) {
    return;
  }
  const VkResult result = vf_.vkWaitForFences(device_, 1, fence, VK_TRUE, 0);
  if (result == VK_SUCCESS || checkDeviceLost(result)) {
    certify(ordinal, result != VK_SUCCESS);
  }
}

VkResult VulkanImmediateCommands::waitFence(const VkFence* fence,
                                            uint64_t ordinal,
                                            uint64_t timeoutNanoseconds) const {
  const VkResult result = vf_.vkWaitForFences(device_, 1, fence, VK_TRUE, timeoutNanoseconds);
  if (result == VK_SUCCESS || checkDeviceLost(result)) {
    certify(ordinal, result != VK_SUCCESS);
  }
  return result;
}

bool VulkanImmediateCommands::isReady(const SubmitHandle handle) const {
  const Ticket ticket = ticketOf(handle);
  if (ticket.kind == Ticket::Kind::Settled) {
    return true;
  }
  if (ticket.kind == Ticket::Kind::Open) {
    // a recording that is still open has not finished; its fence is not polled
    return false;
  }
  pollFor(ticket);
  return retiredThrough_ >= ticket.prefix;
}

void VulkanImmediateCommands::setRowState(CommandBufferWrapper& wrapper, RowState state) noexcept {
  if (state == RowState::Submitted && wrapper.state == RowState::Free) {
    // A stale wrapper (its row was freed): submitting it leaves the row free, and its fields, which
    // already say so, are not touched.
    return;
  }
  wrapper.state = state;
  switch (state) {
  case RowState::Free:
    wrapper.cmdBuf = VK_NULL_HANDLE;
    wrapper.isEncoding = false;
    break;
  case RowState::Encoding:
    // A pool whose construction failed is never used, so every row owns a command buffer.
    IGL_DEBUG_ASSERT(wrapper.cmdBufAllocated != VK_NULL_HANDLE);
    wrapper.cmdBuf = wrapper.cmdBufAllocated;
    wrapper.isEncoding = true;
    break;
  case RowState::Submitted:
    wrapper.isEncoding = false;
    break;
  }
}

bool VulkanImmediateCommands::rowAgreesWithWrapper(const CommandBufferWrapper& wrapper) noexcept {
  return (wrapper.state == RowState::Encoding) == wrapper.isEncoding &&
         (wrapper.state == RowState::Free) == (wrapper.cmdBuf == VK_NULL_HANDLE);
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
    VkSemaphore presentationSemaphore,
    VkCommandBuffer nextCmdBuf) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_SUBMIT);

  ++submitAttempts_;
  IGL_DEBUG_ASSERT(wrapper.isEncoding);
  // A stale wrapper (its row was freed) is a misuse that takes no submission ordinal.
  const bool staleWrapper = wrapper.state == RowState::Free;
  VK_ASSERT(vf_.vkEndCommandBuffer(wrapper.cmdBuf));

  const std::array<VkCommandBuffer, 2> cmdBufs = {wrapper.cmdBuf, nextCmdBuf};
  const uint32_t numCmdBufs = nextCmdBuf != VK_NULL_HANDLE ? 2u : 1u;

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

    const std::array<VkCommandBufferSubmitInfo, 2> bufferSIs = {{
        {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmdBufs[0]},
        {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmdBufs[1]},
    }};
    const VkSubmitInfo2 si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = numWaitSemaphores,
        .pWaitSemaphoreInfos = waitSemaphores,
        .commandBufferInfoCount = numCmdBufs,
        .pCommandBufferInfos = bufferSIs.data(),
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
        .commandBufferCount = numCmdBufs,
        .pCommandBuffers = cmdBufs.data(),
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

  if (!staleWrapper) {
    const_cast<CommandBufferWrapper&>(wrapper).retirement_.prefix = ++submissionCounter_;
  }

  // reset
  setRowState(const_cast<CommandBufferWrapper&>(wrapper), RowState::Submitted);

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
  closeUnsubmitted(const_cast<CommandBufferWrapper&>(wrapper));
}

void VulkanImmediateCommands::freeUnsubmittedRow(CommandBufferWrapper& wrapper) {
  VK_ASSERT(vf_.vkResetCommandBuffer(wrapper.cmdBuf, VkCommandBufferResetFlags{0}));
  RowRetirement& row = wrapper.retirement_;
  row.prefix = submissionCounter_;
  row.closedUnsubmitted = true;
  setRowState(wrapper, RowState::Free);
  numAvailableCommandBuffers_++;
}

void VulkanImmediateCommands::closeUnsubmitted(CommandBufferWrapper& wrapper) {
  freeUnsubmittedRow(wrapper);
  numWaitSemaphores_ = 0;
  numSignalSemaphores_ = 0;
  nextSubmitHandle_ = {};
}

void VulkanImmediateCommands::release(const CommandBufferWrapper& wrapper) {
  IGL_PROFILER_FUNCTION();
  if (!IGL_DEBUG_VERIFY(wrapper.isEncoding)) {
    return;
  }
  // As a discarded recording, it waits for everything submitted before it.
  freeUnsubmittedRow(const_cast<CommandBufferWrapper&>(wrapper));
  if (nextSubmitHandle_ == wrapper.handle) {
    nextSubmitHandle_ = {};
  }
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
