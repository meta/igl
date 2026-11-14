/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanImmediateCommands.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <type_traits>
#include <igl/vulkan/Common.h>

namespace igl::vulkan {

static_assert(std::is_trivially_copyable_v<VulkanImmediateCommands::SubmitHandle>);

namespace {

/// The submit id of an acquisition ordinal.
[[nodiscard]] uint32_t lowBits(uint64_t ordinal) noexcept {
  return static_cast<uint32_t>(ordinal & 0xffffffffu);
}

[[nodiscard]] bool isOutOfMemory(VkResult result) noexcept {
  return result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

/// Waits on `fences` for at most `remainingNanoseconds` (`UINT64_MAX` waits without a limit) and
/// subtracts the time the wait took, so that one budget covers every wait of a call.
VkResult waitForFences(const VulkanFunctionTable& vf,
                       VkDevice device,
                       uint32_t count,
                       const VkFence* fences,
                       uint64_t& remainingNanoseconds) {
  if (remainingNanoseconds == UINT64_MAX || remainingNanoseconds == 0) {
    return vf.vkWaitForFences(device, count, fences, VK_TRUE, remainingNanoseconds);
  }
  const auto start = std::chrono::steady_clock::now();
  const VkResult result = vf.vkWaitForFences(device, count, fences, VK_TRUE, remainingNanoseconds);
  const auto elapsed = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
          .count());
  remainingNanoseconds = elapsed >= remainingNanoseconds ? 0 : remainingNanoseconds - elapsed;
  return result;
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
  checkpoints_.reserve(kMaxCheckpoints);
}

VulkanImmediateCommands::~VulkanImmediateCommands() {
  // A recording that was not submitted uses nothing on the GPU: only submitted work is waited for.
  drainSubmitted();

  // A checkpoint fence cannot be destroyed while its signal is pending.
  for (Checkpoint& checkpoint : checkpoints_) {
    if (checkpoint.state == Checkpoint::State::Pending) {
      const VkResult result =
          vf_.vkWaitForFences(device_, 1, &checkpoint.fence.vkFence_, VK_TRUE, UINT64_MAX);
      if (!checkDeviceLost(result)) {
        VK_ASSERT(result);
      }
    }
  }

  vf_.vkDestroyCommandPool(device_, commandPool_, nullptr);
}

void VulkanImmediateCommands::purge() {
  IGL_PROFILER_FUNCTION();

  pollPendingCheckpoints();
  // The fence of an exported or unknown submission is never polled; a later answer covers it.
  for (auto& buf : buffers_) {
    IGL_DEBUG_ASSERT(rowAgreesWithWrapper(buf));
    const RowRetirement& row = buf.retirement_;
    if (buf.state != RowState::Submitted || row.quarantined || !row.fenceTrusted ||
        retiredThrough_ >= row.prefix) {
      continue;
    }
    const VkResult result = vf_.vkWaitForFences(device_, 1, &buf.fence.vkFence_, VK_TRUE, 0);
    if (result == VK_SUCCESS) {
      certify(row.prefix);
    } else if (checkDeviceLost(result)) {
      certifyLoss();
    } else if (result != VK_TIMEOUT) {
      VK_ASSERT(result);
    }
  }
  for (auto& buf : buffers_) {
    if (buf.state == RowState::Submitted && !buf.retirement_.quarantined &&
        retiredThrough_ >= buf.retirement_.prefix) {
      static_cast<void>(recycleCertifiedRow(buf));
    }
  }
}

void VulkanImmediateCommands::pollPendingCheckpoints() const {
  for (Checkpoint& checkpoint : checkpoints_) {
    if (checkpoint.state != Checkpoint::State::Pending) {
      continue;
    }
    const VkResult result = vf_.vkWaitForFences(device_, 1, &checkpoint.fence.vkFence_, VK_TRUE, 0);
    if (result == VK_SUCCESS || checkDeviceLost(result)) {
      noteFenceSignaled(&checkpoint.fence.vkFence_, checkpoint.covers, result != VK_SUCCESS);
    } else if (result != VK_TIMEOUT) {
      VK_ASSERT(result);
    }
  }
}

VkResult VulkanImmediateCommands::recycleCertifiedRow(CommandBufferWrapper& wrapper) {
  // Its submission retired: one certified successful keeps that answer once the row is reused.
  if (wrapper.retirement_.prefix <= succeededThrough_) {
    wrapper.succeededAcquisition_ = wrapper.retirement_.acquisition;
  }
  const VkResult commandBufferResult =
      vf_.vkResetCommandBuffer(wrapper.cmdBuf, VkCommandBufferResetFlags{0});
  VK_ASSERT(commandBufferResult);
  if (commandBufferResult != VK_SUCCESS) {
    // The command buffer cannot be recorded again until a reset succeeds.
    wrapper.retirement_.quarantined = true;
    IGL_LOG_ERROR_ONCE(
        "VulkanImmediateCommands (%s): a command buffer could not be reset (VkResult %d)\n",
        debugName_.c_str(),
        static_cast<int>(commandBufferResult));
    return commandBufferResult;
  }
  const VkResult result = vf_.vkResetFences(device_, 1, &wrapper.fence.vkFence_);
  VK_ASSERT(result);
  if (result != VK_SUCCESS) {
    // A reported device loss is remembered; it proves nothing about other submissions.
    static_cast<void>(checkDeviceLost(result));
    // The fence may still be signaled: keep the row out of circulation until its fence is reset or
    // replaced.
    wrapper.retirement_.quarantined = true;
    IGL_LOG_ERROR_ONCE("VulkanImmediateCommands (%s): a fence could not be reset (VkResult %d)\n",
                       debugName_.c_str(),
                       static_cast<int>(result));
    return result;
  }
  setRowState(wrapper, RowState::Free);
  numAvailableCommandBuffers_++;
  return VK_SUCCESS;
}

bool VulkanImmediateCommands::recycleOldest() {
  CommandBufferWrapper* oldest = nullptr;
  for (auto& buf : buffers_) {
    if (buf.state == RowState::Submitted && !buf.retirement_.quarantined &&
        (!oldest || buf.retirement_.prefix < oldest->retirement_.prefix)) {
      oldest = &buf;
    }
  }
  if (!oldest) {
    return false;
  }
  const uint32_t availableBefore = numAvailableCommandBuffers_;
  const uint64_t prefix = oldest->retirement_.prefix;
  if (oldest->retirement_.fenceTrusted && retiredThrough_ < prefix) {
    // One wait on the oldest submission's own fence.
    uint64_t remaining = UINT64_MAX;
    const VkResult result = waitForFences(vf_, device_, 1, &oldest->fence.vkFence_, remaining);
    if (result == VK_SUCCESS || checkDeviceLost(result)) {
      // A device loss ends every submission, as lost: the row is reused like a finished one.
      if (result == VK_SUCCESS) {
        certify(prefix);
      } else {
        certifyLoss();
      }
      static_cast<void>(recycleCertifiedRow(*oldest));
    } else if (result != VK_TIMEOUT) {
      VK_ASSERT(result);
    }
    return numAvailableCommandBuffers_ > availableBefore;
  }
  if (retiredThrough_ < prefix) {
    uint64_t remaining = UINT64_MAX;
    static_cast<void>(certifyThrough(prefix, remaining, nullptr));
  }
  if (retiredThrough_ >= prefix) {
    static_cast<void>(recycleCertifiedRow(*oldest));
  }
  return numAvailableCommandBuffers_ > availableBefore;
}

bool VulkanImmediateCommands::recycleDroppedRows() {
  if (!lossCertified_) {
    return false;
  }
  bool freed = false;
  for (auto& buf : buffers_) {
    if (buf.state != RowState::Dropped) {
      continue;
    }
    // Nothing of it was submitted. After a device loss even a fence that was never submitted can
    // answer as signaled, so it is reset too.
    const VkResult commandBufferResult =
        vf_.vkResetCommandBuffer(buf.cmdBuf, VkCommandBufferResetFlags{0});
    VK_ASSERT(commandBufferResult);
    if (commandBufferResult != VK_SUCCESS) {
      continue;
    }
    const VkResult fenceResult = vf_.vkResetFences(device_, 1, &buf.fence.vkFence_);
    VK_ASSERT(fenceResult);
    if (fenceResult != VK_SUCCESS) {
      continue;
    }
    buf.outcomesLost_ = true;
    setRowState(buf, RowState::Free);
    numAvailableCommandBuffers_++;
    freed = true;
  }
  return freed;
}

bool VulkanImmediateCommands::recoverQuarantinedRow() {
  for (auto& buf : buffers_) {
    if (buf.state != RowState::Submitted || !buf.retirement_.quarantined) {
      continue;
    }
    // Its submission finished before the row was quarantined.
    if (vf_.vkResetCommandBuffer(buf.cmdBuf, VkCommandBufferResetFlags{0}) != VK_SUCCESS) {
      continue;
    }
    if (vf_.vkResetFences(device_, 1, &buf.fence.vkFence_) != VK_SUCCESS) {
      // Replace the fence. Nothing pending uses it, and a fence handed out for the old submission
      // (`getVkFenceFromSubmitHandle()`) is valid only until the row is acquired again, which is
      // what follows: rows are recovered only when none is free.
      VulkanFence fence(vf_,
                        device_,
                        VkFenceCreateFlagBits{},
                        buf.fence.exportable(),
                        IGL_FORMAT("Fence: commandBuffer #{}", buf.handle.bufferIndex).c_str());
      if (fence.initResult() != VK_SUCCESS) {
        static_cast<void>(checkDeviceLost(fence.initResult()));
        continue;
      }
      buf.fence = std::move(fence);
    }
    buf.retirement_.quarantined = false;
    setRowState(buf, RowState::Free);
    numAvailableCommandBuffers_++;
    return true;
  }
  return false;
}

bool VulkanImmediateCommands::freeRow() {
  return recoverQuarantinedRow() || recycleDroppedRows() || recycleOldest();
}

VkResult VulkanImmediateCommands::backoff(uint32_t& attempt) {
  constexpr std::array<uint64_t, 5> kDelayNanoseconds = {
      1'000'000, 2'000'000, 4'000'000, 8'000'000, 16'000'000};
  const uint64_t delay = kDelayNanoseconds[attempt];
  if (attempt + 1 < kDelayNanoseconds.size()) {
    ++attempt;
  }
  VkResult result = ensureBackoffFence();
  if (result == VK_SUCCESS && backoffFence_) {
    result = vf_.vkWaitForFences(device_, 1, &backoffFence_->vkFence_, VK_TRUE, delay);
  }
  if (result == VK_TIMEOUT || certifyLossFromUnsignaledFence(result)) {
    return result;
  }
  if (result != VK_SUCCESS) {
    IGL_LOG_ERROR_ONCE(
        "VulkanImmediateCommands (%s): the back-off fence wait failed (VkResult %d)\n",
        debugName_.c_str(),
        static_cast<int>(result));
  }
  // The fence wait failed (or the fence could not be created), or answered at once what proves
  // nothing: wait on the host, so that this does not spin.
  hostBackoffDelay(delay);
  return result;
}

bool VulkanImmediateCommands::certifyLossFromUnsignaledFence(VkResult result) const noexcept {
  // Nothing on a working device signals such a fence. An answer that the device is lost proves that
  // nothing submitted runs any more; so does a success once a loss was already reported.
  if (result == VK_ERROR_DEVICE_LOST || (result == VK_SUCCESS && isDeviceLost())) {
    certifyLoss();
    return true;
  }
  if (result == VK_SUCCESS) {
    IGL_LOG_ERROR_ONCE(
        "VulkanImmediateCommands (%s): a fence that nothing signals answered as signaled, with no "
        "device loss reported\n",
        debugName_.c_str());
  }
  return false;
}

void VulkanImmediateCommands::hostBackoffDelay(uint64_t nanoseconds) {
  std::unique_lock<std::mutex> lock(hostBackoffMutex_);
  hostBackoffWakeup_.wait_for(lock, std::chrono::nanoseconds(nanoseconds), [] { return false; });
}

VkResult VulkanImmediateCommands::ensureBackoffFence() {
  if (!backoffFence_ || backoffFence_->vkFence_ == VK_NULL_HANDLE) {
    backoffFence_.emplace(vf_,
                          device_,
                          VkFenceCreateFlagBits{},
                          false,
                          IGL_FORMAT("Fence: {} back-off", debugName_).c_str());
  }
  return backoffFence_->initResult();
}

const VulkanImmediateCommands::CommandBufferWrapper& VulkanImmediateCommands::acquire() {
  IGL_PROFILER_FUNCTION();
  if (!numAvailableCommandBuffers_) {
    // Free every row whose submission finished, without waiting; wait below only when none did.
    purge();
    if (!numAvailableCommandBuffers_) {
      IGL_LOG_INFO("Waiting for command buffers...\n");
    }
  }

  uint32_t attempt = 0;
  while (!numAvailableCommandBuffers_) {
    IGL_PROFILER_ZONE("Waiting for command buffers...", IGL_PROFILER_COLOR_WAIT);
    // Block, with no time limit, on the oldest submission instead of polling every row in a loop.
    // When no row can be freed now (no checkpoint can be submitted, a fence wait failed, or every
    // busy row is quarantined), wait a little, longer each time, before trying again. A submission
    // whose fence cannot be used is never proven finished while no checkpoint can be submitted:
    // this then waits as long as that lasts, or until the device is lost.
    if (!freeRow()) {
      IGL_LOG_ERROR_ONCE("VulkanImmediateCommands (%s): no command buffer can be freed now\n",
                         debugName_.c_str());
      static_cast<void>(backoff(attempt));
    }
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
  if (ticket.kind == Ticket::Kind::Lost) {
    return VK_ERROR_DEVICE_LOST;
  }
  if (ticket.kind == Ticket::Kind::Dropped) {
    // A dropped ticket names the current recording of the handle's row.
    return waitDropped(buffers_[handle.bufferIndex], timeoutNanoseconds);
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

  uint64_t remaining = timeoutNanoseconds;
  const VkResult result = certifyThrough(ticket.prefix, remaining, ticket.own);

  // The work may have been certified while the time ran out: then that is the answer.
  if (result == VK_TIMEOUT && retiredThrough_ < ticket.prefix) {
    return VK_TIMEOUT;
  }

  if (result != VK_SUCCESS && result != VK_TIMEOUT) {
    IGL_LOG_ERROR_ONCE(
        "VulkanImmediateCommands::wait - Waiting for command buffer fence failed with error %i",
        static_cast<int>(result));
    // Intentional fallthrough: we must purge so that we can release command buffers.
  }

  purge();

  // A success answer after a certified loss does not make the handle's work succeed.
  return retiredThrough_ >= ticket.prefix ? certifiedResult(ticket.prefix) : result;
}

void VulkanImmediateCommands::waitAll() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);
  drainSubmitted();

  // A recording that was not submitted is waited for on its fence, which nothing signals, as the
  // fence of a submission that never reached the queue: only a device loss ends the wait.
  VkFence fences[kMaxCommandBuffers] = {};
  uint32_t numFences = 0;
  for (const auto& buf : buffers_) {
    if (buf.state == RowState::Dropped) {
      fences[numFences++] = buf.fence.vkFence_;
    }
  }
  if (numFences == 0 || lossCertified_) {
    return;
  }
  IGL_LOG_ERROR_ONCE(
      "VulkanImmediateCommands (%s): waitAll() waits for recordings that were not submitted\n",
      debugName_.c_str());
  uint32_t attempt = 0;
  for (;;) {
    const VkResult result = vf_.vkWaitForFences(device_, numFences, fences, VK_TRUE, UINT64_MAX);
    if (certifyLossFromUnsignaledFence(result)) {
      return;
    }
    if (result != VK_SUCCESS) {
      VK_ASSERT(result);
      return;
    }
    // A success with no device loss reported proves nothing: keep waiting, as master does on such a
    // fence, backing off so that a fence that answers at once does not spin.
    static_cast<void>(backoff(attempt));
    if (lossCertified_) {
      return;
    }
  }
}

void VulkanImmediateCommands::drainSubmitted() {
  uint32_t attempt = 0;
  for (;;) {
    uint64_t unbounded = UINT64_MAX;
    [[maybe_unused]] const VkResult result = certifyAll(unbounded);
    if (retiredThrough_ >= submissionCounter_) {
      return;
    }
    // Objects the submitted work uses must not be destroyed before it is proven finished: try
    // again a little later (a checkpoint could not be submitted, or a fence wait failed).
    IGL_LOG_ERROR_ONCE(
        "VulkanImmediateCommands (%s): the submitted work cannot be proven finished now (VkResult "
        "%d)\n",
        debugName_.c_str(),
        static_cast<int>(result));
    static_cast<void>(backoff(attempt));
  }
}

VkResult VulkanImmediateCommands::certifyAll(uint64_t& remainingNanoseconds) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  // Without time, only poll: no checkpoint is submitted and nothing is reset.
  const bool pollOnly = remainingNanoseconds == 0;

  VkFence fences[kMaxCommandBuffers] = {};

  uint32_t numFences = 0;
  uint64_t trustedThrough = 0;
  uint64_t untrustedThrough = 0;

  for (const auto& buf : buffers_) {
    IGL_DEBUG_ASSERT(rowAgreesWithWrapper(buf));
    // A quarantined row's submission finished before its fence reset failed, and that fence's
    // state is undefined: it is never waited on.
    if (buf.state != RowState::Submitted || buf.retirement_.quarantined) {
      continue;
    }
    const RowRetirement& row = buf.retirement_;
    if (row.fenceTrusted) {
      fences[numFences++] = buf.fence.vkFence_;
      trustedThrough = std::max(trustedThrough, row.prefix);
    } else {
      untrustedThrough = std::max(untrustedThrough, row.prefix);
    }
  }

  VkResult result = VK_SUCCESS;
  bool lost = false;
  if (numFences) {
    result = waitForFences(vf_, device_, numFences, fences, remainingNanoseconds);
    if (result == VK_SUCCESS) {
      certify(trustedThrough);
    } else if (checkDeviceLost(result)) {
      // Nothing submitted runs any more: no checkpoint is needed, nor submitted to the lost device.
      certifyLoss();
      if (!pollOnly) {
        purge();
      }
      return VK_ERROR_DEVICE_LOST;
    } else if (result != VK_TIMEOUT) {
      VK_ASSERT(result);
    }
  }

  // Exported and unknown submissions that no trusted fence covers need a checkpoint.
  if (result == VK_SUCCESS && untrustedThrough > retiredThrough_) {
    result = certifyThrough(untrustedThrough, remainingNanoseconds, nullptr);
    if (result == VK_ERROR_DEVICE_LOST) {
      lost = true;
      result = VK_SUCCESS;
    }
  }

  if (!pollOnly) {
    purge();
  }

  return (result == VK_SUCCESS && lost) ? VK_ERROR_DEVICE_LOST : result;
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

VulkanImmediateCommands::Ticket VulkanImmediateCommands::ticketOfRow(
    const CommandBufferWrapper& wrapper) const noexcept {
  if (wrapper.state == RowState::Encoding) {
    return {.kind = Ticket::Kind::Open};
  }
  if (wrapper.state == RowState::Dropped) {
    return {.kind = Ticket::Kind::Dropped, .prefix = wrapper.retirement_.prefix, .own = &wrapper};
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
    if (buf.outcomesLost_) {
      // The row's current recording came after a device loss, or the handle also names a recording
      // that was not submitted (submit ids wrap): it never reads as finished.
      return {.kind = Ticket::Kind::Lost};
    }
    return ticketOfRow(buf);
  }
  if (succeededBefore(buf, handle)) {
    return {.kind = Ticket::Kind::Settled};
  }
  if (buf.outcomesLost_) {
    // Any other older recording of this row was lost with the device or never submitted (see
    // `recycleDroppedRows()`).
    return {.kind = Ticket::Kind::Lost};
  }
  // An older recording of this row: it closed before the row was acquired again, so its prefix is
  // at most the previous recording's.
  return {.kind = Ticket::Kind::Prefix, .prefix = buf.retirement_.previousPrefix};
}

bool VulkanImmediateCommands::succeededBefore(const CommandBufferWrapper& buf,
                                              SubmitHandle handle) noexcept {
  const uint64_t current = buf.retirement_.acquisition;
  // Submit ids order acquisitions modulo 2^32.
  const uint64_t age = static_cast<uint32_t>(lowBits(current) - handle.submitId);
  return buf.succeededAcquisition_ != 0 && age <= current &&
         current - age <= buf.succeededAcquisition_;
}

const VkFence* IGL_NULLABLE
VulkanImmediateCommands::lowestSourceFor(uint64_t target,
                                         const CommandBufferWrapper* IGL_NULLABLE own,
                                         uint64_t& ordinal) const noexcept {
  if (own && own->state == RowState::Submitted && own->retirement_.fenceTrusted &&
      own->retirement_.prefix == target) {
    // The common case: the handle's own submission, whose fence no other source precedes.
    ordinal = target;
    return &own->fence.vkFence_;
  }
  const VkFence* fence = nullptr;
  ordinal = UINT64_MAX;
  // A quarantined row needs no check here: one whose submission finished has its prefix at most
  // retiredThrough_, which is below every target this search is asked for, and the fence of one
  // whose discarded recording could not be reset is not trusted.
  for (const auto& buf : buffers_) {
    const RowRetirement& row = buf.retirement_;
    if (buf.state == RowState::Submitted && row.fenceTrusted && row.prefix >= target &&
        row.prefix < ordinal) {
      ordinal = row.prefix;
      fence = &buf.fence.vkFence_;
    }
  }
  for (const Checkpoint& checkpoint : checkpoints_) {
    if (checkpoint.state == Checkpoint::State::Pending && checkpoint.covers >= target &&
        checkpoint.covers < ordinal) {
      ordinal = checkpoint.covers;
      fence = &checkpoint.fence.vkFence_;
    }
  }
  return fence;
}

void VulkanImmediateCommands::certifyQueueIdle(bool lost) const noexcept {
  if (lost) {
    checkDeviceLost(VK_ERROR_DEVICE_LOST);
  }
  certify(submissionCounter_, lost);
  // The queue is idle, so no checkpoint fence is pending, including those whose submission failed.
  for (Checkpoint& checkpoint : checkpoints_) {
    if (checkpoint.state == Checkpoint::State::Pending ||
        checkpoint.state == Checkpoint::State::Abandoned) {
      checkpoint.state = Checkpoint::State::Signaled;
    }
  }
}

void VulkanImmediateCommands::certify(uint64_t ordinal, bool lost) const noexcept {
  retiredThrough_ = std::max(retiredThrough_, ordinal);
  if (lost) {
    lossCertified_ = true;
  } else if (!lossCertified_) {
    succeededThrough_ = std::max(succeededThrough_, ordinal);
  }
}

void VulkanImmediateCommands::certifyLoss() const noexcept {
  certifyQueueIdle(true);
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
    noteFenceSignaled(fence, ordinal, result != VK_SUCCESS);
  }
}

void VulkanImmediateCommands::noteFenceSignaled(const VkFence* fence,
                                                uint64_t ordinal,
                                                bool lost) const noexcept {
  if (lost) {
    certifyLoss();
  } else {
    certify(ordinal);
  }
  for (Checkpoint& checkpoint : checkpoints_) {
    if (&checkpoint.fence.vkFence_ == fence && checkpoint.state == Checkpoint::State::Pending) {
      checkpoint.state = Checkpoint::State::Signaled;
    }
  }
}

VkResult VulkanImmediateCommands::waitFence(const VkFence* fence,
                                            uint64_t ordinal,
                                            uint64_t& remainingNanoseconds) const {
  const VkResult result = waitForFences(vf_, device_, 1, fence, remainingNanoseconds);
  if (result == VK_SUCCESS || checkDeviceLost(result)) {
    noteFenceSignaled(fence, ordinal, result != VK_SUCCESS);
  }
  return result;
}

VkResult VulkanImmediateCommands::certifyThrough(uint64_t target,
                                                 uint64_t& remainingNanoseconds,
                                                 const CommandBufferWrapper* IGL_NULLABLE own) {
  bool lost = false;
  while (retiredThrough_ < target) {
    uint64_t ordinal = 0;
    const VkFence* fence = lowestSourceFor(target, own, ordinal);
    if (!fence && remainingNanoseconds == 0) {
      // No time is left to wait for a checkpoint: do not submit one.
      return VK_TIMEOUT;
    }
    const VkResult result = fence ? waitFence(fence, ordinal, remainingNanoseconds)
                                  : submitCheckpoint(remainingNanoseconds);
    if (result == VK_ERROR_DEVICE_LOST) {
      lost = true;
    } else if (result != VK_SUCCESS) {
      return result;
    }
  }
  return lost ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

bool VulkanImmediateCommands::replaceCheckpointFence(Checkpoint& checkpoint) {
  VulkanFence fence(
      vf_,
      device_,
      VkFenceCreateFlagBits{},
      false,
      IGL_FORMAT("Fence: {} checkpoint #{}", debugName_, &checkpoint - checkpoints_.data())
          .c_str());
  if (fence.initResult() != VK_SUCCESS) {
    static_cast<void>(checkDeviceLost(fence.initResult()));
    return false;
  }
  checkpoint.fence = std::move(fence);
  checkpoint.state = Checkpoint::State::Unused;
  return true;
}

VkResult VulkanImmediateCommands::submitCheckpoint(uint64_t& remainingNanoseconds) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_SUBMIT);

  Checkpoint* slot = nullptr;
  for (Checkpoint& checkpoint : checkpoints_) {
    if (checkpoint.state == Checkpoint::State::Unused ||
        checkpoint.state == Checkpoint::State::Signaled) {
      slot = &checkpoint;
      break;
    }
  }
  for (Checkpoint& checkpoint : checkpoints_) {
    // A quarantined checkpoint signaled before its reset failed: it can get a new fence.
    if (!slot && checkpoint.state == Checkpoint::State::Quarantined &&
        replaceCheckpointFence(checkpoint)) {
      slot = &checkpoint;
    }
  }
  if (!slot && checkpoints_.size() < kMaxCheckpoints) {
    VulkanFence fence(
        vf_,
        device_,
        VkFenceCreateFlagBits{},
        false,
        IGL_FORMAT("Fence: {} checkpoint #{}", debugName_, checkpoints_.size()).c_str());
    if (fence.initResult() != VK_SUCCESS) {
      // A device loss reported here is remembered, but proves nothing about the submitted work, and
      // another try would fail the same way: report it as an error that ends the call.
      return checkDeviceLost(fence.initResult()) ? VK_ERROR_UNKNOWN : fence.initResult();
    }
    slot = &checkpoints_.emplace_back(Checkpoint{.fence = std::move(fence)});
  }
  if (!slot) {
    // Every checkpoint is pending (or abandoned): wait for the oldest pending one first.
    Checkpoint* oldest = nullptr;
    for (Checkpoint& checkpoint : checkpoints_) {
      if (checkpoint.state == Checkpoint::State::Pending &&
          (!oldest || checkpoint.covers < oldest->covers)) {
        oldest = &checkpoint;
      }
    }
    if (!oldest) {
      return VK_ERROR_UNKNOWN;
    }
    const VkResult result =
        waitFence(&oldest->fence.vkFence_, oldest->covers, remainingNanoseconds);
    if (result != VK_SUCCESS) {
      return result;
    }
    if (remainingNanoseconds == 0) {
      // That wait used the rest of the time: do not reset and submit another checkpoint.
      return VK_TIMEOUT;
    }
    slot = oldest;
  }

  if (slot->state == Checkpoint::State::Signaled) {
    const VkResult resetResult = vf_.vkResetFences(device_, 1, &slot->fence.vkFence_);
    if (resetResult != VK_SUCCESS) {
      IGL_LOG_ERROR_ONCE(
          "VulkanImmediateCommands (%s): a checkpoint fence could not be reset (VkResult %d)\n",
          debugName_.c_str(),
          static_cast<int>(resetResult));
      // A reported device loss is remembered; it proves nothing about the submitted work.
      static_cast<void>(checkDeviceLost(resetResult));
      // The fence may still be signaled: it is never submitted again, but replaced when a new fence
      // can be created.
      if (!replaceCheckpointFence(*slot)) {
        slot->state = Checkpoint::State::Quarantined;
        // A loss that only a reset reported does not prove that the work no longer runs.
        return resetResult == VK_ERROR_DEVICE_LOST ? VK_ERROR_UNKNOWN : resetResult;
      }
    }
  }
  // An empty batch: its fence signals once all work submitted earlier to the queue has completed.
  const VkResult result = vf_.vkQueueSubmit(queue_, 0, nullptr, slot->fence.vkFence_);
  if (result == VK_SUCCESS || checkDeviceLost(result)) {
    slot->covers = submissionCounter_;
    slot->state = Checkpoint::State::Pending;
    if (result == VK_ERROR_DEVICE_LOST) {
      certifyLoss();
    }
    return result;
  }
  if (isOutOfMemory(result)) {
    // Nothing was submitted: the fence is unchanged.
    slot->state = Checkpoint::State::Unused;
    return result;
  }
  slot->state = Checkpoint::State::Abandoned;
  IGL_LOG_ERROR_ONCE("VulkanImmediateCommands (%s): a checkpoint submission failed (VkResult %d)\n",
                     debugName_.c_str(),
                     static_cast<int>(result));
  return VK_ERROR_UNKNOWN;
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
  if (ticket.kind == Ticket::Kind::Lost) {
    return true;
  }
  if (ticket.kind == Ticket::Kind::Dropped) {
    // As a fence that nothing signals: only a certified device loss ends it.
    return lossCertified_;
  }
  pollFor(ticket);
  return retiredThrough_ >= ticket.prefix;
}

bool VulkanImmediateCommands::isRetired(SubmitHandle handle) const {
  const Ticket ticket = ticketOf(handle);
  if (ticket.kind == Ticket::Kind::Settled || ticket.kind == Ticket::Kind::Lost) {
    return true;
  }
  if (ticket.kind == Ticket::Kind::Open) {
    return false;
  }
  if (ticket.kind == Ticket::Kind::Dropped && lossCertified_) {
    return true;
  }
  // A recording that was not submitted never runs: it retires with the work submitted before it,
  // and its own fence is never a source.
  pollFor({.kind = Ticket::Kind::Prefix,
           .prefix = ticket.prefix,
           .own = ticket.kind == Ticket::Kind::Prefix ? ticket.own : nullptr});
  return retiredThrough_ >= ticket.prefix;
}

VkResult VulkanImmediateCommands::waitRetired(SubmitHandle handle, uint64_t timeoutNanoseconds) {
  if (isRetired(handle)) {
    return VK_SUCCESS;
  }
  const Ticket ticket = ticketOf(handle);
  if (!IGL_DEBUG_VERIFY(ticket.kind != Ticket::Kind::Open)) {
    // As `wait()`: the recording was not submitted yet, a logic error of the caller.
    return VK_ERROR_UNKNOWN;
  }
  if (timeoutNanoseconds == 0) {
    return VK_TIMEOUT;
  }
  uint64_t remaining = timeoutNanoseconds;
  const VkResult result = certifyThrough(
      ticket.prefix, remaining, ticket.kind == Ticket::Kind::Prefix ? ticket.own : nullptr);
  return retiredThrough_ >= ticket.prefix ? VK_SUCCESS : result;
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
  case RowState::Dropped:
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
  auto& mutableWrapper = const_cast<CommandBufferWrapper&>(wrapper);
  // A stale wrapper (its row was freed) is a misuse that leaves the row as it is.
  const bool staleWrapper = wrapper.state == RowState::Free;
  lastSubmitDroppedTheRecording_ = false;
  lastSubmitOutcomeUnknown_ = false;

  if (!staleWrapper && (chainPoisoned_ || queueState_->stopped || isDeviceLost())) {
    // This submission would wait on a chain semaphore that a recording that was not submitted never
    // signals, or follows one that did: on the queue it would never run, and nor would anything
    // submitted after it. Nothing is queued any more. After a device loss nothing runs either.
    if (chainPoisoned_) {
      queueState_->stopped = true;
    }
    VK_ASSERT(vf_.vkEndCommandBuffer(wrapper.cmdBuf));
    return dropRecording(mutableWrapper, false);
  }

  const VkResult endResult = vf_.vkEndCommandBuffer(wrapper.cmdBuf);
  VK_ASSERT(endResult);

  // Without the full chain, a submission that waits on injected semaphores still signals its chain
  // semaphore so that the next submission is ordered after those waits.
  const bool signalChainSemaphore = chainSubmitsWithSemaphores_ || numWaitSemaphores_ != 0;
  if (endResult != VK_SUCCESS && !staleWrapper) {
    // A recording that could not be ended is not submitted. One that carries injected waits stops
    // the queue at once: later work, of either pool, must not run before what those waits order.
    if (numWaitSemaphores_ != 0) {
      queueState_->stopped = true;
    }
    return dropRecording(mutableWrapper, signalChainSemaphore);
  }

  // The submission chain and presentation signals have capacity separate from injected signals.
  constexpr size_t kSubmitSemaphoreCapacity = kMaxInjectedSemaphores + 2;
  // @lint-ignore CLANGTIDY
  VkSemaphoreSubmitInfo waitSemaphores[kSubmitSemaphoreCapacity] = {};
  uint32_t numWaitSemaphores = 0;
  for (uint32_t i = 0; i < numWaitSemaphores_; ++i) {
    waitSemaphores[numWaitSemaphores++] = waitSemaphores_[i];
  }
  if (lastSubmitSemaphore_.semaphore) {
    waitSemaphores[numWaitSemaphores++] = lastSubmitSemaphore_;
  }
  // @lint-ignore CLANGTIDY
  VkSemaphoreSubmitInfo signalSemaphores[kSubmitSemaphoreCapacity] = {};
  uint32_t numSignalSemaphores = 0;
  for (uint32_t i = 0; i < numSignalSemaphores_; ++i) {
    signalSemaphores[numSignalSemaphores++] = signalSemaphores_[i];
  }
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

  VkResult result = queueSubmit(wrapper,
                                nextCmdBuf,
                                waitSemaphores,
                                numWaitSemaphores,
                                signalSemaphores,
                                numSignalSemaphores);
  if (!checkDeviceLost(result)) {
    VK_ASSERT(result);
  }
  if (isOutOfMemory(result) && !staleWrapper) {
    // Out of memory left the queue, the semaphores and the fence untouched: try once more.
    result = queueSubmit(wrapper,
                         nextCmdBuf,
                         waitSemaphores,
                         numWaitSemaphores,
                         signalSemaphores,
                         numSignalSemaphores);
    if (!checkDeviceLost(result)) {
      VK_ASSERT(result);
    }
  }

  if (isOutOfMemory(result) && !staleWrapper) {
    // Nothing was submitted. A recording that carries injected waits stops the queue at once, as a
    // failed end does: later work, of either pool, must not run before what those waits order.
    if (numWaitSemaphores_ != 0) {
      queueState_->stopped = true;
    }
    return dropRecording(mutableWrapper, signalChainSemaphore);
  }

  lastSubmitHandle_ = wrapper.handle;
  lastAcceptedSubmitHandle_ = wrapper.handle;
  lastSubmitOutcomeUnknown_ = result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST;

  lastSubmitSemaphore_.semaphore = signalChainSemaphore ? wrapper.semaphore.vkSemaphore_
                                                        : VK_NULL_HANDLE;
  numWaitSemaphores_ = 0;
  numSignalSemaphores_ = 0;

  if (!staleWrapper) {
    // A submission that reached the queue, or may have, takes the next submission ordinal. Its
    // fence is trusted after a success or a device loss (which certifies), but not after any other
    // error: then only a later fence or a checkpoint proves the submission finished.
    RowRetirement& row = mutableWrapper.retirement_;
    row.prefix = ++submissionCounter_;
    row.fenceTrusted = row.fenceTrusted && (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST);
  }

  // reset
  setRowState(mutableWrapper, RowState::Submitted);

  nextSubmitHandle_ = {};

  return lastSubmitHandle_;
}

VulkanImmediateCommands::SubmitHandle VulkanImmediateCommands::dropRecording(
    CommandBufferWrapper& wrapper,
    bool poisonChain) {
  IGL_LOG_ERROR_ONCE(
      "VulkanImmediateCommands (%s): a recording that cannot run was not submitted; its handle "
      "never "
      "completes\n",
      debugName_.c_str());
  lastSubmitDroppedTheRecording_ = true;
  chainPoisoned_ = chainPoisoned_ || poisonChain;
  // It retires with the work submitted before it, and is never an alias of a later recording.
  RowRetirement& row = wrapper.retirement_;
  row.prefix = submissionCounter_;
  row.closedUnsubmitted = false;
  numWaitSemaphores_ = 0;
  numSignalSemaphores_ = 0;
  lastSubmitHandle_ = wrapper.handle;
  setRowState(wrapper, RowState::Dropped);
  nextSubmitHandle_ = {};
  return wrapper.handle;
}

VkResult VulkanImmediateCommands::waitDropped(const CommandBufferWrapper& wrapper,
                                              uint64_t timeoutNanoseconds) const {
  if (lossCertified_) {
    return VK_ERROR_DEVICE_LOST;
  }
  if (timeoutNanoseconds == 0) {
    return VK_TIMEOUT;
  }
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);
  // The recording's fence was never submitted and nothing signals it, as the fence of a submission
  // that never reached the queue: only a lost device answers before the time is up.
  const VkResult result =
      vf_.vkWaitForFences(device_, 1, &wrapper.fence.vkFence_, VK_TRUE, timeoutNanoseconds);
  if (certifyLossFromUnsignaledFence(result)) {
    return VK_ERROR_DEVICE_LOST;
  }
  // A success that proves nothing is not an answer for the recording.
  return result == VK_SUCCESS ? VK_ERROR_UNKNOWN : result;
}

VkResult VulkanImmediateCommands::queueSubmit(const CommandBufferWrapper& wrapper,
                                              VkCommandBuffer nextCmdBuf,
                                              const VkSemaphoreSubmitInfo* waitSemaphores,
                                              uint32_t numWaitSemaphores,
                                              const VkSemaphoreSubmitInfo* signalSemaphores,
                                              uint32_t numSignalSemaphores) const {
  constexpr size_t kSubmitSemaphoreCapacity = kMaxInjectedSemaphores + 2;
  const std::array<VkCommandBuffer, 2> cmdBufs = {wrapper.cmdBuf, nextCmdBuf};
  const uint32_t numCmdBufs = nextCmdBuf != VK_NULL_HANDLE ? 2u : 1u;
  if (useTimelineSemaphoreAndSynchronization2_) {
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

    VkResult result = VK_SUCCESS;
    IGL_PROFILER_ZONE("vkQueueSubmit2KHR()", IGL_PROFILER_COLOR_SUBMIT);
#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkQueueSubmit2KHR()\n\n", wrapper.cmdBuf);
#endif // IGL_VULKAN_PRINT_COMMANDS
    result = vf_.vkQueueSubmit2KHR(queue_, 1u, &si, wrapper.fence.vkFence_);
    IGL_PROFILER_ZONE_END();
    return result;
  }

  // @lint-ignore CLANGTIDY
  VkPipelineStageFlags waitStageMasks[kSubmitSemaphoreCapacity] = {};
  // @lint-ignore CLANGTIDY
  VkSemaphore waits[kSubmitSemaphoreCapacity] = {};
  std::array<uint64_t, kSubmitSemaphoreCapacity> waitValues = {};
  for (uint32_t i = 0; i < numWaitSemaphores; ++i) {
    waitStageMasks[i] = static_cast<VkPipelineStageFlags>(waitSemaphores[i].stageMask);
    waits[i] = waitSemaphores[i].semaphore;
    waitValues[i] = waitSemaphores[i].value;
  }
  // @lint-ignore CLANGTIDY
  VkSemaphore signals[kSubmitSemaphoreCapacity] = {};
  std::array<uint64_t, kSubmitSemaphoreCapacity> signalValues = {};
  for (uint32_t i = 0; i < numSignalSemaphores; ++i) {
    signals[i] = signalSemaphores[i].semaphore;
    signalValues[i] = signalSemaphores[i].value;
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
      .pWaitSemaphores = numWaitSemaphores ? waits : nullptr,
      .pWaitDstStageMask = waitStageMasks,
      .commandBufferCount = numCmdBufs,
      .pCommandBuffers = cmdBufs.data(),
      .signalSemaphoreCount = numSignalSemaphores,
      .pSignalSemaphores = signals,
  };
  // @lint-ignore CLANGTIDY
  const VkFence vkFence = wrapper.fence.vkFence_;
  VkResult result = VK_SUCCESS;
  IGL_PROFILER_ZONE("vkQueueSubmit()", IGL_PROFILER_COLOR_SUBMIT);
#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p vkQueueSubmit()\n\n", wrapper.cmdBuf);
#endif // IGL_VULKAN_PRINT_COMMANDS
  result = vf_.vkQueueSubmit(queue_, 1u, &si, vkFence);
  IGL_PROFILER_ZONE_END();
  return result;
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
  const VkResult result = vf_.vkResetCommandBuffer(wrapper.cmdBuf, VkCommandBufferResetFlags{0});
  VK_ASSERT(result);
  RowRetirement& row = wrapper.retirement_;
  row.prefix = submissionCounter_;
  row.closedUnsubmitted = true;
  if (result != VK_SUCCESS) {
    // The command buffer cannot be recorded again until a reset succeeds: quarantine the row (see
    // `recoverQuarantinedRow()`). Its fence was never submitted, so nothing may wait on it.
    row.quarantined = true;
    row.fenceTrusted = false;
    IGL_LOG_ERROR_ONCE(
        "VulkanImmediateCommands (%s): a command buffer could not be reset (VkResult %d)\n",
        debugName_.c_str(),
        static_cast<int>(result));
    setRowState(wrapper, RowState::Submitted);
    return;
  }
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

  if (isRecycled(handle) || buffers_[handle.bufferIndex].outcomesLost_) {
    return VK_NULL_HANDLE;
  }

  return buffers_[handle.bufferIndex].fence.vkFence_;
}

int VulkanImmediateCommands::exportFenceFd(SubmitHandle handle) {
  int fenceFd = -1;
#if defined(VK_KHR_external_fence_fd)
  if (!vf_.vkGetFenceFdKHR) {
    return -1;
  }
  if (handle.bufferIndex < buffers_.size() && buffers_[handle.bufferIndex].outcomesLost_) {
    // Whichever recording of the row the handle names, it was lost or never submitted.
    return -1;
  }
  const VkFence vkFence = getVkFenceFromSubmitHandle(handle);
  IGL_DEBUG_ASSERT(vkFence != VK_NULL_HANDLE);
  if (vkFence == VK_NULL_HANDLE) {
    // The handle no longer has a fence: its row was acquired again.
    return -1;
  }
  if (buffers_[handle.bufferIndex].state == RowState::Dropped) {
    // A recording that was not submitted: nothing signals its fence, which cannot be exported.
    return -1;
  }
  bool& fenceTrusted = buffers_[handle.bufferIndex].retirement_.fenceTrusted;
  const bool wasTrusted = fenceTrusted;
  // A successful export resets the fence: never poll or wait on it again.
  fenceTrusted = false;

  const VkFenceGetFdInfoKHR getFdInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
      .fence = vkFence,
      .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT,
  };
  const VkResult result = vf_.vkGetFenceFdKHR(device_, &getFdInfo, &fenceFd);
  if (result != VK_SUCCESS) {
    // A failed export leaves the fence as it was.
    fenceTrusted = wasTrusted;
    IGL_LOG_ERROR("Unable to get fence fd from submit handle: %llu",
                  static_cast<unsigned long long>(handle.handle()));
  }
#else
  static_cast<void>(handle);
#endif // defined(VK_KHR_external_fence_fd)
  return fenceFd;
}

void VulkanImmediateCommands::storeFDInSubmitHandle(SubmitHandle handle, int fd) noexcept {
  IGL_DEBUG_ASSERT(handle.bufferIndex < buffers_.size());
  buffers_[handle.bufferIndex].fd = fd;
}

int VulkanImmediateCommands::cachedFDFromSubmitHandle(SubmitHandle handle) const noexcept {
  IGL_DEBUG_ASSERT(handle.bufferIndex < buffers_.size());
  if (buffers_[handle.bufferIndex].outcomesLost_) {
    return -1;
  }
  return buffers_[handle.bufferIndex].fd;
}

} // namespace igl::vulkan
