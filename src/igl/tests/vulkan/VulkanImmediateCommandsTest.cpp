/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanImmediateCommands.h>

#include "../util/TestDevice.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <igl/Buffer.h>
#include <igl/CommandBuffer.h>
#include <igl/ComputeCommandEncoder.h>
#include <igl/ComputePipelineState.h>
#include <igl/ShaderCreator.h>
#include <igl/Timer.h>
#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

namespace {

/// Records the Vulkan calls a `VulkanImmediateCommands` makes through a copy of the function table.
/// Handles are named by the order in which the pool created or allocated them, so two runs of the
/// same scenario produce the same text. Queue submissions and fence waits do not reach the device:
/// the submit succeeds and every fence wait answers success.
struct CallTrace {
  VulkanFunctionTable real = {};
  /// What every traced `vkWaitForFences()` answers.
  VkResult waitResult = VK_SUCCESS;
  /// When set, what a traced zero-timeout `vkWaitForFences()` (a poll) answers instead.
  std::optional<VkResult> pollResult;
  std::vector<std::string> lines;
  std::map<uint64_t, size_t> commandBuffers;
  std::map<uint64_t, size_t> fences;
  std::map<uint64_t, size_t> semaphores;
};
thread_local CallTrace gTrace;

template<typename T>
uint64_t handleKey(T handle) {
  if constexpr (std::is_pointer_v<T>) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
  } else {
    return static_cast<uint64_t>(handle);
  }
}

template<typename T>
std::string handleName(const std::map<uint64_t, size_t>& names, const char* prefix, T handle) {
  const auto it = names.find(handleKey(handle));
  return std::string(prefix) + (it == names.end() ? "?" : std::to_string(it->second));
}

void record(std::string line) {
  gTrace.lines.push_back(std::move(line));
}

VKAPI_ATTR void VKAPI_CALL traceGetDeviceQueue(VkDevice device,
                                               uint32_t family,
                                               uint32_t index,
                                               VkQueue* queue) {
  record("vkGetDeviceQueue index=" + std::to_string(index));
  gTrace.real.vkGetDeviceQueue(device, family, index, queue);
}

VKAPI_ATTR VkResult VKAPI_CALL traceCreateCommandPool(VkDevice device,
                                                      const VkCommandPoolCreateInfo* info,
                                                      const VkAllocationCallbacks* allocator,
                                                      VkCommandPool* pool) {
  record("vkCreateCommandPool flags=" + std::to_string(info->flags));
  return gTrace.real.vkCreateCommandPool(device, info, allocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL traceAllocateCommandBuffers(VkDevice device,
                                                           const VkCommandBufferAllocateInfo* info,
                                                           VkCommandBuffer* buffers) {
  const VkResult result = gTrace.real.vkAllocateCommandBuffers(device, info, buffers);
  for (uint32_t i = 0; i < info->commandBufferCount; ++i) {
    const size_t index = gTrace.commandBuffers.size();
    gTrace.commandBuffers[handleKey(buffers[i])] = index;
    record("vkAllocateCommandBuffers cb" + std::to_string(index));
  }
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL traceCreateFence(VkDevice device,
                                                const VkFenceCreateInfo* info,
                                                const VkAllocationCallbacks* allocator,
                                                VkFence* fence) {
  const VkResult result = gTrace.real.vkCreateFence(device, info, allocator, fence);
  const size_t index = gTrace.fences.size();
  gTrace.fences[handleKey(*fence)] = index;
  record("vkCreateFence fence" + std::to_string(index) + " flags=" + std::to_string(info->flags));
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL traceCreateSemaphore(VkDevice device,
                                                    const VkSemaphoreCreateInfo* info,
                                                    const VkAllocationCallbacks* allocator,
                                                    VkSemaphore* semaphore) {
  const VkResult result = gTrace.real.vkCreateSemaphore(device, info, allocator, semaphore);
  const size_t index = gTrace.semaphores.size();
  gTrace.semaphores[handleKey(*semaphore)] = index;
  record("vkCreateSemaphore sem" + std::to_string(index));
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL traceBeginCommandBuffer(VkCommandBuffer cb,
                                                       const VkCommandBufferBeginInfo* info) {
  record("vkBeginCommandBuffer " + handleName(gTrace.commandBuffers, "cb", cb) +
         " flags=" + std::to_string(info->flags));
  return gTrace.real.vkBeginCommandBuffer(cb, info);
}

VKAPI_ATTR VkResult VKAPI_CALL traceEndCommandBuffer(VkCommandBuffer cb) {
  record("vkEndCommandBuffer " + handleName(gTrace.commandBuffers, "cb", cb));
  return gTrace.real.vkEndCommandBuffer(cb);
}

VKAPI_ATTR VkResult VKAPI_CALL traceResetCommandBuffer(VkCommandBuffer cb,
                                                       VkCommandBufferResetFlags flags) {
  record("vkResetCommandBuffer " + handleName(gTrace.commandBuffers, "cb", cb) +
         " flags=" + std::to_string(flags));
  return gTrace.real.vkResetCommandBuffer(cb, flags);
}

VKAPI_ATTR VkResult VKAPI_CALL traceResetFences(VkDevice device,
                                                uint32_t count,
                                                const VkFence* fences) {
  std::string line = "vkResetFences";
  for (uint32_t i = 0; i < count; ++i) {
    line += " " + handleName(gTrace.fences, "fence", fences[i]);
  }
  record(std::move(line));
  return gTrace.real.vkResetFences(device, count, fences);
}

VKAPI_ATTR VkResult VKAPI_CALL traceWaitForFences(VkDevice /*device*/,
                                                  uint32_t count,
                                                  const VkFence* fences,
                                                  VkBool32 waitAll,
                                                  uint64_t timeout) {
  std::string line = "vkWaitForFences all=" + std::to_string(waitAll) +
                     " timeout=" + (timeout == UINT64_MAX ? "max" : std::to_string(timeout));
  for (uint32_t i = 0; i < count; ++i) {
    line += " " + handleName(gTrace.fences, "fence", fences[i]);
  }
  record(std::move(line));
  return (timeout == 0 && gTrace.pollResult) ? *gTrace.pollResult : gTrace.waitResult;
}

VKAPI_ATTR VkResult VKAPI_CALL traceQueueSubmit(VkQueue /*queue*/,
                                                uint32_t count,
                                                const VkSubmitInfo* submits,
                                                VkFence fence) {
  for (uint32_t i = 0; i < count; ++i) {
    std::string line = "vkQueueSubmit " +
                       handleName(gTrace.commandBuffers, "cb", submits[i].pCommandBuffers[0]) +
                       " " + handleName(gTrace.fences, "fence", fence) +
                       " waits=" + std::to_string(submits[i].waitSemaphoreCount) +
                       " signals=" + std::to_string(submits[i].signalSemaphoreCount);
    for (uint32_t w = 0; w < submits[i].waitSemaphoreCount; ++w) {
      line += " wait:" + handleName(gTrace.semaphores, "sem", submits[i].pWaitSemaphores[w]);
    }
    for (uint32_t w = 0; w < submits[i].signalSemaphoreCount; ++w) {
      line += " signal:" + handleName(gTrace.semaphores, "sem", submits[i].pSignalSemaphores[w]);
    }
    record(std::move(line));
  }
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL traceQueueSubmit2(VkQueue /*queue*/,
                                                 uint32_t count,
                                                 const VkSubmitInfo2* submits,
                                                 VkFence fence) {
  for (uint32_t i = 0; i < count; ++i) {
    std::string line =
        "vkQueueSubmit2KHR " +
        handleName(gTrace.commandBuffers, "cb", submits[i].pCommandBufferInfos[0].commandBuffer) +
        " " + handleName(gTrace.fences, "fence", fence) +
        " waits=" + std::to_string(submits[i].waitSemaphoreInfoCount) +
        " signals=" + std::to_string(submits[i].signalSemaphoreInfoCount);
    record(std::move(line));
  }
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL traceDestroyFence(VkDevice device,
                                             VkFence fence,
                                             const VkAllocationCallbacks* allocator) {
  record("vkDestroyFence " + handleName(gTrace.fences, "fence", fence));
  gTrace.real.vkDestroyFence(device, fence, allocator);
}

VKAPI_ATTR void VKAPI_CALL traceDestroySemaphore(VkDevice device,
                                                 VkSemaphore semaphore,
                                                 const VkAllocationCallbacks* allocator) {
  record("vkDestroySemaphore " + handleName(gTrace.semaphores, "sem", semaphore));
  gTrace.real.vkDestroySemaphore(device, semaphore, allocator);
}

VKAPI_ATTR void VKAPI_CALL traceDestroyCommandPool(VkDevice device,
                                                   VkCommandPool pool,
                                                   const VkAllocationCallbacks* allocator) {
  record("vkDestroyCommandPool");
  gTrace.real.vkDestroyCommandPool(device, pool, allocator);
}

/// A copy of `real` whose calls a `VulkanImmediateCommands` makes are traced in `gTrace`.
[[nodiscard]] VulkanFunctionTable tracedTable(const VulkanFunctionTable& real) {
  gTrace = {};
  gTrace.real = real;
  VulkanFunctionTable vf = real;
  vf.vkGetDeviceQueue = traceGetDeviceQueue;
  vf.vkCreateCommandPool = traceCreateCommandPool;
  vf.vkAllocateCommandBuffers = traceAllocateCommandBuffers;
  vf.vkCreateFence = traceCreateFence;
  vf.vkCreateSemaphore = traceCreateSemaphore;
  vf.vkBeginCommandBuffer = traceBeginCommandBuffer;
  vf.vkEndCommandBuffer = traceEndCommandBuffer;
  vf.vkResetCommandBuffer = traceResetCommandBuffer;
  vf.vkResetFences = traceResetFences;
  vf.vkWaitForFences = traceWaitForFences;
  vf.vkQueueSubmit = traceQueueSubmit;
  vf.vkQueueSubmit2KHR = traceQueueSubmit2;
  vf.vkDestroyFence = traceDestroyFence;
  vf.vkDestroySemaphore = traceDestroySemaphore;
  vf.vkDestroyCommandPool = traceDestroyCommandPool;
  return vf;
}

// The calls the pool makes for the call-trace tests were recorded on the code before ordinals and
// row states existed: the new bookkeeping adds no Vulkan call and reorders none.
constexpr size_t kPoolCreationCalls = 2 + 3 * vulkan::VulkanImmediateCommands::kMaxCommandBuffers;

[[nodiscard]] std::string joinLines(const std::vector<std::string>& lines) {
  std::string text;
  for (const std::string& line : lines) {
    text += line + "\n";
  }
  return text;
}

/// Makes every traced fence wait succeed again when it goes out of scope. Declared after the pool
/// of a test, it runs before the pool's destructor waits for its rows, so a failing expectation
/// cannot leave that wait on a fence that never signals.
struct TracedWaitsSucceedAtScopeEnd {
  TracedWaitsSucceedAtScopeEnd() = default;
  TracedWaitsSucceedAtScopeEnd(const TracedWaitsSucceedAtScopeEnd&) = delete;
  TracedWaitsSucceedAtScopeEnd& operator=(const TracedWaitsSucceedAtScopeEnd&) = delete;
  ~TracedWaitsSucceedAtScopeEnd() {
    gTrace.waitResult = VK_SUCCESS;
  }
};
/// A queue that runs nothing: submissions complete only when a test says so, in submission order.
/// Fences are signaled when the submission that carries them completes. The pool under test owns
/// real fences and command buffers; only the queue, the fence waits, the fence resets and the fence
/// export go through this fake.
struct FakeQueue {
  /// Fences of the submissions, in submission order.
  std::vector<VkFence> submitted;
  /// The submissions before this index have completed.
  size_t completed = 0;
  /// The submission index that each fence's pending or completed signal belongs to.
  std::map<uint64_t, size_t> fenceSubmission;
  /// When set, a blocking wait completes every submission up to the one it waits for.
  bool blockingWaitsComplete = true;
  /// What the next submissions answer, front first (then success).
  std::vector<VkResult> submitResults;
  /// When set, what every fence wait answers instead (e.g. device loss).
  std::optional<VkResult> waitOverride;
  size_t polls = 0;
  size_t blockingWaits = 0;
  std::vector<uint64_t> polledFences;
};
thread_local FakeQueue gQueue;

bool fenceSignaled(VkFence fence) {
  const auto it = gQueue.fenceSubmission.find(handleKey(fence));
  return it != gQueue.fenceSubmission.end() && it->second < gQueue.completed;
}

VKAPI_ATTR VkResult VKAPI_CALL fakeQueueSubmit(VkQueue /*queue*/,
                                               uint32_t count,
                                               const VkSubmitInfo* /*submits*/,
                                               VkFence fence) {
  VkResult result = VK_SUCCESS;
  if (!gQueue.submitResults.empty()) {
    result = gQueue.submitResults.front();
    gQueue.submitResults.erase(gQueue.submitResults.begin());
  }
  if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST || result == VK_ERROR_UNKNOWN) {
    // An unknown result is modeled as a submission that did reach the queue.
    gQueue.fenceSubmission[handleKey(fence)] = gQueue.submitted.size();
    gQueue.submitted.push_back(fence);
  }
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL fakeQueueSubmit2(VkQueue queue,
                                                uint32_t count,
                                                const VkSubmitInfo2* /*submits*/,
                                                VkFence fence) {
  return fakeQueueSubmit(queue, count, nullptr, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL fakeWaitForFences(VkDevice /*device*/,
                                                 uint32_t count,
                                                 const VkFence* fences,
                                                 VkBool32 /*waitAll*/,
                                                 uint64_t timeout) {
  if (timeout == 0) {
    ++gQueue.polls;
    gQueue.polledFences.push_back(handleKey(fences[0]));
  } else {
    ++gQueue.blockingWaits;
  }
  if (gQueue.waitOverride) {
    return *gQueue.waitOverride;
  }
  bool all = true;
  for (uint32_t i = 0; i < count; ++i) {
    if (!fenceSignaled(fences[i])) {
      const auto it = gQueue.fenceSubmission.find(handleKey(fences[i]));
      if (timeout == UINT64_MAX && it == gQueue.fenceSubmission.end()) {
        // A real driver would block forever: fail the test, and answer as if the fence were
        // signaled so that the pool cannot spin on it.
        ADD_FAILURE() << "Unbounded wait on a fence that has no pending signal";
        continue;
      }
      if (timeout != 0 && gQueue.blockingWaitsComplete && it != gQueue.fenceSubmission.end()) {
        gQueue.completed = std::max(gQueue.completed, it->second + 1);
      } else {
        all = false;
      }
    }
  }
  return all ? VK_SUCCESS : VK_TIMEOUT;
}

VKAPI_ATTR VkResult VKAPI_CALL fakeResetFences(VkDevice /*device*/,
                                               uint32_t count,
                                               const VkFence* fences) {
  for (uint32_t i = 0; i < count; ++i) {
    gQueue.fenceSubmission.erase(handleKey(fences[i]));
  }
  return VK_SUCCESS;
}

[[nodiscard]] VulkanFunctionTable fakeQueueTable(const VulkanFunctionTable& real) {
  gQueue = {};
  VulkanFunctionTable vf = real;
  vf.vkQueueSubmit = fakeQueueSubmit;
  vf.vkQueueSubmit2KHR = fakeQueueSubmit2;
  vf.vkWaitForFences = fakeWaitForFences;
  vf.vkResetFences = fakeResetFences;
  return vf;
}

/// Completes the first `n` submissions made to the fake queue.
void completeSubmissions(size_t n) {
  gQueue.completed = std::max(gQueue.completed, n);
}

/// Which descriptor pool each descriptor set came from, and the pool of the last set written with
/// a storage buffer.
struct DescriptorPoolProbeState {
  PFN_vkAllocateDescriptorSets realAllocate = nullptr;
  PFN_vkUpdateDescriptorSets realUpdate = nullptr;
  std::map<uint64_t, VkDescriptorPool> poolOfSet;
  VkDescriptorPool lastStorageBufferPool = VK_NULL_HANDLE;
};
thread_local DescriptorPoolProbeState gProbe;

VKAPI_ATTR VkResult VKAPI_CALL probeAllocateDescriptorSets(VkDevice device,
                                                           const VkDescriptorSetAllocateInfo* info,
                                                           VkDescriptorSet* sets) {
  const VkResult result = gProbe.realAllocate(device, info, sets);
  if (result == VK_SUCCESS) {
    for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      gProbe.poolOfSet[handleKey(sets[i])] = info->descriptorPool;
    }
  }
  return result;
}

VKAPI_ATTR void VKAPI_CALL probeUpdateDescriptorSets(VkDevice device,
                                                     uint32_t writeCount,
                                                     const VkWriteDescriptorSet* writes,
                                                     uint32_t copyCount,
                                                     const VkCopyDescriptorSet* copies) {
  for (uint32_t i = 0; i < writeCount; ++i) {
    if (writes[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      const auto it = gProbe.poolOfSet.find(handleKey(writes[i].dstSet));
      gProbe.lastStorageBufferPool = it == gProbe.poolOfSet.end() ? VK_NULL_HANDLE : it->second;
    }
  }
  gProbe.realUpdate(device, writeCount, writes, copyCount, copies);
}

} // namespace

class VulkanImmediateCommandsTest : public ::testing::Test {
 public:
  VulkanImmediateCommandsTest() = default;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;

  igl::vulkan::VulkanContext& getVulkanContext() {
    auto& device = static_cast<igl::vulkan::Device&>(*iglDev_);
    return device.getVulkanContext();
  }

  struct SubmitPayload {
    bool synchronization2 = false;
    bool timelineInfo = false;
    uint32_t waitValueCount = 0;
    uint32_t signalValueCount = 0;
    std::vector<VkSemaphore> waits;
    std::vector<VkSemaphore> signals;
    std::vector<VkPipelineStageFlags2> waitStages;
    std::vector<uint64_t> waitValues;
    std::vector<uint64_t> signalValues;
  };
  static thread_local SubmitPayload submitPayload;

  static VKAPI_ATTR VkResult VKAPI_CALL captureQueueSubmit([[maybe_unused]] VkQueue queue,
                                                           uint32_t count,
                                                           const VkSubmitInfo* submits,
                                                           [[maybe_unused]] VkFence fence) {
    EXPECT_EQ(count, 1u);
    submitPayload = {};
    const auto& submit = submits[0];
    for (uint32_t i = 0; i < submit.waitSemaphoreCount; ++i) {
      submitPayload.waits.push_back(submit.pWaitSemaphores[i]);
      submitPayload.waitStages.push_back(submit.pWaitDstStageMask[i]);
    }
    submitPayload.signals.assign(submit.pSignalSemaphores,
                                 submit.pSignalSemaphores + submit.signalSemaphoreCount);
    if (submit.pNext != nullptr) {
      const auto& timeline = *static_cast<const VkTimelineSemaphoreSubmitInfo*>(submit.pNext);
      EXPECT_EQ(timeline.sType, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
      EXPECT_EQ(timeline.pNext, nullptr);
      submitPayload.timelineInfo = true;
      submitPayload.waitValueCount = timeline.waitSemaphoreValueCount;
      submitPayload.signalValueCount = timeline.signalSemaphoreValueCount;
      for (uint32_t i = 0; i < timeline.waitSemaphoreValueCount; ++i) {
        submitPayload.waitValues.push_back(timeline.pWaitSemaphoreValues[i]);
      }
      submitPayload.signalValues.assign(
          timeline.pSignalSemaphoreValues,
          timeline.pSignalSemaphoreValues + timeline.signalSemaphoreValueCount);
    }
    return VK_SUCCESS;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL captureQueueSubmit2([[maybe_unused]] VkQueue queue,
                                                            uint32_t count,
                                                            const VkSubmitInfo2* submits,
                                                            [[maybe_unused]] VkFence fence) {
    EXPECT_EQ(count, 1u);
    submitPayload = {};
    submitPayload.synchronization2 = true;
    const auto& submit = submits[0];
    EXPECT_EQ(submit.pNext, nullptr);
    for (uint32_t i = 0; i < submit.waitSemaphoreInfoCount; ++i) {
      const auto& wait = submit.pWaitSemaphoreInfos[i];
      submitPayload.waits.push_back(wait.semaphore);
      submitPayload.waitStages.push_back(wait.stageMask);
      submitPayload.waitValues.push_back(wait.value);
    }
    for (uint32_t i = 0; i < submit.signalSemaphoreInfoCount; ++i) {
      const auto& signal = submit.pSignalSemaphoreInfos[i];
      submitPayload.signals.push_back(signal.semaphore);
      submitPayload.signalValues.push_back(signal.value);
    }
    return VK_SUCCESS;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL completeFenceWait([[maybe_unused]] VkDevice device,
                                                          [[maybe_unused]] uint32_t count,
                                                          [[maybe_unused]] const VkFence* fences,
                                                          [[maybe_unused]] VkBool32 waitAll,
                                                          [[maybe_unused]] uint64_t timeout) {
    return VK_SUCCESS;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL
  loseDeviceOnQueueSubmit([[maybe_unused]] VkQueue queue,
                          [[maybe_unused]] uint32_t count,
                          [[maybe_unused]] const VkSubmitInfo* submits,
                          [[maybe_unused]] VkFence fence) {
    return VK_ERROR_DEVICE_LOST;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL
  loseDeviceOnQueueSubmit2([[maybe_unused]] VkQueue queue,
                           [[maybe_unused]] uint32_t count,
                           [[maybe_unused]] const VkSubmitInfo2* submits,
                           [[maybe_unused]] VkFence fence) {
    return VK_ERROR_DEVICE_LOST;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL
  loseDeviceOnFenceWait([[maybe_unused]] VkDevice device,
                        [[maybe_unused]] uint32_t count,
                        [[maybe_unused]] const VkFence* fences,
                        [[maybe_unused]] VkBool32 waitAll,
                        [[maybe_unused]] uint64_t timeout) {
    return VK_ERROR_DEVICE_LOST;
  }

  [[nodiscard]] VulkanFunctionTable captureSubmits() {
    VulkanFunctionTable vf = getVulkanContext().vf_;
    submitPayload = {};
    vf.vkQueueSubmit = captureQueueSubmit;
    vf.vkQueueSubmit2KHR = captureQueueSubmit2;
    // Payload captures queue no work; their fences complete through the test function table.
    vf.vkWaitForFences = completeFenceWait;
    return vf;
  }

  static VkSemaphore lastSubmitSemaphore(const igl::vulkan::VulkanImmediateCommands& commands) {
    return commands.lastSubmitSemaphore_.semaphore;
  }

  using Commands = igl::vulkan::VulkanImmediateCommands;

  static const char* rowState(const Commands& commands, uint32_t row) {
    switch (commands.buffers_[row].state) {
    case Commands::RowState::Free:
      return "Free";
    case Commands::RowState::Encoding:
      return "Encoding";
    case Commands::RowState::Submitted:
      return "Submitted";
    }
    return "?";
  }

  static const Commands::CommandBufferWrapper& wrapperOf(const Commands& commands, uint32_t row) {
    return commands.buffers_[row];
  }

  /// What `submit()` does to the state of the row of a wrapper.
  static void moveRowToSubmitted(Commands& commands, uint32_t row) {
    commands.setRowState(commands.buffers_[row], Commands::RowState::Submitted);
  }

  /// Frees a row without any Vulkan call, so that a test that moved a row by hand cannot leave the
  /// pool's destructor waiting for a fence that no submission will ever signal.
  static void freeRow(Commands& commands, uint32_t row) {
    commands.setRowState(commands.buffers_[row], Commands::RowState::Free);
  }

  static bool rowAgrees(const Commands& commands, uint32_t row) {
    return Commands::rowAgreesWithWrapper(commands.buffers_[row]);
  }

  static uint64_t retiredThrough(const Commands& commands) {
    return commands.retiredThrough_;
  }
  static uint64_t submissions(const Commands& commands) {
    return commands.submissionCounter_;
  }
  static void setAcquisitionCounter(Commands& commands, uint64_t value) {
    commands.acquisitionCounter_ = value;
  }
  static uint64_t acquisitionOf(const Commands& commands, uint32_t row) {
    return commands.buffers_[row].retirement_.acquisition;
  }

  /// A pool on a fresh fake queue (`gQueue`). The pool keeps a reference to its function table,
  /// which the fixture owns.
  std::unique_ptr<Commands> makeFakePool(const char* name) {
    auto& ctx = getVulkanContext();
    fakeTable_ = fakeQueueTable(ctx.vf_);
    return std::make_unique<Commands>(fakeTable_,
                                      ctx.getVkDevice(),
                                      ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                      false,
                                      false,
                                      false,
                                      name);
  }
  VulkanFunctionTable fakeTable_ = {};

  /// Replaces a pool of the context (`slot`) with a pool on the fake queue until destroyed. Then it
  /// completes everything submitted to the fake queue, runs the deferred tasks and puts the real
  /// pool back, also after a failed assertion.
  class FakeContextPool {
   public:
    FakeContextPool(VulkanImmediateCommandsTest& test,
                    std::unique_ptr<Commands>& slot,
                    const char* name) :
      ctx_(test.getVulkanContext()), slot_(slot), realPool_(std::move(slot)) {
      slot_ = test.makeFakePool(name);
    }
    FakeContextPool(const FakeContextPool&) = delete;
    FakeContextPool& operator=(const FakeContextPool&) = delete;
    ~FakeContextPool() {
      gQueue.waitOverride.reset();
      gQueue.submitResults.clear();
      gQueue.blockingWaitsComplete = true;
      completeSubmissions(gQueue.submitted.size());
      ctx_.waitDeferredTasks();
      slot_ = std::move(realPool_);
    }
    Commands& operator*() const {
      return *slot_;
    }
    Commands* operator->() const {
      return slot_.get();
    }

   private:
    igl::vulkan::VulkanContext& ctx_;
    std::unique_ptr<Commands>& slot_;
    std::unique_ptr<Commands> realPool_;
  };

  /// Replaces one entry of the context's function table until destroyed.
  template<typename Fn>
  class ScopedTableEntry {
   public:
    ScopedTableEntry(const VulkanFunctionTable& table, Fn VulkanFunctionTable::* entry, Fn fn) :
      table_(const_cast<VulkanFunctionTable&>(table)), entry_(entry), saved_(table_.*entry) {
      table_.*entry_ = fn;
    }
    ScopedTableEntry(const ScopedTableEntry&) = delete;
    ScopedTableEntry& operator=(const ScopedTableEntry&) = delete;
    ~ScopedTableEntry() {
      table_.*entry_ = saved_;
    }
    [[nodiscard]] Fn saved() const {
      return saved_;
    }

   private:
    VulkanFunctionTable& table_;
    Fn VulkanFunctionTable::* entry_;
    Fn saved_;
  };

  /// Records, through the context's function table, which pool each descriptor set comes from
  /// (`gProbe`), until destroyed.
  class DescriptorPoolProbe {
   public:
    explicit DescriptorPoolProbe(const VulkanFunctionTable& table) :
      allocate_(table, &VulkanFunctionTable::vkAllocateDescriptorSets, probeAllocateDescriptorSets),
      update_(table, &VulkanFunctionTable::vkUpdateDescriptorSets, probeUpdateDescriptorSets) {
      gProbe = {};
      gProbe.realAllocate = allocate_.saved();
      gProbe.realUpdate = update_.saved();
    }

   private:
    ScopedTableEntry<PFN_vkAllocateDescriptorSets> allocate_;
    ScopedTableEntry<PFN_vkUpdateDescriptorSets> update_;
  };

  /// A compute pipeline with one storage buffer binding, and two buffers to alternate so that every
  /// dispatch writes a new descriptor set.
  struct StorageBufferDispatch {
    std::shared_ptr<IComputePipelineState> pipeline;
    std::array<std::shared_ptr<IBuffer>, 2> buffers;
  };

  StorageBufferDispatch makeStorageBufferDispatch() {
    static constexpr std::string_view kShader =
        "layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
        "layout (std430, binding = 0, set = 1) writeonly buffer outBuf {\n"
        "  uint values[];\n"
        "};\n"
        "void main() {\n"
        "  values[0] = 1u;\n"
        "}\n";
    StorageBufferDispatch dispatch;
    Result ret;
    auto stages = ShaderStagesCreator::fromModuleStringInput(
        *iglDev_,
        kShader.data(), // NOLINT(bugprone-suspicious-stringview-data-usage)
        "main",
        "",
        &ret);
    if (!ret.isOk() || !stages) {
      return dispatch;
    }
    const ComputePipelineDesc desc{.buffersMap = {{0, IGL_NAMEHANDLE("outBuf")}},
                                   .shaderStages = std::move(stages)};
    dispatch.pipeline = iglDev_->createComputePipeline(desc, &ret);
    for (auto& buffer : dispatch.buffers) {
      const BufferDesc bufferDesc{
          .type = BufferDesc::BufferTypeBits::Storage,
          .length = sizeof(uint32_t) * 4,
          .storage = ResourceStorage::Private,
      };
      buffer = iglDev_->createBuffer(bufferDesc, &ret);
      if (!buffer) {
        dispatch.pipeline = nullptr;
      }
    }
    return dispatch;
  }

  /// Records `count` dispatches into `cmdBuf`; each one takes a new storage buffer descriptor set.
  static void recordDispatches(ICommandBuffer& cmdBuf,
                               const StorageBufferDispatch& dispatch,
                               uint32_t count) {
    auto encoder = cmdBuf.createComputeCommandEncoder();
    ASSERT_NE(encoder, nullptr);
    encoder->bindComputePipelineState(dispatch.pipeline);
    for (uint32_t i = 0; i != count; ++i) {
      encoder->bindBuffer(0, dispatch.buffers[i % 2].get());
      encoder->dispatchThreadGroups(Dimensions(1, 1, 1), Dimensions(1, 1, 1));
    }
    encoder->endEncoding();
  }

  /// Discards the recording of a command buffer that was not submitted.
  void discardRecording(ICommandBuffer& cmdBuf) {
    auto& ctx = getVulkanContext();
    const auto handle = static_cast<igl::vulkan::CommandBuffer&>(cmdBuf).getNextSubmitHandle();
    ctx.immediate_->discard(wrapperOf(*ctx.immediate_, handle.bufferIndex));
  }

  igl::vulkan::VulkanImmediateCommands::SubmitHandle submitForPresentation(
      igl::vulkan::VulkanImmediateCommands& commands,
      const igl::vulkan::VulkanImmediateCommands::CommandBufferWrapper& wrapper,
      VkSemaphore semaphore) {
    return commands.submit(wrapper, semaphore);
  }
};

thread_local VulkanImmediateCommandsTest::SubmitPayload VulkanImmediateCommandsTest::submitPayload;

TEST_F(VulkanImmediateCommandsTest, AcquireCommandBuffer) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  const auto& wrapper = ctx.immediate_->acquire();
  EXPECT_TRUE(wrapper.cmdBuf != VK_NULL_HANDLE);
  EXPECT_TRUE(wrapper.isEncoding);
  ctx.immediate_->submit(wrapper);
}

TEST_F(VulkanImmediateCommandsTest, SubmitAndWait) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  const auto& wrapper = ctx.immediate_->acquire();
  ASSERT_TRUE(wrapper.cmdBuf != VK_NULL_HANDLE);

  auto handle = ctx.immediate_->submit(wrapper);
  EXPECT_FALSE(handle.empty());

  VkResult result = ctx.immediate_->wait(handle);
  EXPECT_EQ(result, VK_SUCCESS);
  EXPECT_TRUE(ctx.immediate_->isReady(handle));
}

TEST_F(VulkanImmediateCommandsTest, MultipleSubmitsGetUniqueHandles) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  std::set<uint32_t> submitIds;

  for (int i = 0; i < 3; ++i) {
    const auto& wrapper = ctx.immediate_->acquire();
    auto handle = ctx.immediate_->submit(wrapper);
    EXPECT_FALSE(handle.empty());
    submitIds.insert(handle.submitId);
  }

  EXPECT_EQ(submitIds.size(), 3u);
}

TEST_F(VulkanImmediateCommandsTest, CommandBufferRecycling) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  std::vector<igl::vulkan::VulkanImmediateCommands::SubmitHandle> handles;

  for (int i = 0; i < 4; ++i) {
    const auto& wrapper = ctx.immediate_->acquire();
    auto handle = ctx.immediate_->submit(wrapper);
    handles.push_back(handle);
  }

  for (auto& h : handles) {
    ctx.immediate_->wait(h);
  }

  const auto& wrapper = ctx.immediate_->acquire();
  EXPECT_TRUE(wrapper.cmdBuf != VK_NULL_HANDLE);
  EXPECT_TRUE(wrapper.isEncoding);
  ctx.immediate_->submit(wrapper);
}

TEST_F(VulkanImmediateCommandsTest, GetLastSubmitHandle) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  const auto& wrapper1 = ctx.immediate_->acquire();
  ctx.immediate_->submit(wrapper1);

  const auto& wrapper2 = ctx.immediate_->acquire();
  auto handle2 = ctx.immediate_->submit(wrapper2);

  auto lastHandle = ctx.immediate_->getLastSubmitHandle();
  EXPECT_EQ(lastHandle.submitId, handle2.submitId);
  EXPECT_EQ(lastHandle.bufferIndex, handle2.bufferIndex);
}

TEST_F(VulkanImmediateCommandsTest, IsReadyOnEmptyHandle) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  igl::vulkan::VulkanImmediateCommands::SubmitHandle emptyHandle;
  EXPECT_TRUE(emptyHandle.empty());
  EXPECT_TRUE(ctx.immediate_->isReady(emptyHandle));
}

TEST_F(VulkanImmediateCommandsTest, WaitAllCommandBuffers) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  std::vector<igl::vulkan::VulkanImmediateCommands::SubmitHandle> handles;

  for (int i = 0; i < 4; ++i) {
    const auto& wrapper = ctx.immediate_->acquire();
    auto handle = ctx.immediate_->submit(wrapper);
    handles.push_back(handle);
  }

  ctx.immediate_->waitAll();

  for (auto& h : handles) {
    EXPECT_TRUE(ctx.immediate_->isReady(h));
  }
}

TEST_F(VulkanImmediateCommandsTest, GetNextSubmitHandle) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  // Submit one buffer first so getNextSubmitHandle has a valid state
  const auto& wrapper = ctx.immediate_->acquire();
  ctx.immediate_->submit(wrapper);

  auto nextHandle = ctx.immediate_->getNextSubmitHandle();
  EXPECT_GT(nextHandle.submitId, 0u);
}

TEST_F(VulkanImmediateCommandsTest, SubmitHandleComparison) {
  igl::vulkan::VulkanImmediateCommands::SubmitHandle h1;
  h1.bufferIndex = 0;
  h1.submitId = 1;

  igl::vulkan::VulkanImmediateCommands::SubmitHandle h2;
  h2.bufferIndex = 0;
  h2.submitId = 1;

  EXPECT_EQ(h1, h2);

  h2.submitId = 2;
  EXPECT_NE(h1, h2);
}

TEST_F(VulkanImmediateCommandsTest, SubmitHandleRoundTrip) {
  igl::vulkan::VulkanImmediateCommands::SubmitHandle original;
  original.bufferIndex = 5;
  original.submitId = 42;

  uint64_t packed = original.handle();

  igl::vulkan::VulkanImmediateCommands::SubmitHandle restored(packed);
  EXPECT_EQ(restored.bufferIndex, original.bufferIndex);
  EXPECT_EQ(restored.submitId, original.submitId);
}

TEST_F(VulkanImmediateCommandsTest, GetVkFenceFromSubmitHandle) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  const auto& wrapper = ctx.immediate_->acquire();
  auto handle = ctx.immediate_->submit(wrapper);
  ASSERT_FALSE(handle.empty());

  VkFence fence = ctx.immediate_->getVkFenceFromSubmitHandle(handle);
  EXPECT_NE(fence, VK_NULL_HANDLE);

  ctx.immediate_->wait(handle);
}

TEST_F(VulkanImmediateCommandsTest, StoreFDAndCachedFDFromSubmitHandle) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  const auto& wrapper = ctx.immediate_->acquire();
  auto handle = ctx.immediate_->submit(wrapper);

  EXPECT_EQ(ctx.immediate_->cachedFDFromSubmitHandle(handle), -1);

  const int testFD = 42;
  ctx.immediate_->storeFDInSubmitHandle(handle, testFD);
  EXPECT_EQ(ctx.immediate_->cachedFDFromSubmitHandle(handle), testFD);

  ctx.immediate_->wait(handle);
}

TEST_F(VulkanImmediateCommandsTest, InjectedSemaphoreCapacityFailureIsReported) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);
  const auto& wrapper = ctx.immediate_->acquire();

  for (uint32_t i = 0; i < vulkan::VulkanImmediateCommands::kMaxInjectedSemaphores; ++i) {
    EXPECT_TRUE(ctx.immediate_->waitSemaphore(VK_NULL_HANDLE));
    EXPECT_TRUE(ctx.immediate_->signalSemaphore(VK_NULL_HANDLE, i));
  }

  EXPECT_FALSE(ctx.immediate_->waitSemaphore(VK_NULL_HANDLE));
  EXPECT_FALSE(ctx.immediate_->signalSemaphore(VK_NULL_HANDLE, 0));

  ctx.immediate_->discard(wrapper);
  EXPECT_FALSE(ctx.immediate_->hasPendingInjectedSemaphores());

  const auto& recoveredWrapper = ctx.immediate_->acquire();
  EXPECT_TRUE(ctx.immediate_->waitSemaphore(VK_NULL_HANDLE));
  ctx.immediate_->discard(recoveredWrapper);
}

TEST_F(VulkanImmediateCommandsTest, PresentationSignalPreservesInjectedSignalCapacity) {
  auto& ctx = getVulkanContext();
  if (!ctx.features().has_VK_KHR_timeline_semaphore ||
      !ctx.features().has_VK_KHR_synchronization2) {
    GTEST_SKIP() << "Timeline presentation is unavailable on this device";
  }
  std::vector<std::unique_ptr<vulkan::VulkanSemaphore>> signals;
  vulkan::VulkanSemaphore presentation(ctx.vf_, ctx.getVkDevice(), false, "Presentation test");
  const auto& wrapper = ctx.immediate_->acquire();
  for (uint32_t index = 0; index < vulkan::VulkanImmediateCommands::kMaxInjectedSemaphores;
       ++index) {
    signals.push_back(std::make_unique<vulkan::VulkanSemaphore>(
        ctx.vf_, ctx.getVkDevice(), uint64_t{0}, false, "Injected signal test"));
    EXPECT_TRUE(ctx.immediate_->signalSemaphore(signals.back()->getVkSemaphore(), index + 1));
  }
  const auto handle =
      submitForPresentation(*ctx.immediate_, wrapper, presentation.getVkSemaphore());
  ASSERT_EQ(ctx.immediate_->wait(handle), VK_SUCCESS);
  for (size_t index = 0; index < signals.size(); ++index) {
    uint64_t value = 0;
    ASSERT_EQ(ctx.vf_.vkGetSemaphoreCounterValueKHR(
                  ctx.getVkDevice(), signals[index]->getVkSemaphore(), &value),
              VK_SUCCESS);
    EXPECT_EQ(value, index + 1);
  }
  const auto& consumer = ctx.immediate_->acquire();
  EXPECT_TRUE(ctx.immediate_->waitSemaphore(presentation.getVkSemaphore()));
  const auto consumerHandle = ctx.immediate_->submit(consumer);
  EXPECT_EQ(ctx.immediate_->wait(consumerHandle), VK_SUCCESS);
}

TEST_F(VulkanImmediateCommandsTest, LegacyPresentationSignalPreservesInjectedSignalCapacity) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanImmediateCommands commands(ctx.vf_,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Legacy presentation test");
  std::vector<std::unique_ptr<vulkan::VulkanSemaphore>> signals;
  vulkan::VulkanSemaphore presentation(ctx.vf_, ctx.getVkDevice(), false, "Presentation test");
  const auto& wrapper = commands.acquire();
  for (uint32_t index = 0; index < vulkan::VulkanImmediateCommands::kMaxInjectedSemaphores;
       ++index) {
    signals.push_back(std::make_unique<vulkan::VulkanSemaphore>(
        ctx.vf_, ctx.getVkDevice(), false, "Injected legacy signal test"));
    EXPECT_TRUE(commands.signalSemaphore(signals.back()->getVkSemaphore(), 0));
  }
  const auto handle = submitForPresentation(commands, wrapper, presentation.getVkSemaphore());
  ASSERT_EQ(commands.wait(handle), VK_SUCCESS);
  const auto& consumer = commands.acquire();
  for (const auto& signal : signals) {
    EXPECT_TRUE(commands.waitSemaphore(signal->getVkSemaphore()));
  }
  const auto consumerHandle = commands.submit(consumer);
  ASSERT_EQ(commands.wait(consumerHandle), VK_SUCCESS);
  const auto& presentationConsumer = commands.acquire();
  EXPECT_TRUE(commands.waitSemaphore(presentation.getVkSemaphore()));
  const auto presentationHandle = commands.submit(presentationConsumer);
  EXPECT_EQ(commands.wait(presentationHandle), VK_SUCCESS);
}

TEST_F(VulkanImmediateCommandsTest, LegacySubmitPreservesMixedSemaphoreValues) {
  auto& ctx = getVulkanContext();
  const auto vf = captureSubmits();
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           true,
                                           false,
                                           "Legacy timeline payload test");
  vulkan::VulkanSemaphore binary(ctx.vf_, ctx.getVkDevice(), false, "Binary payload test");
  vulkan::VulkanSemaphore timeline(ctx.vf_, ctx.getVkDevice(), false, "Timeline payload handle");
  vulkan::VulkanSemaphore presentation(ctx.vf_, ctx.getVkDevice(), false, "Presentation payload");
  const auto& first = commands.acquire();
  const VkSemaphore chain = first.semaphore.getVkSemaphore();
  commands.submit(first);
  EXPECT_TRUE(submitPayload.timelineInfo);
  EXPECT_EQ(submitPayload.waitValueCount, 0u);
  EXPECT_TRUE(submitPayload.waitValues.empty());
  EXPECT_EQ(submitPayload.signalValues, (std::vector<uint64_t>{0}));

  const auto& wrapper = commands.acquire();
  constexpr uint64_t kWaitValue = 0x1234567800000029;
  constexpr uint64_t kSignalValue = 0x2345678900000037;
  ASSERT_TRUE(commands.waitSemaphore(binary.getVkSemaphore()));
  ASSERT_TRUE(commands.waitSemaphore(timeline.getVkSemaphore(), 1));
  commands.cancelLastWaitSemaphore();
  ASSERT_TRUE(commands.waitSemaphore(
      timeline.getVkSemaphore(), kWaitValue, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT));
  ASSERT_TRUE(commands.signalSemaphore(binary.getVkSemaphore(), 0));
  ASSERT_TRUE(commands.signalSemaphore(timeline.getVkSemaphore(), kSignalValue));
  submitForPresentation(commands, wrapper, presentation.getVkSemaphore());

  EXPECT_FALSE(submitPayload.synchronization2);
  EXPECT_TRUE(submitPayload.timelineInfo);
  EXPECT_EQ(submitPayload.waits,
            (std::vector<VkSemaphore>{binary.getVkSemaphore(), timeline.getVkSemaphore(), chain}));
  EXPECT_EQ(submitPayload.signals,
            (std::vector<VkSemaphore>{binary.getVkSemaphore(),
                                      timeline.getVkSemaphore(),
                                      wrapper.semaphore.getVkSemaphore(),
                                      presentation.getVkSemaphore()}));
  EXPECT_EQ(submitPayload.waitValueCount, submitPayload.waits.size());
  EXPECT_EQ(submitPayload.signalValueCount, submitPayload.signals.size());
  EXPECT_EQ(submitPayload.waitValues, (std::vector<uint64_t>{0, kWaitValue, 0}));
  EXPECT_EQ(submitPayload.signalValues, (std::vector<uint64_t>{0, kSignalValue, 0, 0}));
  EXPECT_EQ(submitPayload.waitStages,
            (std::vector<VkPipelineStageFlags2>{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}));
  EXPECT_FALSE(commands.hasPendingInjectedSemaphores());
}

TEST_F(VulkanImmediateCommandsTest, BinaryOnlySubmitOmitsTimelineInfo) {
  auto& ctx = getVulkanContext();
  const auto vf = captureSubmits();
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           true,
                                           "Binary payload test");
  vulkan::VulkanSemaphore binary(ctx.vf_, ctx.getVkDevice(), false, "Binary payload");
  const auto& wrapper = commands.acquire();
  ASSERT_TRUE(commands.waitSemaphore(binary.getVkSemaphore(), 0, VK_PIPELINE_STAGE_TRANSFER_BIT));
  ASSERT_TRUE(commands.signalSemaphore(binary.getVkSemaphore(), 0));
  commands.submit(wrapper);

  EXPECT_FALSE(submitPayload.synchronization2);
  EXPECT_FALSE(submitPayload.timelineInfo);
  EXPECT_EQ(submitPayload.waits, (std::vector<VkSemaphore>{binary.getVkSemaphore()}));
  EXPECT_EQ(submitPayload.waitStages,
            (std::vector<VkPipelineStageFlags2>{VK_PIPELINE_STAGE_TRANSFER_BIT}));
  EXPECT_EQ(
      submitPayload.signals,
      (std::vector<VkSemaphore>{binary.getVkSemaphore(), wrapper.semaphore.getVkSemaphore()}));
}

TEST_F(VulkanImmediateCommandsTest, ModernSubmitPreservesTimelineValues) {
  auto& ctx = getVulkanContext();
  const auto vf = captureSubmits();
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           true,
                                           true,
                                           "Modern payload test");
  vulkan::VulkanSemaphore timeline(ctx.vf_, ctx.getVkDevice(), false, "Timeline payload handle");
  vulkan::VulkanSemaphore presentation(ctx.vf_, ctx.getVkDevice(), false, "Presentation payload");
  const auto& wrapper = commands.acquire();
  constexpr uint64_t kWaitValue = 0x1234567800000029;
  constexpr uint64_t kSignalValue = 0x2345678900000037;
  ASSERT_TRUE(commands.waitSemaphore(
      timeline.getVkSemaphore(), kWaitValue, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT));
  ASSERT_TRUE(commands.signalSemaphore(timeline.getVkSemaphore(), kSignalValue));
  submitForPresentation(commands, wrapper, presentation.getVkSemaphore());

  EXPECT_TRUE(submitPayload.synchronization2);
  EXPECT_FALSE(submitPayload.timelineInfo);
  EXPECT_EQ(submitPayload.waits, (std::vector<VkSemaphore>{timeline.getVkSemaphore()}));
  EXPECT_EQ(submitPayload.waitStages,
            (std::vector<VkPipelineStageFlags2>{VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT}));
  EXPECT_EQ(submitPayload.waitValues, (std::vector<uint64_t>{kWaitValue}));
  EXPECT_EQ(submitPayload.signals,
            (std::vector<VkSemaphore>{timeline.getVkSemaphore(),
                                      wrapper.semaphore.getVkSemaphore(),
                                      presentation.getVkSemaphore()}));
  EXPECT_EQ(submitPayload.signalValues, (std::vector<uint64_t>{kSignalValue, 0, 0}));
}

TEST_F(VulkanImmediateCommandsTest, LegacyTimelineSubmitSignalsAndWaits) {
  auto& ctx = getVulkanContext();
  if (!ctx.features().has_VK_KHR_timeline_semaphore) {
    GTEST_SKIP() << "Timeline semaphores are unavailable on this device";
  }
  vulkan::VulkanImmediateCommands commands(ctx.vf_,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           true,
                                           false,
                                           "Legacy timeline test");
  vulkan::VulkanSemaphore timeline(ctx.vf_, ctx.getVkDevice(), uint64_t{0}, false, "Timeline test");
  vulkan::VulkanSemaphore binary(ctx.vf_, ctx.getVkDevice(), false, "Binary test");
  vulkan::VulkanSemaphore presentation(ctx.vf_, ctx.getVkDevice(), false, "Presentation test");
  const auto& producer = commands.acquire();
  ASSERT_TRUE(commands.signalSemaphore(timeline.getVkSemaphore(), 7));
  ASSERT_TRUE(commands.signalSemaphore(binary.getVkSemaphore(), 0));
  const auto producerHandle =
      submitForPresentation(commands, producer, presentation.getVkSemaphore());
  ASSERT_EQ(commands.wait(producerHandle, 5'000'000'000), VK_SUCCESS);
  uint64_t value = 0;
  ASSERT_EQ(
      ctx.vf_.vkGetSemaphoreCounterValueKHR(ctx.getVkDevice(), timeline.getVkSemaphore(), &value),
      VK_SUCCESS);
  EXPECT_EQ(value, 7u);

  const auto& consumer = commands.acquire();
  ASSERT_TRUE(commands.waitSemaphore(timeline.getVkSemaphore(), 7, VK_PIPELINE_STAGE_TRANSFER_BIT));
  ASSERT_TRUE(commands.waitSemaphore(binary.getVkSemaphore()));
  ASSERT_TRUE(commands.waitSemaphore(presentation.getVkSemaphore()));
  ASSERT_TRUE(commands.signalSemaphore(timeline.getVkSemaphore(), 11));
  const auto consumerHandle = commands.submit(consumer);
  ASSERT_EQ(commands.wait(consumerHandle, 5'000'000'000), VK_SUCCESS);
  ASSERT_EQ(
      ctx.vf_.vkGetSemaphoreCounterValueKHR(ctx.getVkDevice(), timeline.getVkSemaphore(), &value),
      VK_SUCCESS);
  EXPECT_EQ(value, 11u);
}

TEST_F(VulkanImmediateCommandsTest, InjectedWaitSemaphoreReachesSubmit) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  vulkan::VulkanSemaphore signal(ctx.vf_, ctx.getVkDevice(), false, "Injected wait test");
  const auto& first = ctx.immediate_->acquire();
  ASSERT_TRUE(ctx.immediate_->signalSemaphore(signal.getVkSemaphore(), 0));
  ctx.immediate_->submit(first);

  // Inject the real semaphore so the array-building path is exercised end-to-end through
  // vkQueueSubmit(2KHR) with a non-null handle, not just the counting/discard path.
  const auto& second = ctx.immediate_->acquire();
  EXPECT_TRUE(ctx.immediate_->waitSemaphore(signal.getVkSemaphore()));
  const auto handle = ctx.immediate_->submit(second);
  EXPECT_FALSE(handle.empty());
  EXPECT_EQ(ctx.immediate_->wait(handle), VK_SUCCESS);
}

TEST_F(VulkanImmediateCommandsTest, SubmitsChainWithSemaphoresByDefault) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanImmediateCommands commands(ctx.vf_,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Default chain test");
  const auto handle = commands.submit(commands.acquire());
  EXPECT_NE(lastSubmitSemaphore(commands), VK_NULL_HANDLE);
  EXPECT_EQ(commands.wait(handle), VK_SUCCESS);
}

TEST_F(VulkanImmediateCommandsTest, UnchainedSubmitsComplete) {
  auto& ctx = getVulkanContext();
  for (const bool useSync2 : {false, true}) {
    if (useSync2 && (!ctx.features().has_VK_KHR_timeline_semaphore ||
                     !ctx.features().has_VK_KHR_synchronization2)) {
      continue;
    }
    vulkan::VulkanImmediateCommands commands(ctx.vf_,
                                             ctx.getVkDevice(),
                                             ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                             false,
                                             useSync2,
                                             useSync2,
                                             "Unchained test",
                                             /*chainSubmitsWithSemaphores=*/false);
    // More submissions than command buffers, so every wrapper is recycled at least once.
    std::vector<vulkan::VulkanImmediateCommands::SubmitHandle> handles;
    for (uint32_t i = 0; i < 2 * vulkan::VulkanImmediateCommands::kMaxCommandBuffers; ++i) {
      handles.push_back(commands.submit(commands.acquire()));
      EXPECT_EQ(lastSubmitSemaphore(commands), VK_NULL_HANDLE);
    }
    EXPECT_EQ(commands.wait(handles.back()), VK_SUCCESS);
    commands.waitAll();
  }
}

TEST_F(VulkanImmediateCommandsTest, UnchainedSubmitWithInjectedWaitChainsToNext) {
  auto& ctx = getVulkanContext();
  for (const bool useSync2 : {false, true}) {
    if (useSync2 && (!ctx.features().has_VK_KHR_timeline_semaphore ||
                     !ctx.features().has_VK_KHR_synchronization2)) {
      continue;
    }
    vulkan::VulkanImmediateCommands commands(ctx.vf_,
                                             ctx.getVkDevice(),
                                             ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                             false,
                                             useSync2,
                                             useSync2,
                                             "Unchained injected wait test",
                                             /*chainSubmitsWithSemaphores=*/false);
    vulkan::VulkanSemaphore signal(ctx.vf_, ctx.getVkDevice(), false, "Injected wait test");

    const auto& producer = commands.acquire();
    ASSERT_TRUE(commands.signalSemaphore(signal.getVkSemaphore(), 0));
    commands.submit(producer);
    EXPECT_EQ(lastSubmitSemaphore(commands), VK_NULL_HANDLE);

    const auto& consumer = commands.acquire();
    ASSERT_TRUE(commands.waitSemaphore(signal.getVkSemaphore()));
    commands.submit(consumer);
    EXPECT_NE(lastSubmitSemaphore(commands), VK_NULL_HANDLE);

    // The next submission consumes the chain semaphore and does not extend the chain.
    const auto handle = commands.submit(commands.acquire());
    EXPECT_EQ(lastSubmitSemaphore(commands), VK_NULL_HANDLE);
    EXPECT_EQ(commands.wait(handle), VK_SUCCESS);
    commands.waitAll();
  }
}

TEST_F(VulkanImmediateCommandsTest, CachedFDDefaultIsNegativeOne) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  const auto& wrapper = ctx.immediate_->acquire();
  auto handle = ctx.immediate_->submit(wrapper);
  ASSERT_FALSE(handle.empty());

  EXPECT_EQ(ctx.immediate_->cachedFDFromSubmitHandle(handle), -1);

  ctx.immediate_->wait(handle);
}

TEST_F(VulkanImmediateCommandsTest, StoreFDAndCachedFDRoundTrip) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.immediate_, nullptr);

  const auto& wrapper = ctx.immediate_->acquire();
  auto handle = ctx.immediate_->submit(wrapper);
  ASSERT_FALSE(handle.empty());

  constexpr int kTestFD = 42;
  ctx.immediate_->storeFDInSubmitHandle(handle, kTestFD);
  EXPECT_EQ(ctx.immediate_->cachedFDFromSubmitHandle(handle), kTestFD);

  ctx.immediate_->wait(handle);
}

TEST_F(VulkanImmediateCommandsTest, DeviceLostOnSubmitIsReported) {
  auto& ctx = getVulkanContext();
  VulkanFunctionTable vf = ctx.vf_;
  vf.vkQueueSubmit = loseDeviceOnQueueSubmit;
  vf.vkQueueSubmit2KHR = loseDeviceOnQueueSubmit2;
  // Nothing reaches the queue, so the fences complete through the test function table.
  vf.vkWaitForFences = completeFenceWait;

  for (const bool synchronization2 : {false, true}) {
    vulkan::VulkanImmediateCommands commands(vf,
                                             ctx.getVkDevice(),
                                             ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                             false,
                                             synchronization2,
                                             synchronization2,
                                             "Device lost on submit test");
    EXPECT_FALSE(commands.isDeviceLost());
    commands.submit(commands.acquire());
    EXPECT_TRUE(commands.isDeviceLost()) << "synchronization2: " << synchronization2;
  }
}

TEST_F(VulkanImmediateCommandsTest, DeviceLostOnFenceWaitIsReported) {
  auto& ctx = getVulkanContext();
  VulkanFunctionTable vf = captureSubmits();
  vf.vkWaitForFences = loseDeviceOnFenceWait;
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Device lost on fence wait test");

  const auto handle = commands.submit(commands.acquire());
  EXPECT_FALSE(commands.isDeviceLost());
  EXPECT_EQ(commands.wait(handle), VK_ERROR_DEVICE_LOST);
  EXPECT_TRUE(commands.isDeviceLost());
}

TEST_F(VulkanImmediateCommandsTest, ARecycledHandleIsReadyAndHasNoFence) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Recycled handle test");
  const TracedWaitsSucceedAtScopeEnd waitsSucceed;
  const auto firstHandle = commands.submit(commands.acquire());
  EXPECT_NE(commands.getVkFenceFromSubmitHandle(firstHandle), VK_NULL_HANDLE);
  EXPECT_EQ(commands.wait(firstHandle), VK_SUCCESS);
  // The loop below acquires every row: a row that is not free here would make `acquire()` wait for
  // a row that never becomes free, so stop first.
  commands.waitAll();
  ASSERT_STREQ(rowState(commands, firstHandle.bufferIndex), "Free");

  // Acquire every row once more: the first row is reused with a later ordinal.
  vulkan::VulkanImmediateCommands::SubmitHandle reused;
  for (uint32_t i = 0; i != vulkan::VulkanImmediateCommands::kMaxCommandBuffers; ++i) {
    const auto handle = commands.submit(commands.acquire());
    if (handle.bufferIndex == firstHandle.bufferIndex) {
      reused = handle;
    }
  }
  ASSERT_FALSE(reused.empty());
  EXPECT_NE(reused.submitId, firstHandle.submitId);
  EXPECT_EQ(reused.bufferIndex, firstHandle.bufferIndex);
  EXPECT_TRUE(commands.isReady(firstHandle));
  EXPECT_EQ(commands.getVkFenceFromSubmitHandle(firstHandle), VK_NULL_HANDLE);
  EXPECT_NE(commands.getVkFenceFromSubmitHandle(reused), VK_NULL_HANDLE);
  commands.waitAll();
}

TEST_F(VulkanImmediateCommandsTest, RowStatesFollowTheLifecycle) {
  auto& ctx = getVulkanContext();
  // Fence waits are stubbed (nothing is queued): `VK_TIMEOUT` is an unsignaled fence.
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Row state test");
  const TracedWaitsSucceedAtScopeEnd waitsSucceed;
  gTrace.waitResult = VK_TIMEOUT;
  for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
    EXPECT_STREQ(rowState(commands, row), "Free");
  }

  const auto& first = commands.acquire();
  const uint32_t firstRow = first.handle.bufferIndex;
  EXPECT_STREQ(rowState(commands, firstRow), "Encoding");
  // The handle of a row that is still being recorded is not ready: its fence is not signaled.
  EXPECT_FALSE(commands.isReady(commands.getNextSubmitHandle()));
  const auto handle = commands.submit(first);
  EXPECT_STREQ(rowState(commands, firstRow), "Submitted");

  const auto& second = commands.acquire();
  const uint32_t secondRow = second.handle.bufferIndex;
  EXPECT_NE(secondRow, firstRow);
  EXPECT_STREQ(rowState(commands, secondRow), "Encoding");
  EXPECT_STREQ(rowState(commands, firstRow), "Submitted");
  commands.discard(second);
  EXPECT_STREQ(rowState(commands, secondRow), "Free");

  // `wait()` returns early when the fence already signaled and does not free the row then, so the
  // row is only guaranteed to be free after `waitAll()`.
  gTrace.waitResult = VK_SUCCESS;
  EXPECT_EQ(commands.wait(handle), VK_SUCCESS);
  commands.waitAll();
  EXPECT_STREQ(rowState(commands, firstRow), "Free");
}

TEST_F(VulkanImmediateCommandsTest, RowStatesAgreeWithTheCommandBufferWrappers) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Row agreement test");
  const TracedWaitsSucceedAtScopeEnd waitsSucceed;
  const auto expectAgreement = [&] {
    for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
      const auto& wrapper = wrapperOf(commands, row);
      const std::string state = rowState(commands, row);
      EXPECT_TRUE(rowAgrees(commands, row)) << "row " << row;
      EXPECT_EQ(state == "Encoding", wrapper.isEncoding) << "row " << row;
      EXPECT_EQ(state == "Free", wrapper.cmdBuf == VK_NULL_HANDLE) << "row " << row;
    }
  };
  expectAgreement();
  for (uint32_t i = 0; i != 5; ++i) {
    static_cast<void>(commands.submit(commands.acquire()));
    expectAgreement();
  }
  const auto& open = commands.acquire();
  expectAgreement();
  commands.discard(open);
  expectAgreement();
  commands.waitAll();
  expectAgreement();
}

TEST_F(VulkanImmediateCommandsTest, AReleasedRowIsFreeAndCanBeAcquiredAgain) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Released row test");
  const TracedWaitsSucceedAtScopeEnd waitsSucceed;
  // A recording that is no longer the current one is released: the current one stays current.
  const auto& released = commands.acquire();
  const uint32_t releasedRow = released.handle.bufferIndex;
  const auto& current = commands.acquire();
  const uint32_t currentRow = current.handle.bufferIndex;
  const auto currentHandle = commands.getNextSubmitHandle();
  commands.release(released);
  EXPECT_STREQ(rowState(commands, releasedRow), "Free");
  EXPECT_TRUE(rowAgrees(commands, releasedRow));
  EXPECT_EQ(commands.getNextSubmitHandle(), currentHandle);
  // The current recording is released too.
  commands.release(current);
  EXPECT_STREQ(rowState(commands, currentRow), "Free");
  EXPECT_TRUE(rowAgrees(commands, currentRow));
  EXPECT_TRUE(commands.getNextSubmitHandle().empty());

  // Every row is acquired again, the released ones included.
  std::set<uint32_t> rows;
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    const auto& wrapper = commands.acquire();
    rows.insert(wrapper.handle.bufferIndex);
    static_cast<void>(commands.submit(wrapper));
  }
  EXPECT_EQ(rows.size(), Commands::kMaxCommandBuffers);
  EXPECT_EQ(rows.count(releasedRow), 1u);
  EXPECT_EQ(rows.count(currentRow), 1u);
  commands.waitAll();
}

TEST_F(VulkanImmediateCommandsTest, SubmittingAStaleWrapperLeavesAFreeRowFree) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Stale wrapper test");
  const TracedWaitsSucceedAtScopeEnd waitsSucceed;
  const auto& wrapper = commands.acquire();
  const uint32_t row = wrapper.handle.bufferIndex;
  commands.discard(wrapper);
  ASSERT_STREQ(rowState(commands, row), "Free");

  // What `submit()` does to the row of a wrapper that is no longer encoding: nothing.
  moveRowToSubmitted(commands, row);
  EXPECT_STREQ(rowState(commands, row), "Free");
  EXPECT_EQ(wrapper.cmdBuf, VK_NULL_HANDLE);
  EXPECT_FALSE(wrapper.isEncoding);
  EXPECT_TRUE(rowAgrees(commands, row));
  // Whatever the row state is now, nothing was submitted on this row: free it before anything else
  // can wait for it (a failing expectation above must not hang the teardown).
  freeRow(commands, row);

  // A row that is encoding does move, when its recording is submitted.
  const auto& again = commands.acquire();
  ASSERT_EQ(again.handle.bufferIndex, row);
  const auto handle = commands.submit(again);
  EXPECT_STREQ(rowState(commands, row), "Submitted");
  EXPECT_FALSE(again.isEncoding);
  EXPECT_NE(again.cmdBuf, VK_NULL_HANDLE);
  EXPECT_TRUE(rowAgrees(commands, row));
  EXPECT_EQ(commands.wait(handle), VK_SUCCESS);
}

TEST_F(VulkanImmediateCommandsTest, DiscardedHandleAnswersAsItAlwaysDid) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Discarded handle test");
  const TracedWaitsSucceedAtScopeEnd waitsSucceed;
  gTrace.waitResult = VK_TIMEOUT; // the fence of the open recording is not signaled
  const auto& first = commands.acquire();
  const auto stale = commands.getNextSubmitHandle();
  commands.discard(first);
  const auto& second = commands.acquire();
  ASSERT_EQ(second.handle.bufferIndex, first.handle.bufferIndex);

  // Every acquisition now has its own handle, so the discarded recording and the one that took its
  // row no longer share one. The stale handle still answers as the open recording, as it did when
  // they shared it: not ready, not waitable, and its fence is the row's.
  EXPECT_NE(second.handle, stale);
  EXPECT_FALSE(commands.isReady(stale));
  EXPECT_EQ(commands.wait(stale, 0), VK_ERROR_UNKNOWN);
  EXPECT_EQ(commands.getVkFenceFromSubmitHandle(stale), second.fence.vkFence_);

  const auto handle = commands.submit(second);
  EXPECT_NE(handle, stale);
  gTrace.waitResult = VK_SUCCESS;
  EXPECT_EQ(commands.wait(handle), VK_SUCCESS);
  EXPECT_TRUE(commands.isReady(stale));
}

TEST_F(VulkanImmediateCommandsTest, WaitAllLeavesARecordingThatIsStillOpenAlone) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Nested recording test");
  gTrace.lines.clear();

  // A second recording starts while the first is open (a nested acquisition), and is submitted.
  const auto& outer = commands.acquire();
  const auto& inner = commands.acquire();
  static_cast<void>(commands.submit(inner));
  commands.waitAll();
  EXPECT_TRUE(outer.isEncoding);
  EXPECT_NE(outer.cmdBuf, VK_NULL_HANDLE);
  EXPECT_STREQ(rowState(commands, outer.handle.bufferIndex), "Encoding");
  EXPECT_STREQ(rowState(commands, inner.handle.bufferIndex), "Free");
  // Only the submitted row is waited on and reset; the open recording is not touched.
  const std::vector<std::string> expected = {
      "vkBeginCommandBuffer cb0 flags=1",
      "vkBeginCommandBuffer cb1 flags=1",
      "vkEndCommandBuffer cb1",
      "vkQueueSubmit cb1 fence1 waits=0 signals=1 signal:sem1",
      "vkWaitForFences all=1 timeout=max fence1",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkResetCommandBuffer cb1 flags=0",
      "vkResetFences fence1",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);

  gTrace.lines.clear();
  static_cast<void>(commands.submit(outer));
  commands.waitAll();
  const std::vector<std::string> expectedAfterwards = {
      "vkEndCommandBuffer cb0",
      "vkQueueSubmit cb0 fence0 waits=1 signals=1 wait:sem1 signal:sem0",
      "vkWaitForFences all=1 timeout=max fence0",
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
  };
  EXPECT_EQ(gTrace.lines, expectedAfterwards) << joinLines(gTrace.lines);
}

TEST_F(VulkanImmediateCommandsTest, TimedOutWaitKeepsTheRowSubmittedAndResetsNothing) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Timeout test");
  const auto handle = commands.submit(commands.acquire());
  gTrace.lines.clear();

  gTrace.waitResult = VK_TIMEOUT;
  EXPECT_EQ(commands.wait(handle, 0), VK_TIMEOUT);
  EXPECT_FALSE(commands.isReady(handle));
  // Only polls: no reset of the command buffer or of the fence. A zero-time `wait()` polls at
  // most once (see `wait()`; it used to poll twice), and so does `isReady()`.
  const std::vector<std::string> expected = {
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkWaitForFences all=1 timeout=0 fence0",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);

  // The row still holds the submission: its fence is still the handle's.
  EXPECT_NE(commands.getVkFenceFromSubmitHandle(handle), VK_NULL_HANDLE);
  EXPECT_STREQ(rowState(commands, handle.bufferIndex), "Submitted");
  gTrace.waitResult = VK_SUCCESS;
  commands.waitAll();
  EXPECT_STREQ(rowState(commands, handle.bufferIndex), "Free");
  EXPECT_TRUE(commands.isReady(handle));
}

TEST_F(VulkanImmediateCommandsTest, AcquireSubmitDiscardWaitCallTraceIsUnchanged) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Trace test");
  ASSERT_EQ(gTrace.lines.size(), kPoolCreationCalls);
  gTrace.lines.clear();

  const auto& first = commands.acquire();
  const auto firstHandle = commands.submit(first);
  const auto& second = commands.acquire();
  commands.discard(second);
  const auto& third = commands.acquire();
  const auto thirdHandle = commands.submit(third);
  EXPECT_TRUE(commands.isReady(firstHandle));
  EXPECT_EQ(commands.wait(thirdHandle), VK_SUCCESS);
  commands.waitAll();

  const std::vector<std::string> expected = {
      "vkBeginCommandBuffer cb0 flags=1",
      "vkEndCommandBuffer cb0",
      "vkQueueSubmit cb0 fence0 waits=0 signals=1 signal:sem0",
      "vkBeginCommandBuffer cb1 flags=1",
      "vkResetCommandBuffer cb1 flags=0",
      "vkBeginCommandBuffer cb1 flags=1",
      "vkEndCommandBuffer cb1",
      "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkWaitForFences all=1 timeout=max fence0 fence1",
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkResetCommandBuffer cb1 flags=0",
      "vkResetFences fence1",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);
}

TEST_F(VulkanImmediateCommandsTest, InjectedSemaphoreSubmitCallTraceIsUnchanged) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Trace test");
  gTrace.lines.clear();

  // Opaque non-null handles: the traced submit never reaches the device.
  // NOLINTBEGIN(performance-no-int-to-ptr)
  const auto waitSemaphore = reinterpret_cast<VkSemaphore>(static_cast<uintptr_t>(0x10));
  const auto signalSemaphore = reinterpret_cast<VkSemaphore>(static_cast<uintptr_t>(0x20));
  // NOLINTEND(performance-no-int-to-ptr)
  ASSERT_TRUE(commands.waitSemaphore(waitSemaphore, 0));
  ASSERT_TRUE(commands.signalSemaphore(signalSemaphore, 0));
  const auto handle = commands.submit(commands.acquire());
  EXPECT_EQ(commands.wait(handle), VK_SUCCESS);

  const std::vector<std::string> expected = {
      "vkBeginCommandBuffer cb0 flags=1",
      "vkEndCommandBuffer cb0",
      "vkQueueSubmit cb0 fence0 waits=1 signals=2 wait:sem? signal:sem? signal:sem0",
      "vkWaitForFences all=1 timeout=0 fence0",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);
}

// The call traces below were recorded on the pool before submissions were ordered. Where a
// scenario's calls or answers differ now, the test says why.

TEST_F(VulkanImmediateCommandsTest, SteadyStateOnePoolCallTrace) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Steady state trace test");
  gTrace.lines.clear();
  // Each frame submits once and asks whether the frame before last finished, as deferred tasks do.
  std::vector<vulkan::VulkanImmediateCommands::SubmitHandle> handles;
  for (uint32_t frame = 0; frame != 3; ++frame) {
    handles.push_back(commands.submit(commands.acquire()));
    if (frame >= 2) {
      EXPECT_TRUE(commands.isReady(handles[frame - 2]));
    }
  }
  EXPECT_EQ(commands.wait(handles.back()), VK_SUCCESS);
  commands.waitAll();
  const std::vector<std::string> expected = {
      "vkBeginCommandBuffer cb0 flags=1",
      "vkEndCommandBuffer cb0",
      "vkQueueSubmit cb0 fence0 waits=0 signals=1 signal:sem0",
      "vkBeginCommandBuffer cb1 flags=1",
      "vkEndCommandBuffer cb1",
      "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
      "vkBeginCommandBuffer cb2 flags=1",
      "vkEndCommandBuffer cb2",
      "vkQueueSubmit cb2 fence2 waits=1 signals=1 wait:sem1 signal:sem2",
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkWaitForFences all=1 timeout=0 fence2",
      "vkWaitForFences all=1 timeout=max fence0 fence1 fence2",
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkResetCommandBuffer cb1 flags=0",
      "vkResetFences fence1",
      "vkWaitForFences all=1 timeout=0 fence2",
      "vkResetCommandBuffer cb2 flags=0",
      "vkResetFences fence2",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);
}

TEST_F(VulkanImmediateCommandsTest, SteadyStateTwoPoolsCallTrace) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands main(vf,
                                       ctx.getVkDevice(),
                                       ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                       false,
                                       false,
                                       false,
                                       "Main trace test");
  vulkan::VulkanImmediateCommands staging(vf,
                                          ctx.getVkDevice(),
                                          ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                          false,
                                          false,
                                          false,
                                          "Staging trace test");
  gTrace.lines.clear();
  // A staging upload nested in an open frame recording, then the frame, twice.
  for (uint32_t frame = 0; frame != 2; ++frame) {
    const auto& frameRecording = main.acquire();
    const auto upload = staging.submit(staging.acquire());
    EXPECT_TRUE(staging.isReady(upload));
    const auto frameHandle = main.submit(frameRecording);
    EXPECT_TRUE(main.isReady(frameHandle));
  }
  main.waitAll();
  staging.waitAll();
  const std::vector<std::string> expected = {
      "vkBeginCommandBuffer cb0 flags=1",
      "vkBeginCommandBuffer cb32 flags=1",
      "vkEndCommandBuffer cb32",
      "vkQueueSubmit cb32 fence32 waits=0 signals=1 signal:sem32",
      "vkWaitForFences all=1 timeout=0 fence32",
      "vkEndCommandBuffer cb0",
      "vkQueueSubmit cb0 fence0 waits=0 signals=1 signal:sem0",
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkBeginCommandBuffer cb1 flags=1",
      "vkBeginCommandBuffer cb33 flags=1",
      "vkEndCommandBuffer cb33",
      "vkQueueSubmit cb33 fence33 waits=1 signals=1 wait:sem32 signal:sem33",
      "vkWaitForFences all=1 timeout=0 fence33",
      "vkEndCommandBuffer cb1",
      "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkWaitForFences all=1 timeout=max fence0 fence1",
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkResetCommandBuffer cb1 flags=0",
      "vkResetFences fence1",
      "vkWaitForFences all=1 timeout=max fence32 fence33",
      "vkWaitForFences all=1 timeout=0 fence32",
      "vkResetCommandBuffer cb32 flags=0",
      "vkResetFences fence32",
      "vkWaitForFences all=1 timeout=0 fence33",
      "vkResetCommandBuffer cb33 flags=0",
      "vkResetFences fence33",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);
}

TEST_F(VulkanImmediateCommandsTest, PendingWaitWithATimeoutCallTrace) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Timeout trace test");
  const auto handle = commands.submit(commands.acquire());
  gTrace.lines.clear();
  gTrace.waitResult = VK_TIMEOUT;
  EXPECT_EQ(commands.wait(handle, 5'000'000), VK_TIMEOUT);
  EXPECT_STREQ(rowState(commands, handle.bufferIndex), "Submitted");
  const std::vector<std::string> expected = {
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkWaitForFences all=1 timeout=5000000 fence0",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);
  gTrace.waitResult = VK_SUCCESS;
  commands.waitAll();
}

TEST_F(VulkanImmediateCommandsTest, DiscardAndReacquireBehindAnInFlightSubmissionCallTrace) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Discard trace test");
  gTrace.lines.clear();
  // A is in flight; B is discarded and its row is taken by C before anything else is submitted.
  gTrace.pollResult = VK_TIMEOUT;
  const auto handleA = commands.submit(commands.acquire());
  const auto& recordingB = commands.acquire();
  const auto handleB = commands.getNextSubmitHandle();
  commands.discard(recordingB);
  const auto& recordingC = commands.acquire();
  ASSERT_EQ(recordingC.handle.bufferIndex, handleB.bufferIndex);
  std::vector<std::string> answers;
  answers.push_back(std::string("B while C is open: ") + (commands.isReady(handleB) ? "1" : "0"));
  const auto handleC = commands.submit(recordingC);
  answers.push_back(std::string("B after C is submitted: ") +
                    (commands.isReady(handleB) ? "1" : "0"));
  gTrace.pollResult.reset();
  answers.push_back(std::string("B once A and C retire: ") +
                    (commands.isReady(handleB) ? "1" : "0"));
  EXPECT_EQ(commands.wait(handleA), VK_SUCCESS);
  EXPECT_EQ(commands.wait(handleC), VK_SUCCESS);
  commands.waitAll();
  const std::vector<std::string> expectedAnswers = {
      "B while C is open: 0",
      "B after C is submitted: 0",
      "B once A and C retire: 1",
  };
  EXPECT_EQ(answers, expectedAnswers) << joinLines(answers);
  // The answers are unchanged. The calls differ in two places: an open recording (C) is not polled,
  // and A's and C's waits make no call, because the poll that saw C's fence signal certified both.
  const std::vector<std::string> expected = {
      "vkBeginCommandBuffer cb0 flags=1",
      "vkEndCommandBuffer cb0",
      "vkQueueSubmit cb0 fence0 waits=0 signals=1 signal:sem0",
      "vkBeginCommandBuffer cb1 flags=1",
      "vkResetCommandBuffer cb1 flags=0",
      "vkBeginCommandBuffer cb1 flags=1",
      "vkEndCommandBuffer cb1",
      "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkWaitForFences all=1 timeout=max fence0 fence1",
      "vkWaitForFences all=1 timeout=0 fence0",
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
      "vkWaitForFences all=1 timeout=0 fence1",
      "vkResetCommandBuffer cb1 flags=0",
      "vkResetFences fence1",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);
}

TEST_F(VulkanImmediateCommandsTest, AReleasedRecordingWaitsForEverythingSubmittedBeforeIt) {
  const auto pool = makeFakePool("Released prefix test");
  const auto handleA = pool->submit(pool->acquire());
  // B is released after a later recording was acquired: it is not the current one.
  const auto& recordingB = pool->acquire();
  const auto handleB = pool->getNextSubmitHandle();
  const auto& recordingC = pool->acquire();
  const auto handleC = pool->getNextSubmitHandle();
  pool->release(recordingB);
  EXPECT_EQ(pool->getNextSubmitHandle(), handleC);

  // B never ran, but A, submitted before B was released, is still running: B's handle is not
  // ready although its row is free.
  EXPECT_FALSE(pool->isReady(handleB));
  EXPECT_EQ(pool->wait(handleB, 0), VK_TIMEOUT);
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(handleA));
  EXPECT_TRUE(pool->isReady(handleB));

  // The current recording is released and its row taken again before anything is submitted: its
  // handle also waits for the recording that took the row, as a discarded one does.
  pool->release(recordingC);
  const auto& recordingD = pool->acquire();
  const auto& recordingE = pool->acquire();
  ASSERT_EQ(recordingE.handle.bufferIndex, handleC.bufferIndex);
  EXPECT_FALSE(pool->isReady(handleC));
  static_cast<void>(pool->submit(recordingD));
  static_cast<void>(pool->submit(recordingE));
  completeSubmissions(2);
  EXPECT_FALSE(pool->isReady(handleC));
  completeSubmissions(3);
  EXPECT_TRUE(pool->isReady(handleC));
}

TEST_F(VulkanImmediateCommandsTest,
       AReleasedRecordingThatIsNotCurrentAnswersForTheRecordingThatTakesItsRow) {
  const auto pool = makeFakePool("Released non-current row reuse test");
  // B is released while a later recording, C, is the current one. Nothing was submitted before.
  const auto& recordingB = pool->acquire();
  const auto handleB = pool->getNextSubmitHandle();
  const auto& recordingC = pool->acquire();
  pool->release(recordingB);

  // Before the next submission, D takes B's row: B's handle answers for D, as a discarded handle
  // answers for the recording that takes its row, although B never ran and nothing ran before it.
  const auto& recordingD = pool->acquire();
  ASSERT_EQ(recordingD.handle.bufferIndex, handleB.bufferIndex);
  EXPECT_FALSE(pool->isReady(handleB));
  static_cast<void>(pool->submit(recordingC));
  static_cast<void>(pool->submit(recordingD));
  completeSubmissions(1);
  EXPECT_FALSE(pool->isReady(handleB));
  completeSubmissions(2);
  EXPECT_TRUE(pool->isReady(handleB));
}

TEST_F(VulkanImmediateCommandsTest,
       ARecordingThatSawASubmissionWhileOpenDoesNotAnswerForTheRecordingThatTakesItsRow) {
  const auto pool = makeFakePool("Submission while open test");
  // H is acquired; G, acquired after it, is submitted while H is open; then H is released and C
  // takes its row before anything else is submitted. H and C never shared a submit id (a
  // submission came between their acquisitions), so H answers for its own prefix, G, not for C.
  const auto& recordingH = pool->acquire();
  const auto handleH = pool->getNextSubmitHandle();
  const auto handleG = pool->submit(pool->acquire());
  pool->release(recordingH);
  const auto& recordingC = pool->acquire();
  ASSERT_EQ(recordingC.handle.bufferIndex, handleH.bufferIndex);

  // While G runs, H is not ready, and a bounded wait times out on G instead of failing on the open
  // recording C; no fence of C answers for H.
  EXPECT_FALSE(pool->isReady(handleH));
  gQueue.blockingWaitsComplete = false;
  EXPECT_EQ(pool->wait(handleH, 10'000'000), VK_TIMEOUT);
  gQueue.blockingWaitsComplete = true;
  EXPECT_EQ(pool->getVkFenceFromSubmitHandle(handleH), VK_NULL_HANDLE);
  // Once G finished, H is ready, while C is still open.
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(handleG));
  EXPECT_TRUE(pool->isReady(handleH));
  EXPECT_EQ(pool->wait(handleH, 10'000'000), VK_SUCCESS);
  pool->discard(recordingC);
}

TEST_F(VulkanImmediateCommandsTest, ADiscardedRecordingWaitsForEverythingSubmittedBeforeIt) {
  const auto pool = makeFakePool("Discarded prefix test");
  const auto handleA = pool->submit(pool->acquire());
  const auto& recordingB = pool->acquire();
  const auto handleB = pool->getNextSubmitHandle();
  pool->discard(recordingB);

  // B never ran, but A, submitted before B was closed, is still running: B's handle is not ready
  // although its row is free (the row used to make it ready at once).
  EXPECT_FALSE(pool->isReady(handleB));
  EXPECT_EQ(pool->wait(handleB, 0), VK_TIMEOUT);

  // B's row is taken again before anything is submitted: B also waits for the recording that took
  // the row, as its handle used to be that recording's.
  const auto& recordingC = pool->acquire();
  ASSERT_EQ(recordingC.handle.bufferIndex, handleB.bufferIndex);
  EXPECT_FALSE(pool->isReady(handleB));
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(handleA));
  EXPECT_FALSE(pool->isReady(handleB));
  static_cast<void>(pool->submit(recordingC));
  EXPECT_FALSE(pool->isReady(handleB));
  completeSubmissions(2);
  EXPECT_TRUE(pool->isReady(handleB));

  // A recording discarded and reacquired after a submission waits only for its own prefix.
  const auto handleD = pool->submit(pool->acquire());
  const auto& recordingX = pool->acquire();
  const auto& recordingE = pool->acquire();
  const auto handleE = pool->getNextSubmitHandle();
  pool->discard(recordingE);
  static_cast<void>(pool->submit(recordingX));
  const auto& reuse = pool->acquire();
  EXPECT_EQ(reuse.handle.bufferIndex, handleE.bufferIndex);
  EXPECT_FALSE(pool->isReady(handleE));
  completeSubmissions(3);
  EXPECT_TRUE(pool->isReady(handleD));
  EXPECT_TRUE(pool->isReady(handleE));
  pool->discard(reuse);
}

TEST_F(VulkanImmediateCommandsTest,
       HandlelessDeferredTaskOfADiscardedRecordingWaitsForEarlierWork) {
  auto& ctx = getVulkanContext();
  // The context's command buffers come from a pool on the fake queue for the duration of the test,
  // so that A stays in flight until the test completes it.
  const FakeContextPool fakePool(*this, ctx.immediate_, "Deferred task test");
  const auto queue = iglDev_->createCommandQueue({}, nullptr);
  ASSERT_NE(queue, nullptr);
  bool ran = false;
  {
    const auto handleA = ctx.immediate_->submit(ctx.immediate_->acquire());
    const auto& recordingB = ctx.immediate_->acquire();
    ctx.deferredTask(std::packaged_task<void()>([&ran] { ran = true; }));
    ctx.immediate_->discard(recordingB);

    // A frame submitted through the command queue runs the deferred tasks that are ready.
    const auto frame = queue->createCommandBuffer({}, nullptr);
    ASSERT_NE(frame, nullptr);
    queue->submit(*frame);
    EXPECT_FALSE(ran);

    completeSubmissions(1);
    EXPECT_TRUE(ctx.immediate_->isReady(handleA));
    const auto nextFrame = queue->createCommandBuffer({}, nullptr);
    ASSERT_NE(nextFrame, nullptr);
    queue->submit(*nextFrame);
    // The first frame took B's row before anything else was submitted, so B also waits for it.
    EXPECT_FALSE(ran);

    completeSubmissions(gQueue.submitted.size());
    const auto lastFrame = queue->createCommandBuffer({}, nullptr);
    ASSERT_NE(lastFrame, nullptr);
    queue->submit(*lastFrame);
    EXPECT_TRUE(ran);
  }
}

TEST_F(VulkanImmediateCommandsTest, ReadinessFollowsSubmissionsWhileARecordingStaysOpen) {
  const auto pool = makeFakePool("Held recording test");

  // An independent model: the submission count each submitted handle waits for, by row, checked
  // against what the fake queue has completed.
  std::map<uint32_t, std::pair<Commands::SubmitHandle, size_t>> submittedByRow;
  size_t numSubmitted = 0;
  const auto submit = [&](const Commands::CommandBufferWrapper& wrapper) {
    const auto handle = pool->submit(wrapper);
    submittedByRow[handle.bufferIndex] = {handle, ++numSubmitted};
  };
  const auto expectModel = [&](uint32_t cycle) {
    for (const auto& [row, entry] : submittedByRow) {
      if (wrapperOf(*pool, row).handle != entry.first) {
        continue; // the row was acquired again
      }
      EXPECT_EQ(pool->isReady(entry.first), gQueue.completed >= entry.second)
          << "cycle " << cycle << " row " << row;
    }
  };

  const auto& held = pool->acquire();
  const auto heldHandle = pool->getNextSubmitHandle();
  const Commands::CommandBufferWrapper* outer = nullptr;
  const Commands::CommandBufferWrapper* inner = nullptr;
  for (uint32_t cycle = 0; cycle != 3 * Commands::kMaxCommandBuffers + 5; ++cycle) {
    if (cycle % 11 == 3) {
      // Two more recordings stay open for a while; the later one is submitted first.
      outer = &pool->acquire();
      inner = &pool->acquire();
    } else if (cycle % 11 == 6 && inner) {
      submit(*inner);
      inner = nullptr;
    } else if (cycle % 11 == 8 && outer) {
      submit(*outer);
      outer = nullptr;
    } else if (cycle % 7 == 5) {
      pool->discard(pool->acquire());
    } else {
      submit(pool->acquire());
    }
    if (cycle % 5 == 4) {
      // Everything but the newest submission completes.
      completeSubmissions(numSubmitted - 1);
    }
    expectModel(cycle);
    EXPECT_FALSE(pool->isReady(heldHandle)) << "cycle " << cycle;
  }
  EXPECT_GT(numSubmitted, 2 * Commands::kMaxCommandBuffers);

  if (inner) {
    submit(*inner);
  }
  if (outer) {
    submit(*outer);
  }
  submit(held);
  expectModel(1000);
  EXPECT_FALSE(pool->isReady(heldHandle));
  completeSubmissions(numSubmitted);
  expectModel(1001);
  EXPECT_TRUE(pool->isReady(heldHandle));
}

TEST_F(VulkanImmediateCommandsTest, OverlappingRecordingsRetireInSubmissionOrder) {
  const auto pool = makeFakePool("Overlap test");
  const auto& first = pool->acquire();
  const auto& second = pool->acquire();
  // The recording acquired second submits first: its handle carries the higher submit id, but it
  // is the first submission.
  const auto secondHandle = pool->submit(second);
  const auto firstHandle = pool->submit(first);
  EXPECT_GT(secondHandle.submitId, firstHandle.submitId);
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(secondHandle));
  EXPECT_FALSE(pool->isReady(firstHandle));
  completeSubmissions(2);
  EXPECT_TRUE(pool->isReady(firstHandle));
}

TEST_F(VulkanImmediateCommandsTest, SubmitIdIsTheLowBitsOfTheAcquisitionOrdinalAndSkipsZero) {
  const auto pool = makeFakePool("Ordinal wrap test");
  setAcquisitionCounter(*pool, 0xfffffffeull);
  const auto& beforeWrap = pool->acquire();
  EXPECT_EQ(beforeWrap.handle.submitId, 0xffffffffu);
  const auto& afterWrap = pool->acquire();
  EXPECT_EQ(afterWrap.handle.submitId, 1u);
  EXPECT_EQ(acquisitionOf(*pool, afterWrap.handle.bufferIndex), 0x100000001ull);
  // Readiness follows the submissions across the wrap, whichever recording submits first.
  const auto afterHandle = pool->submit(afterWrap);
  const auto beforeHandle = pool->submit(beforeWrap);
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(afterHandle));
  EXPECT_FALSE(pool->isReady(beforeHandle));
  completeSubmissions(2);
  EXPECT_TRUE(pool->isReady(beforeHandle));

  // A recording discarded just before the wrap answers as the recording that took its row after
  // the wrap.
  setAcquisitionCounter(*pool, 0x1fffffffeull);
  const auto& discarded = pool->acquire();
  const auto discardedHandle = pool->getNextSubmitHandle();
  EXPECT_EQ(discardedHandle.submitId, 0xffffffffu);
  pool->discard(discarded);
  const auto& taker = pool->acquire();
  ASSERT_EQ(taker.handle.bufferIndex, discardedHandle.bufferIndex);
  EXPECT_EQ(taker.handle.submitId, 1u);
  EXPECT_FALSE(pool->isReady(discardedHandle));
  const auto takerHandle = pool->submit(taker);
  EXPECT_FALSE(pool->isReady(discardedHandle));
  completeSubmissions(3);
  EXPECT_TRUE(pool->isReady(takerHandle));
  EXPECT_TRUE(pool->isReady(discardedHandle));
}

TEST_F(VulkanImmediateCommandsTest, ZeroTimeQueriesOnlyPoll) {
  const auto pool = makeFakePool("Zero-time query test");
  const auto handle = pool->submit(pool->acquire());
  EXPECT_FALSE(pool->isReady(handle));
  EXPECT_EQ(pool->wait(handle, 0), VK_TIMEOUT);
  EXPECT_EQ(gQueue.polls, 2u);
  EXPECT_EQ(gQueue.blockingWaits, 0u);
  EXPECT_STREQ(rowState(*pool, handle.bufferIndex), "Submitted");
  // An open recording is not ready, and nothing is polled for it.
  const auto& open = pool->acquire();
  EXPECT_FALSE(pool->isReady(pool->getNextSubmitHandle()));
  EXPECT_EQ(gQueue.polls, 2u);
  pool->discard(open);
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(handle));
  EXPECT_EQ(retiredThrough(*pool), 1u);
}

TEST_F(VulkanImmediateCommandsTest, AZeroTimeQueryThatSeesDeviceLossCertifies) {
  const auto pool = makeFakePool("Zero-time device loss test");
  const auto handle = pool->submit(pool->acquire());
  gQueue.waitOverride = VK_ERROR_DEVICE_LOST;
  EXPECT_TRUE(pool->isReady(handle));
  EXPECT_TRUE(pool->isDeviceLost());
  EXPECT_EQ(gQueue.polls, 1u);
  EXPECT_EQ(pool->wait(handle, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(gQueue.polls, 1u);
  gQueue.waitOverride.reset();
  completeSubmissions(1);
}

TEST_F(VulkanImmediateCommandsTest, AWaitThatTimesOutKeepsTheSubmissionAndResetsNothing) {
  const auto pool = makeFakePool("Timeout property test");
  const auto handle = pool->submit(pool->acquire());
  gQueue.blockingWaitsComplete = false;
  EXPECT_EQ(pool->wait(handle, 1'000'000), VK_TIMEOUT);
  EXPECT_STREQ(rowState(*pool, handle.bufferIndex), "Submitted");
  EXPECT_EQ(retiredThrough(*pool), 0u);
  EXPECT_NE(pool->getVkFenceFromSubmitHandle(handle), VK_NULL_HANDLE);
  gQueue.blockingWaitsComplete = true;
  pool->waitAll();
  EXPECT_STREQ(rowState(*pool, handle.bufferIndex), "Free");
}

TEST_F(VulkanImmediateCommandsTest, ACachedDeviceLossAloneCertifiesNothing) {
  const auto pool = makeFakePool("Device loss latch test");
  const auto pending = pool->submit(pool->acquire());
  gQueue.submitResults = {VK_ERROR_DEVICE_LOST};
  const auto lostSubmit = pool->submit(pool->acquire());
  ASSERT_TRUE(pool->isDeviceLost());
  // No fence answered device loss: the cached loss proves nothing.
  EXPECT_FALSE(pool->isReady(pending));
  EXPECT_FALSE(pool->isReady(lostSubmit));
  gQueue.blockingWaitsComplete = false;
  EXPECT_EQ(pool->wait(pending, 1'000'000), VK_TIMEOUT);
  EXPECT_EQ(retiredThrough(*pool), 0u);
  gQueue.blockingWaitsComplete = true;
  completeSubmissions(2);
  EXPECT_TRUE(pool->isReady(pending));
  EXPECT_TRUE(pool->isReady(lostSubmit));
}

TEST_F(
    VulkanImmediateCommandsTest,
    AnExhaustedDescriptorPoolIsNotReusedBeforeTheWorkSubmittedBeforeADiscardedRecordingFinishes) {
  auto& ctx = getVulkanContext();
  const auto dispatch = makeStorageBufferDispatch();
  ASSERT_NE(dispatch.pipeline, nullptr);
  const DescriptorPoolProbe probe(ctx.vf_);
  // The context's command buffers are submitted to the fake queue, so that A stays in flight until
  // the test completes it.
  const FakeContextPool fakePool(*this, ctx.immediate_, "Descriptor pool test");
  const auto queue = iglDev_->createCommandQueue({}, nullptr);
  ASSERT_NE(queue, nullptr);
  // A descriptor pool of the arena holds this many sets.
  constexpr uint32_t kSetsPerPool = 64;

  // O stays open throughout.
  const auto recordingO = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(recordingO, nullptr);
  // A uses every set of the first pool and stays in flight.
  const auto recordingA = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(recordingA, nullptr);
  recordDispatches(*recordingA, dispatch, kSetsPerPool);
  const VkDescriptorPool firstPool = gProbe.lastStorageBufferPool;
  ASSERT_NE(firstPool, VK_NULL_HANDLE);
  queue->submit(*recordingA);
  // B's first set moves the arena to a second pool and parks the first one under B's handle. Then
  // B is discarded.
  const auto recordingB = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(recordingB, nullptr);
  recordDispatches(*recordingB, dispatch, 1);
  ASSERT_NE(gProbe.lastStorageBufferPool, firstPool);
  discardRecording(*recordingB);

  // O uses up the second pool. A may still read the first pool's sets: a third pool is used.
  recordDispatches(*recordingO, dispatch, kSetsPerPool);
  EXPECT_NE(gProbe.lastStorageBufferPool, firstPool);
  // Once A finished, the next switch reuses the first pool.
  completeSubmissions(gQueue.submitted.size());
  recordDispatches(*recordingO, dispatch, kSetsPerPool);
  EXPECT_EQ(gProbe.lastStorageBufferPool, firstPool);
  queue->submit(*recordingO);
}

TEST_F(VulkanImmediateCommandsTest,
       AnExhaustedDescriptorPoolParkedUnderADiscardedHandleWaitsForTheRecordingThatTookItsRow) {
  auto& ctx = getVulkanContext();
  const auto dispatch = makeStorageBufferDispatch();
  ASSERT_NE(dispatch.pipeline, nullptr);
  const DescriptorPoolProbe probe(ctx.vf_);
  const FakeContextPool fakePool(*this, ctx.immediate_, "Descriptor pool alias test");
  const auto queue = iglDev_->createCommandQueue({}, nullptr);
  ASSERT_NE(queue, nullptr);
  constexpr uint32_t kSetsPerPool = 64;

  const auto recordingA = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(recordingA, nullptr);
  recordDispatches(*recordingA, dispatch, kSetsPerPool);
  const VkDescriptorPool firstPool = gProbe.lastStorageBufferPool;
  ASSERT_NE(firstPool, VK_NULL_HANDLE);
  queue->submit(*recordingA);
  const auto recordingB = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(recordingB, nullptr);
  recordDispatches(*recordingB, dispatch, 1);
  ASSERT_NE(gProbe.lastStorageBufferPool, firstPool);
  const auto handleB = static_cast<vulkan::CommandBuffer&>(*recordingB).getNextSubmitHandle();
  discardRecording(*recordingB);

  // C takes B's row before anything else is submitted, so B's handle answers as C, as it did when
  // the two shared one handle.
  const auto recordingC = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(recordingC, nullptr);
  ASSERT_EQ(static_cast<vulkan::CommandBuffer&>(*recordingC).getNextSubmitHandle().bufferIndex,
            handleB.bufferIndex);
  recordDispatches(*recordingC, dispatch, kSetsPerPool);
  EXPECT_NE(gProbe.lastStorageBufferPool, firstPool);
  // A finished, but C is still open.
  completeSubmissions(gQueue.submitted.size());
  recordDispatches(*recordingC, dispatch, kSetsPerPool);
  EXPECT_NE(gProbe.lastStorageBufferPool, firstPool);
  // Once C finished, the first pool is reused.
  queue->submit(*recordingC);
  completeSubmissions(gQueue.submitted.size());
  const auto recordingD = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(recordingD, nullptr);
  recordDispatches(*recordingD, dispatch, kSetsPerPool);
  EXPECT_EQ(gProbe.lastStorageBufferPool, firstPool);
  queue->submit(*recordingD);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
