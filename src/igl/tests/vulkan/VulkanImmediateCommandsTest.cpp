/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <igl/vulkan/VulkanImmediateCommands.h>

#include "../util/TestDevice.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdarg>
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
#include <igl/Framebuffer.h>
#include <igl/ShaderCreator.h>
#include <igl/Timer.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanStagingDevice.h>
#include <igl/vulkan/VulkanTexture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

#if IGL_PLATFORM_ANDROID
#include <cstring>
#include <unistd.h>
#include <igl/vulkan/HWDevice.h>
#endif // IGL_PLATFORM_ANDROID

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
  /// What the next traced `vkQueueSubmit()` calls answer, front first (then success).
  std::vector<VkResult> submitResults;
  /// What the next traced `vkEndCommandBuffer()` answers instead of ending the command buffer.
  VkResult endResult = VK_SUCCESS;
  std::vector<std::string> lines;
  std::map<uint64_t, size_t> commandBuffers;
  std::map<uint64_t, size_t> fences;
  std::map<uint64_t, size_t> semaphores;
};
thread_local CallTrace gTrace;

/// A bound for the waits of tests that run on the device.
constexpr uint64_t kTestTimeout = 10'000'000'000ull;

template<typename T>
uint64_t handleKey(T handle) {
  if constexpr (std::is_pointer_v<T>) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
  } else {
    return static_cast<uint64_t>(handle);
  }
}

/// An opaque non-null Vulkan handle for submissions that never reach the device. A non-dispatchable
/// handle is a 64-bit integer on 32-bit platforms.
template<typename T>
T opaqueHandle(uint64_t value) {
  if constexpr (std::is_pointer_v<T>) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<T>(static_cast<uintptr_t>(value));
  } else {
    return static_cast<T>(value);
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
  if (gTrace.endResult != VK_SUCCESS) {
    return std::exchange(gTrace.endResult, VK_SUCCESS);
  }
  return gTrace.real.vkEndCommandBuffer(cb);
}

/// The traced names of the command buffers of a batch, joined with `+`: `none` for a batch without
/// one.
template<typename T>
std::string batchCommandBuffer(uint32_t count, const T* commandBuffers) {
  if (count == 0) {
    return "none";
  }
  std::string names = handleName(gTrace.commandBuffers, "cb", commandBuffers[0]);
  for (uint32_t i = 1; i < count; ++i) {
    names += "+" + handleName(gTrace.commandBuffers, "cb", commandBuffers[i]);
  }
  return names;
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
  if (count == 0) {
    record("vkQueueSubmit count=0 " + handleName(gTrace.fences, "fence", fence));
  }
  for (uint32_t i = 0; i < count; ++i) {
    std::string line =
        "vkQueueSubmit " +
        batchCommandBuffer(submits[i].commandBufferCount, submits[i].pCommandBuffers) + " " +
        handleName(gTrace.fences, "fence", fence) +
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
  if (gTrace.submitResults.empty()) {
    return VK_SUCCESS;
  }
  const VkResult result = gTrace.submitResults.front();
  gTrace.submitResults.erase(gTrace.submitResults.begin());
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL traceQueueSubmit2(VkQueue /*queue*/,
                                                 uint32_t count,
                                                 const VkSubmitInfo2* submits,
                                                 VkFence fence) {
  for (uint32_t i = 0; i < count; ++i) {
    std::string line =
        "vkQueueSubmit2KHR " +
        (submits[i].commandBufferInfoCount == 0
             ? std::string("none")
             : handleName(
                   gTrace.commandBuffers, "cb", submits[i].pCommandBufferInfos[0].commandBuffer)) +
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
  /// Fences of the submissions, in submission order (a checkpoint has no command buffer).
  std::vector<VkFence> submitted;
  /// The submissions before this index have completed.
  size_t completed = 0;
  /// The submission index that each fence's pending or completed signal belongs to.
  std::map<uint64_t, size_t> fenceSubmission;
  /// When set, a blocking wait completes every submission up to the one it waits for.
  bool blockingWaitsComplete = true;
  /// When set too, a blocking wait completes every submission, as if later work finished while
  /// it waited.
  bool blockingWaitsCompleteEverything = false;
  /// What the next submissions answer, front first (then success).
  std::vector<VkResult> submitResults;
  /// When set, what every fence wait answers instead (e.g. device loss).
  std::optional<VkResult> waitOverride;
  /// What the next blocking fence waits answer, front first (then as usual).
  std::vector<VkResult> blockingWaitResults;
  /// What the next zero-time fence polls answer, front first (then as usual).
  std::vector<VkResult> pollResults;
  /// What the next `vkEndCommandBuffer()` answers (then success).
  VkResult endResult = VK_SUCCESS;
  /// What `vkGetFenceFdKHR()` answers.
  VkResult exportResult = VK_SUCCESS;
  /// When set, the next `vkCreateFence()` fails with this result and creates nothing.
  std::optional<VkResult> createFenceResult;
  /// What the next `vkResetFences()` calls answer, front first (then success). A failed reset
  /// leaves the fence as it was.
  std::vector<VkResult> resetResults;
  /// What the next `vkResetCommandBuffer()` calls answer, front first (then the driver's answer).
  /// A failed reset leaves the command buffer as it was.
  std::vector<VkResult> resetCommandBufferResults;
  /// How many fences were created exportable. The fake queue creates them without the export
  /// information, so no driver support for exports is needed.
  size_t exportableFenceCreates = 0;
  /// Blocking waits answered by `waitOverride`, or that did not complete.
  size_t unansweredBlockingWaits = 0;
  size_t polls = 0;
  size_t blockingWaits = 0;
  size_t checkpoints = 0;
  size_t exports = 0;
  /// Every `vkQueueSubmit()` call, including those that fail.
  size_t submitCalls = 0;
  std::vector<uint64_t> polledFences;
  /// Fences exported and not reset or submitted since.
  std::set<uint64_t> exportedFences;
  /// The back-off fences of the pools on the fake queue, with their pools (see `makeFakePool()`).
  std::map<uint64_t, void*> backoffFences;
  /// Called with the pool of a back-off fence once the test failed for too many back-offs, to end
  /// that pool's retries (see `VulkanImmediateCommandsTest::endRetries()`).
  void (*endRetries)(void* IGL_NONNULL pool) = nullptr;
  /// The time limit of every wait on a back-off fence.
  std::vector<uint64_t> backoffWaits;
  /// What the next waits on a back-off fence answer, front first (then `VK_TIMEOUT`).
  std::vector<VkResult> backoffResults;
};
thread_local FakeQueue gQueue;

/// The functions that raised the debug aborts while a `DebugAborts` is in scope.
thread_local std::vector<std::string> gDebugAborts;

void recordDebugAbort(const char* IGL_NONNULL /*category*/,
                      const char* IGL_NONNULL /*reason*/,
                      const char* IGL_NONNULL /*file*/,
                      const char* IGL_NONNULL func,
                      int /*line*/,
                      const char* IGL_NONNULL /*format*/,
                      va_list /*ap*/) {
  gDebugAborts.emplace_back(func);
}

/// Records the debug aborts raised while it is in scope, instead of handling them.
class DebugAborts {
 public:
  DebugAborts() : saved_(iglGetDebugAbortListener()) {
    gDebugAborts.clear();
    iglSetDebugAbortListener(recordDebugAbort);
  }
  DebugAborts(const DebugAborts&) = delete;
  DebugAborts& operator=(const DebugAborts&) = delete;
  ~DebugAborts() {
    iglSetDebugAbortListener(saved_);
  }
  /// How many were raised in a function whose name contains `func`.
  [[nodiscard]] size_t in(std::string_view func) const {
    return static_cast<size_t>(
        std::count_if(gDebugAborts.begin(), gDebugAborts.end(), [func](const std::string& name) {
          return name.find(func) != std::string::npos;
        }));
  }

 private:
  IGLErrorHandlerFunc IGL_NULLABLE saved_;
};

/// How many debug aborts one failure raises: none in builds without them.
#if IGL_DEBUG_ABORT_ENABLED
constexpr size_t kDebugAbortsPerFailure = 1;
#else
constexpr size_t kDebugAbortsPerFailure = 0;
#endif // IGL_DEBUG_ABORT_ENABLED

/// The time limits of `n` back-offs in one call: 1 ms, then twice as long each time up to 16 ms.
std::vector<uint64_t> backoffs(size_t n) {
  std::vector<uint64_t> limits;
  for (size_t i = 0; i != n; ++i) {
    limits.push_back(uint64_t{1'000'000} << std::min<size_t>(i, 4));
  }
  return limits;
}
/// More blocking waits than this that never complete mean that a pool retries forever: the fake
/// queue fails the test, and lets the waits complete.
constexpr size_t kMaxUnansweredBlockingWaits = 1000;
/// More back-offs than this mean that a pool retries forever: the fake queue fails the test, lets
/// everything submitted complete and the next submissions succeed, and ends the pool's retries,
/// so that the test ends instead of hanging.
constexpr size_t kMaxBackoffs = 1000;

bool fenceSignaled(VkFence fence) {
  const auto it = gQueue.fenceSubmission.find(handleKey(fence));
  return it != gQueue.fenceSubmission.end() && it->second < gQueue.completed;
}

VKAPI_ATTR VkResult VKAPI_CALL fakeQueueSubmit(VkQueue /*queue*/,
                                               uint32_t count,
                                               const VkSubmitInfo* /*submits*/,
                                               VkFence fence) {
  VkResult result = VK_SUCCESS;
  ++gQueue.submitCalls;
  if (!gQueue.submitResults.empty()) {
    result = gQueue.submitResults.front();
    gQueue.submitResults.erase(gQueue.submitResults.begin());
  }
  if (count == 0) {
    ++gQueue.checkpoints;
  }
  gQueue.exportedFences.erase(handleKey(fence));
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
  const auto backoffFence = count == 1 ? gQueue.backoffFences.find(handleKey(fences[0]))
                                       : gQueue.backoffFences.end();
  if (backoffFence != gQueue.backoffFences.end()) {
    gQueue.backoffWaits.push_back(timeout);
    if (!gQueue.backoffResults.empty()) {
      const VkResult result = gQueue.backoffResults.front();
      gQueue.backoffResults.erase(gQueue.backoffResults.begin());
      return result;
    }
    if (gQueue.backoffWaits.size() > kMaxBackoffs) {
      if (gQueue.backoffWaits.size() == kMaxBackoffs + 1) {
        ADD_FAILURE() << "More than " << kMaxBackoffs << " back-offs";
        gQueue.submitResults.clear();
        gQueue.completed = gQueue.submitted.size();
        gQueue.blockingWaitsComplete = true;
      }
      if (gQueue.endRetries != nullptr) {
        gQueue.endRetries(backoffFence->second);
      }
    }
    return VK_TIMEOUT;
  }
  if (timeout == 0) {
    ++gQueue.polls;
    gQueue.polledFences.push_back(handleKey(fences[0]));
    if (!gQueue.pollResults.empty()) {
      const VkResult result = gQueue.pollResults.front();
      gQueue.pollResults.erase(gQueue.pollResults.begin());
      return result;
    }
  } else {
    ++gQueue.blockingWaits;
    if (!gQueue.blockingWaitResults.empty()) {
      const VkResult result = gQueue.blockingWaitResults.front();
      gQueue.blockingWaitResults.erase(gQueue.blockingWaitResults.begin());
      if (result != VK_TIMEOUT || timeout != UINT64_MAX) {
        return result;
      }
      ADD_FAILURE() << "A wait without a time limit cannot time out";
    }
  }
  if (gQueue.waitOverride && *gQueue.waitOverride == VK_TIMEOUT && timeout == UINT64_MAX) {
    // A conforming driver blocks instead: fail the test, and answer as the rest of the fake does.
    ADD_FAILURE() << "A wait without a time limit cannot time out";
  } else if (gQueue.waitOverride) {
    // A pool that retries forever on an answer that never changes would hang the test: fail it,
    // and let the waits complete.
    if (timeout == 0 || ++gQueue.unansweredBlockingWaits <= kMaxUnansweredBlockingWaits) {
      return *gQueue.waitOverride;
    }
    ADD_FAILURE() << "More than " << kMaxUnansweredBlockingWaits << " blocking waits answered "
                  << *gQueue.waitOverride;
    gQueue.waitOverride.reset();
  }
  bool all = true;
  for (uint32_t i = 0; i < count; ++i) {
    if (gQueue.exportedFences.count(handleKey(fences[i])) != 0) {
      // The export reset the fence: a real driver would never signal it. Fail the test, and answer
      // as if it were signaled so that the pool cannot spin on it.
      ADD_FAILURE() << "A wait or poll on an exported fence";
      continue;
    }
    if (!fenceSignaled(fences[i])) {
      const auto it = gQueue.fenceSubmission.find(handleKey(fences[i]));
      if (timeout == UINT64_MAX && it == gQueue.fenceSubmission.end()) {
        // A real driver would block forever: fail the test, and answer as if the fence were
        // signaled so that the pool cannot spin on it.
        ADD_FAILURE() << "Unbounded wait on a fence that has no pending signal";
        continue;
      }
      if (timeout == UINT64_MAX && !gQueue.blockingWaitsComplete) {
        // A conforming driver would block until the submission completes, which this test never
        // lets happen: fail the test, and let it complete so that the pool cannot spin on it.
        ADD_FAILURE() << "A wait without a time limit on a submission that does not complete";
        gQueue.completed = std::max(gQueue.completed, it->second + 1);
      } else if (timeout != 0 && gQueue.blockingWaitsComplete &&
                 it != gQueue.fenceSubmission.end()) {
        gQueue.completed = gQueue.blockingWaitsCompleteEverything
                               ? gQueue.submitted.size()
                               : std::max(gQueue.completed, it->second + 1);
      } else {
        all = false;
      }
    }
  }
  if (!all && timeout != 0 && ++gQueue.unansweredBlockingWaits > kMaxUnansweredBlockingWaits) {
    ADD_FAILURE() << "More than " << kMaxUnansweredBlockingWaits
                  << " blocking waits did not complete";
    gQueue.blockingWaitsComplete = true;
  }
  return all ? VK_SUCCESS : VK_TIMEOUT;
}

VKAPI_ATTR VkResult VKAPI_CALL fakeResetFences(VkDevice /*device*/,
                                               uint32_t count,
                                               const VkFence* fences) {
  if (!gQueue.resetResults.empty()) {
    const VkResult result = gQueue.resetResults.front();
    gQueue.resetResults.erase(gQueue.resetResults.begin());
    if (result != VK_SUCCESS) {
      return result;
    }
  }
  for (uint32_t i = 0; i < count; ++i) {
    gQueue.fenceSubmission.erase(handleKey(fences[i]));
    gQueue.exportedFences.erase(handleKey(fences[i]));
  }
  return VK_SUCCESS;
}

#if defined(VK_KHR_external_fence_fd)
VKAPI_ATTR VkResult VKAPI_CALL fakeGetFenceFd(VkDevice /*device*/,
                                              const VkFenceGetFdInfoKHR* info,
                                              int* fd) {
  ++gQueue.exports;
  if (gQueue.exportResult != VK_SUCCESS) {
    return gQueue.exportResult;
  }
  // Copy transference: the export resets the fence.
  gQueue.fenceSubmission.erase(handleKey(info->fence));
  gQueue.exportedFences.insert(handleKey(info->fence));
  *fd = 1000 + static_cast<int>(gQueue.exports);
  return VK_SUCCESS;
}
#endif // defined(VK_KHR_external_fence_fd)

thread_local PFN_vkEndCommandBuffer gRealEndCommandBuffer = nullptr;
thread_local PFN_vkCreateFence gRealCreateFence = nullptr;
thread_local PFN_vkDestroyFence gRealDestroyFence = nullptr;
thread_local PFN_vkResetCommandBuffer gRealResetCommandBuffer = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL fakeCreateFence(VkDevice device,
                                               const VkFenceCreateInfo* info,
                                               const VkAllocationCallbacks* allocator,
                                               VkFence* fence) {
  if (gQueue.createFenceResult) {
    const VkResult result = *gQueue.createFenceResult;
    gQueue.createFenceResult.reset();
    return result;
  }
  VkFenceCreateInfo createInfo = *info;
  if (createInfo.pNext != nullptr &&
      static_cast<const VkBaseInStructure*>(createInfo.pNext)->sType ==
          VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO) {
    ++gQueue.exportableFenceCreates;
    createInfo.pNext = nullptr;
  }
  return gRealCreateFence(device, &createInfo, allocator, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL fakeResetCommandBuffer(VkCommandBuffer cb,
                                                      VkCommandBufferResetFlags flags) {
  if (!gQueue.resetCommandBufferResults.empty()) {
    const VkResult result = gQueue.resetCommandBufferResults.front();
    gQueue.resetCommandBufferResults.erase(gQueue.resetCommandBufferResults.begin());
    if (result != VK_SUCCESS) {
      return result;
    }
  }
  return gRealResetCommandBuffer(cb, flags);
}

VKAPI_ATTR VkResult VKAPI_CALL fakeEndCommandBuffer(VkCommandBuffer cb) {
  if (gQueue.endResult != VK_SUCCESS) {
    const VkResult result = gQueue.endResult;
    gQueue.endResult = VK_SUCCESS;
    return result;
  }
  return gRealEndCommandBuffer(cb);
}

VKAPI_ATTR void VKAPI_CALL fakeDestroyFence(VkDevice device,
                                            VkFence fence,
                                            const VkAllocationCallbacks* allocator) {
  // A later fence may get the handle of a destroyed back-off fence.
  gQueue.backoffFences.erase(handleKey(fence));
  gRealDestroyFence(device, fence, allocator);
}

[[nodiscard]] VulkanFunctionTable fakeQueueTable(const VulkanFunctionTable& real) {
  gQueue = {};
  gRealEndCommandBuffer = real.vkEndCommandBuffer;
  gRealCreateFence = real.vkCreateFence;
  gRealDestroyFence = real.vkDestroyFence;
  gRealResetCommandBuffer = real.vkResetCommandBuffer;
  VulkanFunctionTable vf = real;
  vf.vkEndCommandBuffer = fakeEndCommandBuffer;
  vf.vkCreateFence = fakeCreateFence;
  vf.vkDestroyFence = fakeDestroyFence;
  vf.vkResetCommandBuffer = fakeResetCommandBuffer;
  vf.vkQueueSubmit = fakeQueueSubmit;
  vf.vkQueueSubmit2KHR = fakeQueueSubmit2;
  vf.vkWaitForFences = fakeWaitForFences;
  vf.vkResetFences = fakeResetFences;
#if defined(VK_KHR_external_fence_fd)
  vf.vkGetFenceFdKHR = fakeGetFenceFd;
#endif // defined(VK_KHR_external_fence_fd)
  return vf;
}

/// Completes the first `n` submissions made to the fake queue.
void completeSubmissions(size_t n) {
  gQueue.completed = std::max(gQueue.completed, n);
}

/// How many submissions to the fake queue carried `fence`.
size_t timesSubmitted(VkFence fence) {
  return static_cast<size_t>(std::count(gQueue.submitted.begin(), gQueue.submitted.end(), fence));
}

// Entries for the context's own function table (see `ScopedTableEntry`).

VKAPI_ATTR VkResult VKAPI_CALL failingEndCommandBuffer(VkCommandBuffer /*cb*/) {
  return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VKAPI_ATTR VkResult VKAPI_CALL outOfMemoryQueueSubmit(VkQueue /*queue*/,
                                                      uint32_t /*count*/,
                                                      const VkSubmitInfo* /*submits*/,
                                                      VkFence /*fence*/) {
  return VK_ERROR_OUT_OF_HOST_MEMORY;
}

VKAPI_ATTR VkResult VKAPI_CALL outOfMemoryQueueSubmit2(VkQueue /*queue*/,
                                                       uint32_t /*count*/,
                                                       const VkSubmitInfo2* /*submits*/,
                                                       VkFence /*fence*/) {
  return VK_ERROR_OUT_OF_HOST_MEMORY;
}

/// Counts, and can fail, the recordings, submissions, barriers and fence waits of staging
/// operations.
struct StagingFaults {
  PFN_vkEndCommandBuffer realEnd = nullptr;
  PFN_vkQueueSubmit realSubmit = nullptr;
  PFN_vkQueueSubmit2KHR realSubmit2 = nullptr;
  PFN_vkCmdPipelineBarrier realBarrier = nullptr;
  uint32_t ends = 0;
  uint32_t submits = 0;
  uint32_t barriers = 0;
  /// The recording, counted from 1, whose end fails; 0 for none.
  uint32_t failingEnd = 0;
  /// The submission, counted from 1, from which two submissions run out of memory; 0 for none.
  uint32_t firstFailingSubmit = 0;
  PFN_vkWaitForFences realWait = nullptr;
  /// The submission, counted from 1, that runs but reports VK_ERROR_UNKNOWN; 0 for none.
  uint32_t unknownSubmit = 0;
  /// What every fence wait and poll answers, unless VK_SUCCESS.
  VkResult fenceAnswer = VK_SUCCESS;
};
thread_local StagingFaults gStaging;

VKAPI_ATTR VkResult VKAPI_CALL stagingEndCommandBuffer(VkCommandBuffer cb) {
  return ++gStaging.ends == gStaging.failingEnd ? VK_ERROR_OUT_OF_DEVICE_MEMORY
                                                : gStaging.realEnd(cb);
}

bool stagingSubmitFails() {
  ++gStaging.submits;
  return gStaging.firstFailingSubmit != 0 && gStaging.submits >= gStaging.firstFailingSubmit &&
         gStaging.submits < gStaging.firstFailingSubmit + 2;
}

VkResult stagingSubmitResult(VkResult realResult) {
  return gStaging.submits == gStaging.unknownSubmit && realResult == VK_SUCCESS ? VK_ERROR_UNKNOWN
                                                                                : realResult;
}

VKAPI_ATTR VkResult VKAPI_CALL stagingQueueSubmit(VkQueue queue,
                                                  uint32_t count,
                                                  const VkSubmitInfo* submits,
                                                  VkFence fence) {
  return stagingSubmitFails()
             ? VK_ERROR_OUT_OF_HOST_MEMORY
             : stagingSubmitResult(gStaging.realSubmit(queue, count, submits, fence));
}

VKAPI_ATTR VkResult VKAPI_CALL stagingQueueSubmit2(VkQueue queue,
                                                   uint32_t count,
                                                   const VkSubmitInfo2* submits,
                                                   VkFence fence) {
  return stagingSubmitFails()
             ? VK_ERROR_OUT_OF_HOST_MEMORY
             : stagingSubmitResult(gStaging.realSubmit2(queue, count, submits, fence));
}

VKAPI_ATTR VkResult VKAPI_CALL stagingWaitForFences(VkDevice device,
                                                    uint32_t count,
                                                    const VkFence* fences,
                                                    VkBool32 waitAll,
                                                    uint64_t timeout) {
  return gStaging.fenceAnswer != VK_SUCCESS
             ? gStaging.fenceAnswer
             : gStaging.realWait(device, count, fences, waitAll, timeout);
}

VKAPI_ATTR void VKAPI_CALL stagingCmdPipelineBarrier(VkCommandBuffer cb,
                                                     VkPipelineStageFlags srcStageMask,
                                                     VkPipelineStageFlags dstStageMask,
                                                     VkDependencyFlags dependencyFlags,
                                                     uint32_t memoryBarrierCount,
                                                     const VkMemoryBarrier* memoryBarriers,
                                                     uint32_t bufferBarrierCount,
                                                     const VkBufferMemoryBarrier* bufferBarriers,
                                                     uint32_t imageBarrierCount,
                                                     const VkImageMemoryBarrier* imageBarriers) {
  ++gStaging.barriers;
  gStaging.realBarrier(cb,
                       srcStageMask,
                       dstStageMask,
                       dependencyFlags,
                       memoryBarrierCount,
                       memoryBarriers,
                       bufferBarrierCount,
                       bufferBarriers,
                       imageBarrierCount,
                       imageBarriers);
}

thread_local PFN_vkGetQueryPoolResults gRealGetQueryPoolResults = nullptr;
thread_local size_t gQueryReads = 0;

VKAPI_ATTR VkResult VKAPI_CALL countingGetQueryPoolResults(VkDevice device,
                                                           VkQueryPool queryPool,
                                                           uint32_t firstQuery,
                                                           uint32_t queryCount,
                                                           size_t dataSize,
                                                           void* data,
                                                           VkDeviceSize stride,
                                                           VkQueryResultFlags flags) {
  ++gQueryReads;
  return gRealGetQueryPoolResults(
      device, queryPool, firstQuery, queryCount, dataSize, data, stride, flags);
}

thread_local PFN_vkDestroyCommandPool gRealDestroyCommandPool = nullptr;
thread_local size_t gCommandPoolsDestroyed = 0;

/// The command pools destroyed, in order.
thread_local std::vector<VkCommandPool> gDestroyedCommandPools;

VKAPI_ATTR void VKAPI_CALL recordingDestroyCommandPool(VkDevice device,
                                                       VkCommandPool commandPool,
                                                       const VkAllocationCallbacks* allocator) {
  gDestroyedCommandPools.push_back(commandPool);
  gRealDestroyCommandPool(device, commandPool, allocator);
}

VKAPI_ATTR void VKAPI_CALL countingDestroyCommandPool(VkDevice device,
                                                      VkCommandPool commandPool,
                                                      const VkAllocationCallbacks* allocator) {
  ++gCommandPoolsDestroyed;
  gRealDestroyCommandPool(device, commandPool, allocator);
}

thread_local PFN_vkWaitForFences gRealWaitForFences = nullptr;
/// The fences that the context's pools polled or waited on.
thread_local std::vector<uint64_t> gWaitedFences;

VKAPI_ATTR VkResult VKAPI_CALL recordingWaitForFences(VkDevice device,
                                                      uint32_t count,
                                                      const VkFence* fences,
                                                      VkBool32 waitAll,
                                                      uint64_t timeout) {
  for (uint32_t i = 0; i < count; ++i) {
    gWaitedFences.push_back(handleKey(fences[i]));
  }
  return gRealWaitForFences(device, count, fences, waitAll, timeout);
}

thread_local PFN_vkQueueWaitIdle gRealQueueWaitIdle = nullptr;
/// What a successful `vkQueueWaitIdle()` answers instead.
thread_local VkResult gIdleResult = VK_SUCCESS;
thread_local bool gIdleSeen = false;

VKAPI_ATTR VkResult VKAPI_CALL answeringQueueWaitIdle(VkQueue queue) {
  gIdleSeen = true;
  const VkResult result = gRealQueueWaitIdle(queue);
  return result == VK_SUCCESS ? gIdleResult : result;
}

thread_local PFN_vkQueueSubmit gRealContextSubmit = nullptr;
thread_local PFN_vkQueueSubmit2KHR gRealContextSubmit2 = nullptr;
thread_local size_t gSubmitsAfterIdle = 0;

VKAPI_ATTR VkResult VKAPI_CALL countingQueueSubmit(VkQueue queue,
                                                   uint32_t count,
                                                   const VkSubmitInfo* submits,
                                                   VkFence fence) {
  gSubmitsAfterIdle += gIdleSeen ? 1 : 0;
  return gRealContextSubmit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL countingQueueSubmit2(VkQueue queue,
                                                    uint32_t count,
                                                    const VkSubmitInfo2* submits,
                                                    VkFence fence) {
  gSubmitsAfterIdle += gIdleSeen ? 1 : 0;
  return gRealContextSubmit2(queue, count, submits, fence);
}

thread_local size_t gCheckpointFailures = 0;
thread_local PFN_vkDestroyDevice gRealDestroyDevice = nullptr;
thread_local bool gDeviceDestroyed = false;
thread_local PFN_vkDestroyPipelineCache gRealDestroyPipelineCache = nullptr;
thread_local const vulkan::VulkanContext* gTeardownContext = nullptr;
thread_local size_t gTasksLeftAfterTheSweeps = 0;

/// Called by the context's destructor after its last deferred-task sweep: records the tasks left.
VKAPI_ATTR void VKAPI_CALL recordingDestroyPipelineCache(VkDevice device,
                                                         VkPipelineCache cache,
                                                         const VkAllocationCallbacks* allocator) {
  gTasksLeftAfterTheSweeps = gTeardownContext ? gTeardownContext->deferredTasks.size() : 0;
  gRealDestroyPipelineCache(device, cache, allocator);
}

/// Set when the object that the requeueing tasks own is destroyed after the device.
thread_local bool gOwnedDestroyedAfterTheDevice = false;

/// Owned by the requeueing tasks, as a task that destroys a semaphore owns it.
struct OwnedByTasks {
  OwnedByTasks() = default;
  OwnedByTasks(const OwnedByTasks&) = delete;
  OwnedByTasks& operator=(const OwnedByTasks&) = delete;
  OwnedByTasks(OwnedByTasks&&) = delete;
  OwnedByTasks& operator=(OwnedByTasks&&) = delete;
  ~OwnedByTasks() {
    gOwnedDestroyedAfterTheDevice = gDeviceDestroyed;
  }
};

/// A deferred task that queues another such task, with the default handle, each time it runs. Each
/// one owns a share of an `OwnedByTasks`.
struct RequeueingTask {
  vulkan::VulkanContext* IGL_NONNULL context;
  int* IGL_NONNULL runs;
  std::shared_ptr<OwnedByTasks> owned;
  void operator()() const {
    ++*runs;
    context->deferredTask(std::packaged_task<void()>(
        RequeueingTask{.context = context, .runs = runs, .owned = owned}));
  }
};

/// Checkpoints (empty batches) run out of memory while `gCheckpointFailures` lasts.
VKAPI_ATTR VkResult VKAPI_CALL failingCheckpointSubmit(VkQueue queue,
                                                       uint32_t count,
                                                       const VkSubmitInfo* submits,
                                                       VkFence fence) {
  if (count == 0 && gCheckpointFailures != 0) {
    --gCheckpointFailures;
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }
  return gRealContextSubmit(queue, count, submits, fence);
}

VKAPI_ATTR void VKAPI_CALL recordingDestroyDevice(VkDevice device,
                                                  const VkAllocationCallbacks* allocator) {
  gDeviceDestroyed = true;
  gRealDestroyDevice(device, allocator);
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
    case Commands::RowState::Dropped:
      return "Dropped";
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

  static VkResult certifyAll(Commands& commands, uint64_t timeoutNanoseconds) {
    uint64_t remaining = timeoutNanoseconds;
    return commands.certifyAll(remaining);
  }
  static int exportFenceFd(Commands& commands, Commands::SubmitHandle handle) {
    return commands.exportFenceFd(handle);
  }

  static uint64_t retiredThrough(const Commands& commands) {
    return commands.retiredThrough_;
  }
  static uint32_t numAvailable(const Commands& commands) {
    return commands.numAvailableCommandBuffers_;
  }
  static bool droppedTheRecording(const Commands& commands) {
    return commands.lastSubmitDroppedTheRecording();
  }
  static Commands::SubmitHandle lastAccepted(const Commands& commands) {
    return commands.lastAcceptedSubmitHandle();
  }
  static bool isRetired(const Commands& commands, Commands::SubmitHandle handle) {
    return commands.isRetired(handle);
  }
  static bool outcomesLost(const Commands& commands, uint32_t row) {
    return commands.buffers_[row].outcomesLost_;
  }
  static bool chainPoisoned(const Commands& commands) {
    return commands.chainPoisoned_;
  }
  static bool recycleDroppedRows(Commands& commands) {
    return commands.recycleDroppedRows();
  }
  static VkResult waitRetired(Commands& commands,
                              Commands::SubmitHandle handle,
                              uint64_t timeoutNanoseconds) {
    return commands.waitRetired(handle, timeoutNanoseconds);
  }
  /// One try of an exhausted `acquire()` to free a row.
  static bool tryFreeRow(Commands& commands) {
    return commands.freeRow();
  }
  /// The queue state the pool uses, and the one it owns.
  static const void* sharedQueueState(const Commands& commands) {
    return commands.queueState_;
  }
  static const void* ownQueueState(const Commands& commands) {
    return &commands.ownQueueState_;
  }
  static VkCommandPool commandPoolOf(const Commands& commands) {
    return commands.commandPool_;
  }
  static bool queueStopped(const Commands& commands) {
    return commands.queueState_->stopped;
  }
  /// Makes `commands` stop when `other` stops, as the two pools of a context do.
  static uint64_t lossAnswers(const Commands& commands) {
    return commands.queueState_->lossAnswers;
  }
  /// One back-off, the first of a call.
  static VkResult backoff(Commands& commands) {
    uint32_t attempt = 0;
    return commands.backoff(attempt);
  }
  static Commands::QueueState& queueState(const Commands& commands) {
    return *commands.queueState_;
  }
  static void shareQueueStop(Commands& commands, const Commands& other) {
    commands.queueState_ = other.queueState_;
  }
  /// Whether every submission waits on the previous one's semaphore (see `VulkanContextConfig`).
  static bool& chained(Commands& commands) {
    return commands.chainSubmitsWithSemaphores_;
  }
  /// A reusable command buffer whose run A is in flight and whose run B, from the next slot, was
  /// not submitted because two tries to submit it ran out of memory, and an ordinary command
  /// buffer.
  struct ReusableRunsWithADroppedRun {
    std::shared_ptr<ICommandQueue> queue;
    std::shared_ptr<IFramebuffer> framebuffer;
    std::shared_ptr<ICommandBuffer> reusable;
    std::shared_ptr<ICommandBuffer> ordinary;
    Commands::SubmitHandle runA;
    SubmitHandle runB = 0;
  };

  /// Sets up `runs` on `pool`, a pool on the fake queue that the context submits to.
  void startReusableRunsWithADroppedRun(const Commands& pool, ReusableRunsWithADroppedRun& runs) {
    auto& platformDevice = *iglDev_->getPlatformDevice<vulkan::PlatformDevice>();
    ASSERT_TRUE(platformDevice.supportsReusableCommandBuffers());
    Result ret;
    runs.queue = iglDev_->createCommandQueue({}, &ret);
    ASSERT_NE(runs.queue, nullptr);
    const auto texture = iglDev_->createTexture(
        TextureDesc::new2D(
            TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Attachment),
        &ret);
    ASSERT_NE(texture, nullptr);
    FramebufferDesc framebufferDesc;
    framebufferDesc.colorAttachments[0].texture = texture;
    runs.framebuffer = iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_NE(runs.framebuffer, nullptr);
    runs.reusable =
        platformDevice.createReusableCommandBuffer({.debugName = "Dropped run test"}, &ret);
    ASSERT_NE(runs.reusable, nullptr);
    const RenderPassDesc renderPass{
        .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}}};

    auto encoder = runs.reusable->createRenderCommandEncoder(renderPass, runs.framebuffer, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encoder->endEncoding();
    ASSERT_NE(runs.queue->submit(*runs.reusable), 0u);
    runs.runA = pool.getLastSubmitHandle();

    encoder = runs.reusable->createRenderCommandEncoder(renderPass, runs.framebuffer, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encoder->endEncoding();
    // Both tries to submit run B run out of memory: it is not submitted, and the caller gets its
    // handle, which never completes, as on a queue where it is never signaled.
    gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
    runs.runB = runs.queue->submit(*runs.reusable);
    ASSERT_TRUE(droppedTheRecording(pool));
    ASSERT_NE(runs.runB, 0u);
    ASSERT_EQ(pool.getLastSubmitHandle().handle(), runs.runB);
    ASSERT_EQ(lastAccepted(pool), runs.runA);

    // Run A took the first row, and run B the second, which stays out of circulation.
    ASSERT_EQ(runs.runA.bufferIndex, 0u);
    ASSERT_STREQ(rowState(pool, 1), "Dropped");
    runs.ordinary = runs.queue->createCommandBuffer({}, &ret);
    ASSERT_NE(runs.ordinary, nullptr);
    ASSERT_EQ(gQueue.completed, 0u);
  }
  static size_t checkpointFences(const Commands& commands) {
    return commands.checkpoints_.size();
  }
  static VkFence checkpointFence(const Commands& commands, size_t index) {
    return commands.checkpoints_[index].fence.vkFence_;
  }
  static const char* checkpointState(const Commands& commands, size_t index) {
    switch (commands.checkpoints_[index].state) {
    case Commands::Checkpoint::State::Unused:
      return "Unused";
    case Commands::Checkpoint::State::Pending:
      return "Pending";
    case Commands::Checkpoint::State::Signaled:
      return "Signaled";
    case Commands::Checkpoint::State::Abandoned:
      return "Abandoned";
    case Commands::Checkpoint::State::Quarantined:
      return "Quarantined";
    }
    return "?";
  }
  /// What an export or an unknown submit result does to the fence of a row.
  static void distrustFence(Commands& commands, uint32_t row) {
    commands.buffers_[row].retirement_.fenceTrusted = false;
  }
  static bool fenceTrusted(const Commands& commands, uint32_t row) {
    return commands.buffers_[row].retirement_.fenceTrusted;
  }
  static bool quarantined(const Commands& commands, uint32_t row) {
    return commands.buffers_[row].retirement_.quarantined;
  }
  /// Submits a recording on every row but `exceptRow`; none of them finishes.
  static void fillOtherRows(Commands& pool, uint32_t exceptRow) {
    for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
      if (row != exceptRow) {
        static_cast<void>(pool.submit(pool.acquire()));
      }
    }
  }
  static void certifyQueueIdle(const Commands& commands, bool lost) {
    commands.certifyQueueIdle(lost);
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
  std::unique_ptr<Commands> makeFakePool(const char* name, bool exportableFences = false) {
    auto& ctx = getVulkanContext();
    // The back-off fences of the pools made before stay known.
    std::map<uint64_t, void*> backoffFences = std::move(gQueue.backoffFences);
    fakeTable_ = fakeQueueTable(ctx.vf_);
    gQueue.backoffFences = std::move(backoffFences);
    auto pool = std::make_unique<Commands>(fakeTable_,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           exportableFences,
                                           false,
                                           false,
                                           name);
    // The fake queue tells the back-off fence apart from the others: create it now (a pool creates
    // it at its first back-off).
    EXPECT_EQ(pool->ensureBackoffFence(), VK_SUCCESS);
    gQueue.backoffFences.emplace(handleKey(pool->backoffFence_->vkFence_), pool.get());
    gQueue.endRetries = endRetries;
    return pool;
  }
  /// Ends the retries of a pool that backed off too often (the test already failed): its work
  /// ends as lost, and a row is freed when none is, so that `acquire()` returns.
  static void endRetries(void* IGL_NONNULL pool) {
    auto& commands = *static_cast<Commands*>(pool);
    commands.certifyQueueIdle(true);
    if (commands.numAvailableCommandBuffers_ != 0) {
      return;
    }
    for (auto& buf : commands.buffers_) {
      if (buf.state != Commands::RowState::Encoding) {
        commands.setRowState(buf, Commands::RowState::Free);
        ++commands.numAvailableCommandBuffers_;
        return;
      }
    }
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
      // As the pool it replaces, it shares the context's queue state, which is put back as it was.
      shareQueueStop(*slot_, *realPool_);
      savedQueueState_ = queueState(*realPool_);
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
      queueState(*slot_) = savedQueueState_;
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
    Commands::QueueState savedQueueState_;
  };

  /// Holds `value` at `temporary` until destroyed.
  template<typename T>
  class ScopedValue {
   public:
    ScopedValue(T& value, T temporary) : value_(value), saved_(value) {
      value_ = temporary;
    }
    ScopedValue(const ScopedValue&) = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;
    ~ScopedValue() {
      value_ = saved_;
    }

   private:
    T& value_;
    T saved_;
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

  /// Submits `wrapper` with `next` after it in the same batch, as a reusable command buffer's run.
  static Commands::SubmitHandle submitWithNext(Commands& commands,
                                               const Commands::CommandBufferWrapper& wrapper,
                                               VkCommandBuffer next) {
    return commands.submit(wrapper, VK_NULL_HANDLE, next);
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

TEST_F(VulkanImmediateCommandsTest,
       AnExhaustedAcquirePollsEachRowOnceThenWaitsOnTheOldestCallTrace) {
  auto& ctx = getVulkanContext();
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Exhausted acquire trace test");
  const TracedWaitsSucceedAtScopeEnd waitsSucceed;
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    static_cast<void>(commands.submit(commands.acquire()));
  }
  gTrace.lines.clear();
  gTrace.pollResult = VK_TIMEOUT; // nothing has finished yet
  const auto& next = commands.acquire();
  gTrace.pollResult.reset();
  // One poll of every row's fence, then one wait on the oldest one; only its row is recycled.
  std::vector<std::string> expected;
  for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
    expected.push_back("vkWaitForFences all=1 timeout=0 fence" + std::to_string(row));
  }
  expected.insert(expected.end(),
                  {
                      "vkWaitForFences all=1 timeout=max fence0",
                      "vkResetCommandBuffer cb0 flags=0",
                      "vkResetFences fence0",
                      "vkBeginCommandBuffer cb0 flags=1",
                  });
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);
  EXPECT_EQ(next.handle.bufferIndex, 0u);
  for (uint32_t row = 1; row != Commands::kMaxCommandBuffers; ++row) {
    EXPECT_STREQ(rowState(commands, row), "Submitted");
  }
  commands.discard(next);
}

TEST_F(VulkanImmediateCommandsTest, AnExhaustedAcquireFreesEveryRowWhoseSubmissionFinished) {
  // As a cross-API synchronizer does every frame: a signal injected into the next submission, which
  // finishes right after it is made, until the last frame, when no fence answers any more. An
  // exhausted acquisition frees every finished row at once, so the last frame still finds a free
  // row, without a wait or a back-off.
  auto& ctx = getVulkanContext();
  const auto pool = makeFakePool("Finished rows test");
  const vulkan::VulkanSemaphore signal(ctx.vf_, ctx.getVkDevice(), false, "Finished rows test");
  constexpr uint32_t kFrames = 200;
  uint32_t exhausted = 0;
  for (uint32_t frame = 0; frame != kFrames; ++frame) {
    SCOPED_TRACE(frame);
    if (frame + 1 == kFrames) {
      gQueue.waitOverride = VK_TIMEOUT;
    }
    exhausted += numAvailable(*pool) == 0 ? 1 : 0;
    ASSERT_TRUE(pool->signalSemaphore(signal.getVkSemaphore(), frame + 1));
    static_cast<void>(pool->submit(pool->acquire()));
    if (frame + 1 != kFrames) {
      completeSubmissions(gQueue.submitted.size());
    }
  }
  gQueue.waitOverride.reset();
  EXPECT_EQ(gQueue.submitted.size(), kFrames);
  EXPECT_TRUE(gQueue.backoffWaits.empty());
  EXPECT_EQ(gQueue.blockingWaits, 0u);
  EXPECT_EQ(checkpointFences(*pool), 0u);
  // Freed in batches: the rows run out once every 32 frames, not on every frame after the first 32.
  EXPECT_EQ(exhausted, (kFrames - 1) / Commands::kMaxCommandBuffers);
  completeSubmissions(gQueue.submitted.size());
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
  EXPECT_EQ(wrapper.cmdBuf, nullptr);
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
  EXPECT_NE(again.cmdBuf, nullptr);
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
  EXPECT_NE(outer.cmdBuf, nullptr);
  EXPECT_STREQ(rowState(commands, outer.handle.bufferIndex), "Encoding");
  EXPECT_STREQ(rowState(commands, inner.handle.bufferIndex), "Free");
  // Only the submitted row is waited on and reset; the open recording is not touched.
  const std::vector<std::string> expected = {
      "vkBeginCommandBuffer cb0 flags=1",
      "vkBeginCommandBuffer cb1 flags=1",
      "vkEndCommandBuffer cb1",
      "vkQueueSubmit cb1 fence1 waits=0 signals=1 signal:sem1",
      "vkWaitForFences all=1 timeout=max fence1",
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

  // The rows a wait certified are freed without another poll of their fences.
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
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
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
  const auto waitSemaphore = opaqueHandle<VkSemaphore>(0x10);
  const auto signalSemaphore = opaqueHandle<VkSemaphore>(0x20);
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
  // The rows a wait certified are freed without another poll of their fences.
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
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
      "vkResetCommandBuffer cb1 flags=0",
      "vkResetFences fence1",
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
  // The rows a wait certified are freed without another poll of their fences.
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
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
      "vkResetCommandBuffer cb1 flags=0",
      "vkResetFences fence1",
      "vkWaitForFences all=1 timeout=max fence32 fence33",
      "vkResetCommandBuffer cb32 flags=0",
      "vkResetFences fence32",
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
      "vkResetCommandBuffer cb0 flags=0",
      "vkResetFences fence0",
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

TEST_F(VulkanImmediateCommandsTest, FailedSubmitsCallTrace) {
  auto& ctx = getVulkanContext();
  struct Case {
    const char* name;
    /// What the submissions of the second recording answer, and what its end answers.
    std::vector<VkResult> submitResults;
    VkResult endResult = VK_SUCCESS;
    bool chained = true;
    /// Whether the second and the third recordings are submitted.
    bool secondSubmitted = true;
    bool thirdSubmitted = true;
    std::vector<std::string> expected;
  };
  // Out of memory once: the same batch is submitted again. A recording that cannot run (its end
  // fails, or two tries to submit it run out of memory) is not submitted: its row is not reset, and
  // the next submission, which would wait on its chain semaphore, is not submitted either; without
  // the chain the next submission waits on the last semaphore that was signaled (sem0). An unknown
  // result takes a submission whose fence is never polled: the later fence covers it, and the row
  // is reset without a poll. A device loss keeps a trusted fence, and nothing is submitted after
  // it.
  const std::vector<Case> cases = {
      {.name = "out of memory once",
       .submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY},
       .expected =
           {
               "vkBeginCommandBuffer cb0 flags=1",
               "vkEndCommandBuffer cb0",
               "vkQueueSubmit cb0 fence0 waits=0 signals=1 signal:sem0",
               "vkBeginCommandBuffer cb1 flags=1",
               "vkEndCommandBuffer cb1",
               "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
               "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
               "vkBeginCommandBuffer cb2 flags=1",
               "vkEndCommandBuffer cb2",
               "vkQueueSubmit cb2 fence2 waits=1 signals=1 wait:sem1 signal:sem2",
               "vkWaitForFences all=1 timeout=0 fence0",
               "vkWaitForFences all=1 timeout=0 fence2",
           }},
      {.name = "out of memory twice",
       .submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY},
       .secondSubmitted = false,
       .thirdSubmitted = false,
       .expected =
           {
               "vkBeginCommandBuffer cb0 flags=1",
               "vkEndCommandBuffer cb0",
               "vkQueueSubmit cb0 fence0 waits=0 signals=1 signal:sem0",
               "vkBeginCommandBuffer cb1 flags=1",
               "vkEndCommandBuffer cb1",
               "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
               "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
               "vkBeginCommandBuffer cb2 flags=1",
               "vkEndCommandBuffer cb2",
               "vkWaitForFences all=1 timeout=0 fence0",
           }},
      {.name = "out of memory twice, without the chain",
       .submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY},
       .chained = false,
       .secondSubmitted = false,
       .expected =
           {
               "vkBeginCommandBuffer cb0 flags=1",
               "vkEndCommandBuffer cb0",
               "vkQueueSubmit cb0 fence0 waits=0 signals=0",
               "vkBeginCommandBuffer cb1 flags=1",
               "vkEndCommandBuffer cb1",
               "vkQueueSubmit cb1 fence1 waits=0 signals=0",
               "vkQueueSubmit cb1 fence1 waits=0 signals=0",
               "vkBeginCommandBuffer cb2 flags=1",
               "vkEndCommandBuffer cb2",
               "vkQueueSubmit cb2 fence2 waits=0 signals=0",
               "vkWaitForFences all=1 timeout=0 fence0",
               "vkWaitForFences all=1 timeout=0 fence2",
           }},
      {.name = "failed end",
       .endResult = VK_ERROR_OUT_OF_HOST_MEMORY,
       .secondSubmitted = false,
       .thirdSubmitted = false,
       .expected =
           {
               "vkBeginCommandBuffer cb0 flags=1",
               "vkEndCommandBuffer cb0",
               "vkQueueSubmit cb0 fence0 waits=0 signals=1 signal:sem0",
               "vkBeginCommandBuffer cb1 flags=1",
               "vkEndCommandBuffer cb1",
               "vkBeginCommandBuffer cb2 flags=1",
               "vkEndCommandBuffer cb2",
               "vkWaitForFences all=1 timeout=0 fence0",
           }},
      {.name = "unknown result",
       .submitResults = {VK_ERROR_UNKNOWN},
       .expected =
           {
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
           }},
      {.name = "device lost",
       .submitResults = {VK_ERROR_DEVICE_LOST},
       .thirdSubmitted = false,
       .expected =
           {
               "vkBeginCommandBuffer cb0 flags=1",
               "vkEndCommandBuffer cb0",
               "vkQueueSubmit cb0 fence0 waits=0 signals=1 signal:sem0",
               "vkBeginCommandBuffer cb1 flags=1",
               "vkEndCommandBuffer cb1",
               "vkQueueSubmit cb1 fence1 waits=1 signals=1 wait:sem0 signal:sem1",
               "vkBeginCommandBuffer cb2 flags=1",
               "vkEndCommandBuffer cb2",
               "vkWaitForFences all=1 timeout=0 fence0",
               "vkWaitForFences all=1 timeout=0 fence1",
           }},
  };
  for (const Case& c : cases) {
    const VulkanFunctionTable vf = tracedTable(ctx.vf_);
    vulkan::VulkanImmediateCommands commands(vf,
                                             ctx.getVkDevice(),
                                             ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                             false,
                                             false,
                                             false,
                                             "Failed submit trace test",
                                             c.chained);
    gTrace.lines.clear();
    const auto first = commands.submit(commands.acquire());
    gTrace.submitResults = c.submitResults;
    gTrace.endResult = c.endResult;
    const auto second = commands.submit(commands.acquire());
    EXPECT_EQ(droppedTheRecording(commands), !c.secondSubmitted) << c.name;
    gTrace.submitResults.clear();
    const auto third = commands.submit(commands.acquire());
    EXPECT_EQ(droppedTheRecording(commands), !c.thirdSubmitted) << c.name;
    EXPECT_EQ(commands.wait(first), VK_SUCCESS) << c.name;
    if (c.thirdSubmitted) {
      EXPECT_EQ(commands.wait(third), VK_SUCCESS) << c.name;
    }
    // A recording that was not submitted never completes.
    EXPECT_EQ(commands.isReady(second), c.secondSubmitted) << c.name;
    EXPECT_EQ(gTrace.lines, c.expected) << c.name << "\n" << joinLines(gTrace.lines);
  }
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

TEST_F(VulkanImmediateCommandsTest, HandlelessDeferredTaskOfAnOpenRecordingWaitsForItsSubmission) {
  auto& ctx = getVulkanContext();
  const FakeContextPool fakePool(*this, ctx.immediate_, "Open recording deferred task test");
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, 1'000'000);
  const auto earlier = ctx.immediate_->submit(ctx.immediate_->acquire());
  completeSubmissions(1);
  ASSERT_TRUE(ctx.immediate_->isReady(earlier));
  bool ran = false;
  const auto& recording = ctx.immediate_->acquire();
  ctx.deferredTask(std::packaged_task<void()>([&ran] { ran = true; }));
  const auto handle = ctx.immediate_->submit(recording);

  // The task waits for the recording that was open, not just for the work submitted before it.
  gQueue.blockingWaitsComplete = false;
  ctx.waitDeferredTasks();
  EXPECT_FALSE(ran);
  gQueue.blockingWaitsComplete = true;
  ctx.waitDeferredTasks();
  EXPECT_TRUE(ran);
  EXPECT_TRUE(ctx.immediate_->isReady(handle));
}

TEST_F(VulkanImmediateCommandsTest,
       HandlelessDeferredTaskWithoutAnOpenRecordingWaitsForTheLastSubmission) {
  auto& ctx = getVulkanContext();
  const FakeContextPool fakePool(*this, ctx.immediate_, "Last submission deferred task test");
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, 1'000'000);
  const auto last = ctx.immediate_->submit(ctx.immediate_->acquire());
  bool ran = false;
  ctx.deferredTask(std::packaged_task<void()>([&ran] { ran = true; }));

  gQueue.blockingWaitsComplete = false;
  ctx.waitDeferredTasks();
  EXPECT_FALSE(ran);
  gQueue.blockingWaitsComplete = true;
  ctx.waitDeferredTasks();
  EXPECT_TRUE(ran);
  EXPECT_TRUE(ctx.immediate_->isReady(last));
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
  EXPECT_EQ(gQueue.checkpoints, 0u);
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
  EXPECT_EQ(certifyAll(*pool, 1'000'000), VK_TIMEOUT);
  EXPECT_STREQ(rowState(*pool, handle.bufferIndex), "Submitted");
  EXPECT_EQ(retiredThrough(*pool), 0u);
  EXPECT_NE(pool->getVkFenceFromSubmitHandle(handle), VK_NULL_HANDLE);
  gQueue.blockingWaitsComplete = true;
  EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_SUCCESS);
  EXPECT_STREQ(rowState(*pool, handle.bufferIndex), "Free");
}

TEST_F(VulkanImmediateCommandsTest, ARowFreedAfterAWaitProvesItsSubmissionFinished) {
  const auto pool = makeFakePool("Freed rows test");
  const auto first = pool->submit(pool->acquire());
  const auto second = pool->submit(pool->acquire());
  const auto& recording = pool->acquire();
  const auto discarded = recording.handle;
  pool->discard(recording);
  // The later submission finishes while the wait for the first one blocks. The wait then frees
  // both rows, which proves that both submissions finished: the later handle and the recording
  // discarded after it are ready without a poll.
  gQueue.blockingWaitsCompleteEverything = true;
  ASSERT_EQ(pool->wait(first, kTestTimeout), VK_SUCCESS);
  EXPECT_STREQ(rowState(*pool, second.bufferIndex), "Free");
  const size_t polls = gQueue.polls;
  EXPECT_TRUE(pool->isReady(second));
  EXPECT_TRUE(pool->isReady(discarded));
  EXPECT_EQ(gQueue.polls, polls);
}

TEST_F(VulkanImmediateCommandsTest, FailedSubmitsTakeOrdinalsOnlyWhenTheyMayHaveRun) {
  const auto pool = makeFakePool("Failed submit ordinal test");
  // Without the chain, so that a recording that is not submitted does not stop the later ones.
  chained(*pool) = false;

  // Out of memory once: the batch is submitted again, and takes an ordinal.
  const auto first = pool->submit(pool->acquire());
  gQueue.submitResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto retried = pool->submit(pool->acquire());
  EXPECT_EQ(gQueue.submitCalls, 3u);
  EXPECT_EQ(submissions(*pool), 2u);
  EXPECT_FALSE(droppedTheRecording(*pool));
  EXPECT_STREQ(rowState(*pool, retried.bufferIndex), "Submitted");
  EXPECT_FALSE(pool->isReady(retried));
  completeSubmissions(2);
  EXPECT_TRUE(pool->isReady(first));
  EXPECT_TRUE(pool->isReady(retried));

  // When the second try runs out of memory too, nothing was submitted: no ordinal, and the handle
  // never completes.
  const auto before = pool->submit(pool->acquire());
  gQueue.submitResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_HOST_MEMORY};
  const auto& closed = pool->acquire();
  const auto notSubmitted = closed.handle;
  EXPECT_EQ(pool->submit(closed), notSubmitted);
  EXPECT_EQ(gQueue.submitCalls, 6u);
  EXPECT_EQ(submissions(*pool), 3u);
  EXPECT_TRUE(droppedTheRecording(*pool));
  EXPECT_STREQ(rowState(*pool, notSubmitted.bufferIndex), "Dropped");
  completeSubmissions(3);
  EXPECT_TRUE(pool->isReady(before));
  EXPECT_FALSE(pool->isReady(notSubmitted));

  // An unknown result takes an ordinal whose fence is never polled or waited on; a later fence
  // certifies it.
  gQueue.submitResults = {VK_ERROR_UNKNOWN};
  const auto unknown = pool->submit(pool->acquire());
  EXPECT_EQ(submissions(*pool), 4u);
  EXPECT_FALSE(droppedTheRecording(*pool));
  const VkFence unknownFence = pool->getVkFenceFromSubmitHandle(unknown);
  const auto later = pool->submit(pool->acquire());
  EXPECT_FALSE(pool->isReady(unknown));
  completeSubmissions(5);
  EXPECT_TRUE(pool->isReady(unknown));
  EXPECT_TRUE(pool->isReady(later));
  for (const uint64_t polled : gQueue.polledFences) {
    EXPECT_NE(polled, handleKey(unknownFence));
  }

  // An unknown submission that nothing later covers is certified by a checkpoint.
  gQueue.submitResults = {VK_ERROR_UNKNOWN};
  const auto lastUnknown = pool->submit(pool->acquire());
  EXPECT_EQ(pool->wait(lastUnknown, 1'000'000'000), VK_SUCCESS);
  EXPECT_EQ(gQueue.checkpoints, 1u);

  // A device loss takes an ordinal and keeps a trusted fence. Nothing is submitted after it.
  gQueue.submitResults = {VK_ERROR_DEVICE_LOST};
  const auto lost = pool->submit(pool->acquire());
  EXPECT_EQ(submissions(*pool), 7u);
  EXPECT_TRUE(pool->isDeviceLost());
  EXPECT_EQ(pool->wait(lost, 1'000'000'000), VK_SUCCESS);
  EXPECT_EQ(gQueue.checkpoints, 1u);
  const size_t submitCalls = gQueue.submitCalls;
  const auto afterTheLoss = pool->submit(pool->acquire());
  EXPECT_TRUE(droppedTheRecording(*pool));
  EXPECT_EQ(gQueue.submitCalls, submitCalls);
  EXPECT_EQ(submissions(*pool), 7u);
  EXPECT_STREQ(rowState(*pool, afterTheLoss.bufferIndex), "Dropped");
}

TEST_F(VulkanImmediateCommandsTest, ExportedRowsAreNeverPolledAndAcquireUsesOneCheckpoint) {
  const auto pool = makeFakePool("Exported rows test");
  // The acquisition below blocks until a checkpoint proves that a row finished. A bounded wait
  // checks first that a checkpoint proves an exported submission finished, so that a broken
  // checkpoint fails this test instead of blocking it.
  const auto probe = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, probe), 0);
  ASSERT_EQ(pool->wait(probe, kTestTimeout), VK_SUCCESS);
  ASSERT_EQ(gQueue.checkpoints, 1u);
  std::vector<Commands::SubmitHandle> handles;
  std::set<uint64_t> exportedFences;
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    handles.push_back(pool->submit(pool->acquire()));
    exportedFences.insert(handleKey(pool->getVkFenceFromSubmitHandle(handles.back())));
    EXPECT_GE(exportFenceFd(*pool, handles.back()), 0);
  }
  EXPECT_EQ(gQueue.exports, Commands::kMaxCommandBuffers + 1);
  // Until its row is reused, the fence of an exported submission is still returned.
  EXPECT_NE(pool->getVkFenceFromSubmitHandle(handles.front()), VK_NULL_HANDLE);
  // No row is free and no fence can be trusted: the acquisition submits one checkpoint and waits
  // for it. Its zero-time attempt submits and polls nothing.
  const size_t polls = gQueue.polls;
  const auto& next = pool->acquire();
  EXPECT_EQ(gQueue.checkpoints, 2u);
  EXPECT_EQ(checkpointFences(*pool), 1u);
  EXPECT_EQ(gQueue.polls, polls);
  for (const uint64_t polled : gQueue.polledFences) {
    EXPECT_EQ(exportedFences.count(polled), 0u);
  }
  EXPECT_TRUE(pool->isReady(handles.back()));
  pool->discard(next);
}

TEST_F(VulkanImmediateCommandsTest, CertifyAllCoversExportedAndUnknownSubmissions) {
  const auto pool = makeFakePool("Certify all test");
  const auto trusted = pool->submit(pool->acquire());
  const auto exported = pool->submit(pool->acquire());
  EXPECT_GE(exportFenceFd(*pool, exported), 0);
  gQueue.submitResults = {VK_ERROR_UNKNOWN};
  const auto unknown = pool->submit(pool->acquire());
  const auto& open = pool->acquire();
  EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_SUCCESS);
  EXPECT_EQ(gQueue.checkpoints, 1u);
  EXPECT_TRUE(pool->isReady(trusted));
  EXPECT_TRUE(pool->isReady(exported));
  EXPECT_TRUE(pool->isReady(unknown));
  // The open recording is not waited for.
  EXPECT_STREQ(rowState(*pool, open.handle.bufferIndex), "Encoding");
  for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
    EXPECT_STRNE(rowState(*pool, row), "Submitted");
  }
  pool->discard(open);
}

TEST_F(VulkanImmediateCommandsTest, CertifyAllReportsWhyItCouldNotCertify) {
  {
    const auto pool = makeFakePool("Certify all checkpoint memory test");
    const auto exported = pool->submit(pool->acquire());
    EXPECT_GE(exportFenceFd(*pool, exported), 0);
    gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY};
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_FALSE(pool->isReady(exported));
    gQueue.submitResults = {VK_ERROR_UNKNOWN};
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_ERROR_UNKNOWN);
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_SUCCESS);
  }
  {
    const auto pool = makeFakePool("Certify all device loss test");
    const auto handle = pool->submit(pool->acquire());
    gQueue.waitOverride = VK_ERROR_DEVICE_LOST;
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_ERROR_DEVICE_LOST);
    EXPECT_TRUE(pool->isDeviceLost());
    EXPECT_TRUE(pool->isReady(handle));
    EXPECT_EQ(pool->wait(handle, 1'000'000), VK_ERROR_DEVICE_LOST);
    gQueue.waitOverride.reset();
  }
}

TEST_F(VulkanImmediateCommandsTest, AFailedEndSubmitsNothing) {
  const auto pool = makeFakePool("Failed end test");
  // Without the chain, so that a recording that is not submitted does not stop the later ones.
  chained(*pool) = false;
  const auto first = pool->submit(pool->acquire());
  gQueue.endResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  const auto& notEnded = pool->acquire();
  const auto notEndedHandle = notEnded.handle;
  const VkFence notEndedFence = notEnded.fence.vkFence_;
  // The recording never runs: nothing is submitted, and it takes no ordinal. Its handle is
  // returned, and never completes.
  EXPECT_EQ(pool->submit(notEnded), notEndedHandle);
  EXPECT_EQ(gQueue.submitCalls, 1u);
  EXPECT_EQ(submissions(*pool), 1u);
  EXPECT_TRUE(droppedTheRecording(*pool));
  EXPECT_STREQ(rowState(*pool, notEndedHandle.bufferIndex), "Dropped");
  EXPECT_EQ(pool->getLastSubmitHandle(), notEndedHandle);
  EXPECT_EQ(lastAccepted(*pool), first);
  EXPECT_EQ(timesSubmitted(notEndedFence), 0u);
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(first));
  EXPECT_FALSE(pool->isReady(notEndedHandle));

  // The next recording runs.
  const auto next = pool->submit(pool->acquire());
  EXPECT_FALSE(droppedTheRecording(*pool));
  EXPECT_EQ(submissions(*pool), 2u);
  EXPECT_EQ(lastAccepted(*pool), next);
}

TEST_F(VulkanImmediateCommandsTest, ACheckpointWaitsForTheOldestPendingOneWhenBothAreInUse) {
  const auto pool = makeFakePool("Checkpoint slot test");
  std::vector<Commands::SubmitHandle> handles;
  gQueue.blockingWaitsComplete = false;
  for (uint32_t i = 0; i != 2; ++i) {
    handles.push_back(pool->submit(pool->acquire()));
    EXPECT_GE(exportFenceFd(*pool, handles.back()), 0);
    EXPECT_EQ(pool->wait(handles.back(), 1'000'000), VK_TIMEOUT);
  }
  EXPECT_EQ(gQueue.checkpoints, 2u);
  handles.push_back(pool->submit(pool->acquire()));
  EXPECT_GE(exportFenceFd(*pool, handles.back()), 0);
  // The wait for the oldest pending checkpoint uses the same budget.
  EXPECT_EQ(pool->wait(handles.back(), 1'000'000), VK_TIMEOUT);
  EXPECT_EQ(gQueue.checkpoints, 2u);
  gQueue.blockingWaitsComplete = true;
  // Both checkpoints are pending: the oldest is waited for and reused, within one finite budget.
  EXPECT_EQ(pool->wait(handles.back(), 5'000'000'000), VK_SUCCESS);
  EXPECT_EQ(gQueue.checkpoints, 3u);
  for (const auto& handle : handles) {
    EXPECT_TRUE(pool->isReady(handle));
  }
}

TEST_F(VulkanImmediateCommandsTest, ACheckpointThatLosesTheDeviceCertifiesAsLost) {
  const auto pool = makeFakePool("Checkpoint device loss test");
  const auto exported = pool->submit(pool->acquire());
  EXPECT_GE(exportFenceFd(*pool, exported), 0);
  gQueue.submitResults = {VK_ERROR_DEVICE_LOST};
  EXPECT_EQ(pool->wait(exported, 1'000'000'000), VK_ERROR_DEVICE_LOST);
  // The lost answer itself proves that nothing runs any more: nothing waits on the checkpoint.
  EXPECT_EQ(gQueue.blockingWaits, 0u);
  EXPECT_EQ(gQueue.polls, 0u);
  EXPECT_TRUE(pool->isDeviceLost());
  EXPECT_TRUE(pool->isReady(exported));
}

TEST_F(VulkanImmediateCommandsTest, AFailedExportKeepsTheFenceTrusted) {
  auto& ctx = getVulkanContext();
  {
    const auto pool = makeFakePool("Failed export test");
    gQueue.exportResult = VK_ERROR_OUT_OF_HOST_MEMORY;
    const auto handle = pool->submit(pool->acquire());
    EXPECT_EQ(exportFenceFd(*pool, handle), -1);
    EXPECT_EQ(gQueue.exports, 1u);
    // A failed export leaves the fence as it was: it is still polled, and no checkpoint is needed.
    completeSubmissions(1);
    EXPECT_TRUE(pool->isReady(handle));
    EXPECT_EQ(gQueue.polls, 1u);
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_SUCCESS);
    EXPECT_EQ(gQueue.checkpoints, 0u);
  }
  {
    fakeTable_ = fakeQueueTable(ctx.vf_);
    fakeTable_.vkGetFenceFdKHR = nullptr;
    Commands pool(fakeTable_,
                  ctx.getVkDevice(),
                  ctx.deviceQueues_.graphicsQueueFamilyIndex,
                  false,
                  false,
                  false,
                  "Export unavailable test");
    const auto handle = pool.submit(pool.acquire());
    EXPECT_EQ(exportFenceFd(pool, handle), -1);
    completeSubmissions(1);
    // Without the extension nothing was exported: the fence is still polled.
    EXPECT_TRUE(pool.isReady(handle));
    EXPECT_EQ(gQueue.checkpoints, 0u);
  }
}

TEST_F(VulkanImmediateCommandsTest, CertifyAllReportsFenceAndCheckpointErrors) {
  {
    const auto pool = makeFakePool("Certify all fence error test");
    const auto handle = pool->submit(pool->acquire());
    gQueue.waitOverride = VK_ERROR_OUT_OF_HOST_MEMORY;
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_ERROR_OUT_OF_HOST_MEMORY);
    gQueue.waitOverride.reset();
    EXPECT_FALSE(pool->isReady(handle));
    completeSubmissions(1);
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_SUCCESS);
  }
  {
    const auto pool = makeFakePool("Certify all checkpoint loss test");
    const auto exported = pool->submit(pool->acquire());
    EXPECT_GE(exportFenceFd(*pool, exported), 0);
    gQueue.submitResults = {VK_ERROR_DEVICE_LOST};
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_ERROR_DEVICE_LOST);
    EXPECT_TRUE(pool->isReady(exported));
  }
  {
    const auto pool = makeFakePool("Checkpoint fence creation test");
    const auto exported = pool->submit(pool->acquire());
    EXPECT_GE(exportFenceFd(*pool, exported), 0);
    gQueue.createFenceResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    EXPECT_EQ(pool->wait(exported, 1'000'000'000), VK_ERROR_OUT_OF_DEVICE_MEMORY);
    EXPECT_FALSE(pool->isReady(exported));
    EXPECT_EQ(pool->wait(exported, 1'000'000'000), VK_SUCCESS);
  }
  {
    // A checkpoint whose submission failed with an unknown result is never reused.
    const auto pool = makeFakePool("Abandoned checkpoints test");
    const auto exported = pool->submit(pool->acquire());
    EXPECT_GE(exportFenceFd(*pool, exported), 0);
    gQueue.submitResults = {VK_ERROR_UNKNOWN, VK_ERROR_UNKNOWN};
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_ERROR_UNKNOWN);
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_ERROR_UNKNOWN);
    const size_t attempts = gQueue.checkpoints;
    EXPECT_EQ(certifyAll(*pool, UINT64_MAX), VK_ERROR_UNKNOWN);
    EXPECT_EQ(gQueue.checkpoints, attempts);
    EXPECT_FALSE(pool->isReady(exported));
    // Free the row for the destructor: a later trusted fence covers the exported submission.
    const auto later = pool->submit(pool->acquire());
    completeSubmissions(gQueue.submitted.size());
    EXPECT_TRUE(pool->isReady(later));
    EXPECT_TRUE(pool->isReady(exported));
  }
}

TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllReportsAPoolThatCannotCertify) {
  auto& ctx = getVulkanContext();
  for (const bool failStaging : {false, true}) {
    // Both pools are on the fake queue: a device loss that one of them answers ends the other's
    // work too.
    const FakeContextPool fakeMain(*this, ctx.immediate_, "Context certify all failure test");
    const FakeContextPool fakeStaging(
        *this, ctx.stagingDevice_->immediate, "Context certify all failure test");
    const FakeContextPool& fakePool = failStaging ? fakeStaging : fakeMain;
    const auto exported = fakePool->submit(fakePool->acquire());
    EXPECT_GE(exportFenceFd(*fakePool, exported), 0);
    gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY};
    EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_ERROR_OUT_OF_HOST_MEMORY)
        << "staging " << failStaging;
    // Without time left, no checkpoint is submitted.
    EXPECT_EQ(ctx.certifyAll(0), VK_TIMEOUT) << "staging " << failStaging;
    EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_SUCCESS) << "staging " << failStaging;
    // A fence that reports device loss: nothing runs any more.
    static_cast<void>(fakePool->submit(fakePool->acquire()));
    gQueue.waitOverride = VK_ERROR_DEVICE_LOST;
    EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_ERROR_DEVICE_LOST) << "staging " << failStaging;
    EXPECT_EQ(retiredThrough(*fakeMain), submissions(*fakeMain)) << "staging " << failStaging;
    EXPECT_EQ(retiredThrough(*fakeStaging), submissions(*fakeStaging)) << "staging " << failStaging;
    gQueue.waitOverride.reset();
  }
}

TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllEndsBothPoolsWhenAFenceAnswersADeviceLoss) {
  auto& ctx = getVulkanContext();
  struct Case {
    const char* name;
    /// The submissions of each pool, in order: true for one whose fence was exported, which no
    /// fence wait can answer for.
    std::vector<bool> main;
    std::vector<bool> staging;
  };
  const Case cases[] = {
      {.name = "the main pool's fence answers; the staging pool's work was exported",
       .main = {false},
       .staging = {true}},
      {.name = "the staging pool's fence answers; the main pool's work was exported",
       .main = {true},
       .staging = {false}},
      {.name = "the main pool's fence answers; its later work was exported",
       .main = {false, true},
       .staging = {}},
  };
  const auto submitAll = [](Commands& pool, const std::vector<bool>& exports) {
    for (const bool exported : exports) {
      const auto handle = pool.submit(pool.acquire());
      if (exported) {
        EXPECT_GE(exportFenceFd(pool, handle), 0);
      }
    }
  };
  for (const Case& c : cases) {
    const FakeContextPool main(*this, ctx.immediate_, "Context device loss test");
    const FakeContextPool staging(*this, ctx.stagingDevice_->immediate, "Context device loss test");
    submitAll(*main, c.main);
    submitAll(*staging, c.staging);
    // Every fence answers that the device is lost, and no time is left for a checkpoint: the
    // exported work cannot be waited for, and still no longer runs.
    gQueue.waitOverride = VK_ERROR_DEVICE_LOST;
    EXPECT_EQ(ctx.certifyAll(0), VK_ERROR_DEVICE_LOST) << c.name;
    EXPECT_EQ(retiredThrough(*main), submissions(*main)) << c.name;
    EXPECT_EQ(retiredThrough(*staging), submissions(*staging)) << c.name;
  }
}

TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllDoesNotEndWorkForALossAFenceResetReported) {
  auto& ctx = getVulkanContext();
  for (const bool resetInMain : {true, false}) {
    // One pool's submission finished, and its fence reset reports a device loss (only remembered);
    // the other pool's submission does not finish.
    const FakeContextPool main(*this, ctx.immediate_, "Reset loss test");
    const FakeContextPool staging(*this, ctx.stagingDevice_->immediate, "Reset loss test");
    Commands& finished = resetInMain ? *main : *staging;
    Commands& pending = resetInMain ? *staging : *main;
    static_cast<void>(finished.submit(finished.acquire()));
    static_cast<void>(pending.submit(pending.acquire()));
    completeSubmissions(1);
    gQueue.resetResults = {VK_ERROR_DEVICE_LOST};
    gQueue.blockingWaitsComplete = false;
    EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_TIMEOUT) << "reset in main " << resetInMain;
    EXPECT_TRUE(finished.isDeviceLost()) << "reset in main " << resetInMain;
    EXPECT_LT(retiredThrough(pending), submissions(pending)) << "reset in main " << resetInMain;
  }
  for (const bool resetInMain : {true, false}) {
    // Both pools' submissions finished, and one fence reset reports a device loss, which is only
    // remembered: the call reports that the work finished.
    const FakeContextPool main(*this, ctx.immediate_, "Reset loss test");
    const FakeContextPool staging(*this, ctx.stagingDevice_->immediate, "Reset loss test");
    static_cast<void>(main->submit(main->acquire()));
    static_cast<void>(staging->submit(staging->acquire()));
    completeSubmissions(2);
    // The main pool resets its fence first.
    gQueue.resetResults = resetInMain ? std::vector<VkResult>{VK_ERROR_DEVICE_LOST}
                                      : std::vector<VkResult>{VK_SUCCESS, VK_ERROR_DEVICE_LOST};
    EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_SUCCESS) << "reset in main " << resetInMain;
    EXPECT_TRUE((resetInMain ? *main : *staging).isDeviceLost()) << "reset in main " << resetInMain;
    EXPECT_EQ(retiredThrough(*main), submissions(*main)) << "reset in main " << resetInMain;
    EXPECT_EQ(retiredThrough(*staging), submissions(*staging)) << "reset in main " << resetInMain;
  }
}

TEST_F(VulkanImmediateCommandsTest,
       ContextCertifyAllDoesNotEndWorkForALossACheckpointResetReported) {
  auto& ctx = getVulkanContext();
  {
    // The main pool's checkpoint fence reset reports a device loss (only remembered), and its next
    // checkpoint proves its work finished; the staging pool's exported submission is not proven.
    const FakeContextPool main(*this, ctx.immediate_, "Checkpoint reset loss test");
    const FakeContextPool staging(
        *this, ctx.stagingDevice_->immediate, "Checkpoint reset loss test");
    const auto first = main->submit(main->acquire());
    ASSERT_GE(exportFenceFd(*main, first), 0);
    ASSERT_EQ(main->wait(first, kTestTimeout), VK_SUCCESS);
    ASSERT_STREQ(checkpointState(*main, 0), "Signaled");
    const VkFence resetFails = checkpointFence(*main, 0);
    const auto unproven = staging->submit(staging->acquire());
    ASSERT_GE(exportFenceFd(*staging, unproven), 0);
    const auto second = main->submit(main->acquire());
    ASSERT_GE(exportFenceFd(*main, second), 0);
    gQueue.resetResults = {VK_ERROR_DEVICE_LOST};
    // The main pool's next checkpoint is the next submission; it completes, and the staging pool's
    // checkpoint after it does not.
    completeSubmissions(gQueue.submitted.size() + 1);
    gQueue.blockingWaitsComplete = false;
    EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_TIMEOUT);
    // The checkpoint's fence, which could not be reset, was replaced.
    EXPECT_NE(checkpointFence(*main, 0), resetFails);
    EXPECT_TRUE(main->isDeviceLost());
    EXPECT_EQ(retiredThrough(*main), submissions(*main));
    EXPECT_LT(retiredThrough(*staging), submissions(*staging));
  }
  {
    // The other order: the main pool cannot create its checkpoint fence, so its exported
    // submission is not proven; the staging pool's checkpoint fence reset reports a device loss,
    // and its next checkpoint proves its work finished.
    const FakeContextPool main(*this, ctx.immediate_, "Checkpoint reset loss test");
    const FakeContextPool staging(
        *this, ctx.stagingDevice_->immediate, "Checkpoint reset loss test");
    const auto first = staging->submit(staging->acquire());
    ASSERT_GE(exportFenceFd(*staging, first), 0);
    ASSERT_EQ(staging->wait(first, kTestTimeout), VK_SUCCESS);
    ASSERT_STREQ(checkpointState(*staging, 0), "Signaled");
    const VkFence resetFails = checkpointFence(*staging, 0);
    const auto unproven = main->submit(main->acquire());
    ASSERT_GE(exportFenceFd(*main, unproven), 0);
    const auto second = staging->submit(staging->acquire());
    ASSERT_GE(exportFenceFd(*staging, second), 0);
    gQueue.createFenceResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    gQueue.resetResults = {VK_ERROR_DEVICE_LOST};
    EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_ERROR_OUT_OF_DEVICE_MEMORY);
    EXPECT_NE(checkpointFence(*staging, 0), resetFails);
    EXPECT_TRUE(staging->isDeviceLost());
    EXPECT_EQ(retiredThrough(*staging), submissions(*staging));
    EXPECT_LT(retiredThrough(*main), submissions(*main));
  }
}

TEST_F(VulkanImmediateCommandsTest, ACheckpointWhoseCreationReportsADeviceLossIsNotRetried) {
  const auto pool = makeFakePool("Checkpoint creation loss test");
  const auto exported = pool->submit(pool->acquire());
  EXPECT_GE(exportFenceFd(*pool, exported), 0);
  // The exported submission needs a checkpoint, and creating its fence reports a device loss: the
  // loss is remembered, the call ends with an error that proves nothing, and it does not try again.
  gQueue.createFenceResult = VK_ERROR_DEVICE_LOST;
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_ERROR_UNKNOWN);
  EXPECT_EQ(checkpointFences(*pool), 0u);
  EXPECT_TRUE(pool->isDeviceLost());
  EXPECT_LT(retiredThrough(*pool), submissions(*pool));
  // A later call creates the checkpoint.
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(retiredThrough(*pool), submissions(*pool));
}

TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllEndsBothPoolsForALossABackoffAnsweredBefore) {
  auto& ctx = getVulkanContext();
  const FakeContextPool main(*this, ctx.immediate_, "Earlier device loss test");
  const FakeContextPool staging(*this, ctx.stagingDevice_->immediate, "Earlier device loss test");
  // Exported work in both pools, which only a checkpoint can prove finished, and none can be
  // submitted.
  const auto mainWork = main->submit(main->acquire());
  ASSERT_GE(exportFenceFd(*main, mainWork), 0);
  const auto stagingWork = staging->submit(staging->acquire());
  ASSERT_GE(exportFenceFd(*staging, stagingWork), 0);
  gQueue.submitResults.assign(100, VK_ERROR_OUT_OF_HOST_MEMORY);
  // A back-off of the main pool answered, before the call, that the device is lost.
  gQueue.backoffResults = {VK_ERROR_DEVICE_LOST};
  main->waitAll();
  ASSERT_EQ(main->wait(mainWork, 0), VK_ERROR_DEVICE_LOST);
  // The call takes that answer as one: the staging pool's work no longer runs either.
  EXPECT_EQ(ctx.certifyAll(0), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(staging->wait(stagingWork, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(retiredThrough(*staging), submissions(*staging));
  // Applied once: a later call does not take it again for an answer; nothing is left to wait for.
  const uint64_t answers = lossAnswers(*main);
  EXPECT_EQ(ctx.certifyAll(0), VK_SUCCESS);
  EXPECT_EQ(lossAnswers(*main), answers);
  gQueue.submitResults.clear();
}
TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllDoesNotWaitAfterALossABackoffAnsweredBefore) {
  auto& ctx = getVulkanContext();
  const FakeContextPool main(*this, ctx.immediate_, "Earlier device loss wait test");
  const FakeContextPool staging(
      *this, ctx.stagingDevice_->immediate, "Earlier device loss wait test");
  // The main pool's work has not finished. The staging pool's work is exported: only a checkpoint
  // can prove it finished, and none can be submitted.
  const auto mainWork = main->submit(main->acquire());
  const auto stagingWork = staging->submit(staging->acquire());
  ASSERT_GE(exportFenceFd(*staging, stagingWork), 0);
  gQueue.submitResults.assign(100, VK_ERROR_OUT_OF_HOST_MEMORY);
  // A back-off of the staging pool answered, before the call, that the device is lost.
  gQueue.backoffResults = {VK_ERROR_DEVICE_LOST};
  staging->waitAll();
  ASSERT_EQ(staging->wait(stagingWork, 0), VK_ERROR_DEVICE_LOST);
  // The call takes that answer at once, without waiting for the main pool's work, which no longer
  // runs either.
  gQueue.blockingWaitsComplete = false;
  const size_t blockingWaits = gQueue.blockingWaits;
  EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(gQueue.blockingWaits, blockingWaits);
  EXPECT_EQ(main->wait(mainWork, 0), VK_ERROR_DEVICE_LOST);
  gQueue.blockingWaitsComplete = true;
  gQueue.submitResults.clear();
}

TEST_F(VulkanImmediateCommandsTest, ABackoffCountsADeviceLossOnce) {
  const auto pool = makeFakePool("Back-off loss count test");
  gQueue.backoffResults = {VK_ERROR_DEVICE_LOST, VK_SUCCESS, VK_ERROR_DEVICE_LOST};
  for (int i = 0; i != 3; ++i) {
    EXPECT_NE(backoff(*pool), VK_TIMEOUT) << i;
  }
  EXPECT_EQ(lossAnswers(*pool), 1u);
}

TEST_F(VulkanImmediateCommandsTest, ALossThatAPoolAnsweredEndsTheWaitOfTheOtherPoolOfItsQueue) {
  const auto pool = makeFakePool("Shared loss test");
  const auto other = makeFakePool("Shared loss test");
  shareQueueStop(*other, *pool);
  // Exported work in both pools, which no checkpoint can prove finished.
  const auto work = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, work), 0);
  const auto otherWork = other->submit(other->acquire());
  ASSERT_GE(exportFenceFd(*other, otherWork), 0);
  gQueue.submitResults.assign(100, VK_ERROR_OUT_OF_HOST_MEMORY);
  gQueue.backoffResults = {VK_ERROR_DEVICE_LOST};
  pool->waitAll();
  ASSERT_EQ(gQueue.backoffWaits.size(), 1u);
  // The other pool's wait (as in its destructor) ends on that answer, without a back-off of its
  // own.
  other->waitAll();
  EXPECT_EQ(gQueue.backoffWaits.size(), 1u);
  EXPECT_EQ(other->wait(otherWork, 0), VK_ERROR_DEVICE_LOST);
  gQueue.submitResults.clear();
}
TEST_F(VulkanImmediateCommandsTest, ALossThatAPoolAnsweredFreesTheRowsOfTheOtherPoolOfItsQueue) {
  const auto pool = makeFakePool("Shared loss acquire test");
  const auto other = makeFakePool("Shared loss acquire test");
  shareQueueStop(*other, *pool);
  // Every row of the other pool holds exported work, which no checkpoint can prove finished.
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    ASSERT_GE(exportFenceFd(*other, other->submit(other->acquire())), 0);
  }
  const auto work = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, work), 0);
  gQueue.submitResults.assign(100, VK_ERROR_OUT_OF_HOST_MEMORY);
  gQueue.backoffResults = {VK_ERROR_DEVICE_LOST};
  pool->waitAll();
  ASSERT_EQ(gQueue.backoffWaits.size(), 1u);
  // The other pool's exhausted acquisition frees its rows on that answer, without a back-off of
  // its own.
  const auto& wrapper = other->acquire();
  EXPECT_EQ(gQueue.backoffWaits.size(), 1u);
  EXPECT_EQ(retiredThrough(*other), submissions(*other));
  other->discard(wrapper);
  gQueue.submitResults.clear();
}

TEST_F(VulkanImmediateCommandsTest,
       ALossThatAPoolAnsweredFreesTheRowsOfTheOtherPoolWhoseWaitFailed) {
  const auto pool = makeFakePool("Shared loss failed wait test");
  const auto other = makeFakePool("Shared loss failed wait test");
  shareQueueStop(*other, *pool);
  // Every row of the other pool holds work with a trusted fence; the first pool proves the device
  // lost.
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    static_cast<void>(other->submit(other->acquire()));
  }
  const auto work = pool->submit(pool->acquire());
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  ASSERT_EQ(pool->wait(work, kTestTimeout), VK_ERROR_DEVICE_LOST);
  // The other pool's exhausted acquisition waits on its oldest fence, and that wait fails: the
  // answer the first pool counted frees the rows, without a back-off.
  const DebugAborts aborts;
  gQueue.blockingWaitResults = {VK_ERROR_OUT_OF_HOST_MEMORY};
  const auto& wrapper = other->acquire();
  EXPECT_TRUE(gQueue.backoffWaits.empty());
  EXPECT_EQ(retiredThrough(*other), submissions(*other));
  other->discard(wrapper);
}

TEST_F(VulkanImmediateCommandsTest,
       ALossThatAPoolAnsweredEndsTheDrainOfTheOtherPoolWhoseWaitFailed) {
  const auto pool = makeFakePool("Shared loss failed drain test");
  const auto other = makeFakePool("Shared loss failed drain test");
  shareQueueStop(*other, *pool);
  const auto otherWork = other->submit(other->acquire());
  const auto work = pool->submit(pool->acquire());
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  ASSERT_EQ(pool->wait(work, kTestTimeout), VK_ERROR_DEVICE_LOST);
  // The other pool's drain (as in its destructor) waits on its trusted fence, and that wait fails:
  // the answer the first pool counted ends the drain, without a back-off.
  const DebugAborts aborts;
  gQueue.blockingWaitResults = {VK_ERROR_OUT_OF_HOST_MEMORY};
  other->waitAll();
  EXPECT_TRUE(gQueue.backoffWaits.empty());
  EXPECT_EQ(other->wait(otherWork, 0), VK_ERROR_DEVICE_LOST);
}

TEST_F(VulkanImmediateCommandsTest, APoolTakesALossAnswerOnce) {
  const DebugAborts aborts;
  const auto pool = makeFakePool("Loss answer once test");
  const auto other = makeFakePool("Loss answer once test");
  shareQueueStop(*pool, *other);
  // Every row of the pool holds exported work, which no checkpoint can prove finished.
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    ASSERT_GE(exportFenceFd(*pool, pool->submit(pool->acquire())), 0);
  }
  const auto otherWork = other->submit(other->acquire());
  ASSERT_GE(exportFenceFd(*other, otherWork), 0);
  gQueue.submitResults.assign(100, VK_ERROR_OUT_OF_HOST_MEMORY);
  // A back-off of the other pool answers that the device is lost.
  gQueue.backoffResults = {VK_ERROR_DEVICE_LOST};
  other->waitAll();
  ASSERT_EQ(gQueue.backoffWaits.size(), 1u);
  // The pool takes that answer, which ends its work, but resetting the row it frees fails once: it
  // then backs off once, instead of taking the same answer again, and recovers that row.
  gQueue.resetCommandBufferResults.assign(1, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  const auto& wrapper = pool->acquire();
  EXPECT_EQ(gQueue.backoffWaits.size(), 2u);
  EXPECT_EQ(retiredThrough(*pool), submissions(*pool));
  pool->discard(wrapper);
  EXPECT_EQ(aborts.in("recycleCertifiedRow"), kDebugAbortsPerFailure);
  gQueue.submitResults.clear();
}

TEST_F(VulkanImmediateCommandsTest,
       ALossProvedByARecordingThatWasNotSubmittedEndsTheWorkOfBothPools) {
  // The fence of a recording that was not submitted answers that the device is lost, or answers
  // success once a loss was reported (here by the creation of a checkpoint fence).
  for (const bool successAfterAReportedLoss : {false, true}) {
    SCOPED_TRACE(successAfterAReportedLoss);
    auto& ctx = getVulkanContext();
    const FakeContextPool main(*this, ctx.immediate_, "Dropped loss test");
    const FakeContextPool staging(*this, ctx.stagingDevice_->immediate, "Dropped loss test");
    // Exported work in both pools, which only a checkpoint can prove finished.
    const auto stagingWork = staging->submit(staging->acquire());
    ASSERT_GE(exportFenceFd(*staging, stagingWork), 0);
    const auto mainWork = main->submit(main->acquire());
    ASSERT_GE(exportFenceFd(*main, mainWork), 0);
    if (successAfterAReportedLoss) {
      gQueue.createFenceResult = VK_ERROR_DEVICE_LOST;
      ASSERT_EQ(certifyAll(*main, kTestTimeout), VK_ERROR_UNKNOWN);
      ASSERT_TRUE(main->isDeviceLost());
    } else {
      gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
    }
    const auto own = main->submit(main->acquire());
    ASSERT_TRUE(droppedTheRecording(*main));
    const uint64_t answers = lossAnswers(*main);
    gQueue.blockingWaitResults = {successAfterAReportedLoss ? VK_SUCCESS : VK_ERROR_DEVICE_LOST};
    EXPECT_EQ(main->wait(own, kTestTimeout), VK_ERROR_DEVICE_LOST);
    // One answer is counted, and every submission of the pool ends, as lost.
    EXPECT_EQ(lossAnswers(*main), answers + 1);
    EXPECT_EQ(retiredThrough(*main), submissions(*main));
    EXPECT_EQ(main->wait(mainWork, 0), VK_ERROR_DEVICE_LOST);
    // The context takes the answer at once: the staging pool's work no longer runs either, with no
    // wait or back-off of its own.
    const size_t blockingWaits = gQueue.blockingWaits;
    EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(staging->wait(stagingWork, 0), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(gQueue.blockingWaits, blockingWaits);
    EXPECT_TRUE(gQueue.backoffWaits.empty());
    EXPECT_EQ(gQueue.checkpoints, 0u);
    // Waiting again counts nothing more.
    EXPECT_EQ(main->wait(own, kTestTimeout), VK_ERROR_DEVICE_LOST);
    main->waitAll();
    EXPECT_EQ(lossAnswers(*main), answers + 1);
  }
}

TEST_F(VulkanImmediateCommandsTest, APoolSubmitsNothingAfterTheOtherPoolOfItsQueueProvedALoss) {
  const auto pool = makeFakePool("Loss before submit test");
  const auto other = makeFakePool("Loss before submit test");
  shareQueueStop(*other, *pool);
  const auto otherWork = other->submit(other->acquire());
  // The fence of the pool's submission answers that the device is lost.
  const auto work = pool->submit(pool->acquire());
  gQueue.pollResults = {VK_TIMEOUT};
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  ASSERT_EQ(pool->wait(work, kTestTimeout), VK_ERROR_DEVICE_LOST);
  // The other pool still has free rows: its next recording is not submitted, and its submitted work
  // no longer runs.
  const size_t submitCalls = gQueue.submitCalls;
  const auto later = other->submit(other->acquire());
  EXPECT_TRUE(droppedTheRecording(*other));
  EXPECT_EQ(gQueue.submitCalls, submitCalls);
  EXPECT_EQ(other->wait(otherWork, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(other->wait(later, 0), VK_ERROR_DEVICE_LOST);
}

TEST_F(VulkanImmediateCommandsTest,
       ASuccessAfterTheOtherPoolOfItsQueueProvedALossDoesNotMakeWorkSucceed) {
  const auto pool = makeFakePool("Loss before success test");
  const auto other = makeFakePool("Loss before success test");
  shareQueueStop(*other, *pool);
  const auto otherWork = other->submit(other->acquire());
  const auto work = pool->submit(pool->acquire());
  gQueue.pollResults = {VK_TIMEOUT};
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  ASSERT_EQ(pool->wait(work, kTestTimeout), VK_ERROR_DEVICE_LOST);
  // The other pool's fence then answers success, as a lost device may: the loss comes first.
  completeSubmissions(gQueue.submitted.size());
  EXPECT_EQ(other->wait(otherWork, kTestTimeout), VK_ERROR_DEVICE_LOST);
  EXPECT_TRUE(other->isReady(otherWork));
}

TEST_F(VulkanImmediateCommandsTest, AWaitAfterTheOtherPoolOfItsQueueProvedALossWaitsForNothing) {
  // The pool's work is exported and both checkpoints are pending, so a wait would wait for the
  // oldest checkpoint, which signals and takes the whole (tiny) budget; or the work keeps its
  // trusted fence, which a wait would wait on.
  for (const bool exported : {true, false}) {
    SCOPED_TRACE(exported);
    const auto pool = makeFakePool("Loss before wait test");
    const auto other = makeFakePool("Loss before wait test");
    shareQueueStop(*other, *pool);
    Commands::SubmitHandle work;
    if (exported) {
      gQueue.blockingWaitsComplete = false;
      for (int i = 0; i != 2; ++i) {
        ASSERT_GE(exportFenceFd(*pool, pool->submit(pool->acquire())), 0);
        ASSERT_EQ(certifyAll(*pool, 1'000'000), VK_TIMEOUT);
      }
      ASSERT_STREQ(checkpointState(*pool, 0), "Pending");
      ASSERT_STREQ(checkpointState(*pool, 1), "Pending");
      gQueue.blockingWaitsComplete = true;
      work = pool->submit(pool->acquire());
      ASSERT_GE(exportFenceFd(*pool, work), 0);
    } else {
      work = pool->submit(pool->acquire());
    }
    // The other pool proves the device lost.
    const auto otherWork = other->submit(other->acquire());
    gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
    ASSERT_EQ(other->wait(otherWork, kTestTimeout), VK_ERROR_DEVICE_LOST);
    // The wait returns that loss, waiting on no fence and submitting no checkpoint.
    const size_t blockingWaits = gQueue.blockingWaits;
    const size_t checkpoints = gQueue.checkpoints;
    EXPECT_EQ(pool->wait(work, 1), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(gQueue.blockingWaits, blockingWaits);
    EXPECT_EQ(gQueue.checkpoints, checkpoints);
    EXPECT_TRUE(pool->isReady(work));
  }
}

TEST_F(VulkanImmediateCommandsTest, TheFirstQueryAfterTheOtherPoolOfItsQueueProvedALossSeesIt) {
  enum class Query : uint8_t { ZeroTimeWait, IsReady, IsRetired, DroppedWait };
  for (const Query query :
       {Query::ZeroTimeWait, Query::IsReady, Query::IsRetired, Query::DroppedWait}) {
    SCOPED_TRACE(static_cast<int>(query));
    const auto pool = makeFakePool("First query after a loss test");
    const auto other = makeFakePool("First query after a loss test");
    shareQueueStop(*other, *pool);
    // Work that has not finished, with its trusted fence, or a recording that was not submitted.
    if (query == Query::DroppedWait) {
      gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
    }
    const auto work = pool->submit(pool->acquire());
    ASSERT_EQ(droppedTheRecording(*pool), query == Query::DroppedWait);
    // The other pool proves the device lost.
    const auto otherWork = other->submit(other->acquire());
    gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
    ASSERT_EQ(other->wait(otherWork, kTestTimeout), VK_ERROR_DEVICE_LOST);
    // The first query of this pool answers that loss, without a blocking wait.
    const size_t blockingWaits = gQueue.blockingWaits;
    switch (query) {
    case Query::ZeroTimeWait:
      EXPECT_EQ(pool->wait(work, 0), VK_ERROR_DEVICE_LOST);
      break;
    case Query::IsReady:
      EXPECT_TRUE(pool->isReady(work));
      break;
    case Query::IsRetired:
      EXPECT_TRUE(isRetired(*pool, work));
      break;
    case Query::DroppedWait:
      EXPECT_EQ(pool->wait(work, kTestTimeout), VK_ERROR_DEVICE_LOST);
      break;
    }
    EXPECT_EQ(gQueue.blockingWaits, blockingWaits);
    EXPECT_TRUE(pool->isDeviceLost());
  }
}

TEST_F(VulkanImmediateCommandsTest,
       ACertifyAllAfterTheOtherPoolOfItsQueueProvedALossSubmitsNothing) {
  const auto pool = makeFakePool("Loss before certifyAll test");
  const auto other = makeFakePool("Loss before certifyAll test");
  shareQueueStop(*other, *pool);
  const auto work = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, work), 0);
  const auto otherWork = other->submit(other->acquire());
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  ASSERT_EQ(other->wait(otherWork, kTestTimeout), VK_ERROR_DEVICE_LOST);
  // Only a checkpoint could prove the exported work finished: none is submitted.
  const size_t checkpoints = gQueue.checkpoints;
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(gQueue.checkpoints, checkpoints);
  EXPECT_EQ(pool->wait(work, 0), VK_ERROR_DEVICE_LOST);
}

TEST_F(VulkanImmediateCommandsTest, WaitAllOnARecordingThatWasNotSubmittedEndsOnTheOtherPoolsLoss) {
  const auto pool = makeFakePool("Loss before waitAll test");
  const auto other = makeFakePool("Loss before waitAll test");
  shareQueueStop(*other, *pool);
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto own = pool->submit(pool->acquire());
  ASSERT_TRUE(droppedTheRecording(*pool));
  const auto otherWork = other->submit(other->acquire());
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  ASSERT_EQ(other->wait(otherWork, kTestTimeout), VK_ERROR_DEVICE_LOST);
  // The wait for the recording that was not submitted ends at once: no wait on its fence (which
  // the fake queue would fail), and no back-off.
  pool->waitAll();
  EXPECT_TRUE(gQueue.backoffWaits.empty());
  EXPECT_EQ(pool->wait(own, 0), VK_ERROR_DEVICE_LOST);
}

TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllSkipsAPoolThatWasNeverCreated) {
  auto& ctx = getVulkanContext();
  // The staging device has no pool during the test; it gets its pool back afterwards, also after a
  // failed assertion.
  class PutBack {
   public:
    explicit PutBack(std::unique_ptr<Commands>& slot) : slot_(slot), pool_(std::move(slot)) {}
    PutBack(const PutBack&) = delete;
    PutBack& operator=(const PutBack&) = delete;
    ~PutBack() {
      slot_ = std::move(pool_);
    }

   private:
    std::unique_ptr<Commands>& slot_;
    std::unique_ptr<Commands> pool_;
  };
  const PutBack stagingPool(ctx.stagingDevice_->immediate);
  ASSERT_EQ(ctx.stagingDevice_->immediate, nullptr);
  const FakeContextPool main(*this, ctx.immediate_, "Missing pool test");
  static_cast<void>(main->submit(main->acquire()));
  EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_SUCCESS);
  // A device loss ends the work of the pool that exists.
  static_cast<void>(main->submit(main->acquire()));
  gQueue.waitOverride = VK_ERROR_DEVICE_LOST;
  EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(retiredThrough(*main), submissions(*main));
}

TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllBeforeThePoolsAreCreatedCertifiesNothing) {
  // A context whose device, and so whose pools, were never created has submitted nothing.
  vulkan::VulkanContext context(vulkan::VulkanContextConfig{}, nullptr);
  EXPECT_EQ(context.certifyAll(0), VK_SUCCESS);
  EXPECT_EQ(context.certifyAll(kTestTimeout), VK_SUCCESS);
}

TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllDoesNotTakeARememberedDeviceLossForAnAnswer) {
  auto& ctx = getVulkanContext();
  const FakeContextPool main(*this, ctx.immediate_, "Remembered device loss test");
  const FakeContextPool staging(
      *this, ctx.stagingDevice_->immediate, "Remembered device loss test");
  // A submission reported the loss: the staging pool remembers it, but no fence wait or checkpoint
  // answered it.
  gQueue.submitResults = {VK_ERROR_DEVICE_LOST};
  static_cast<void>(staging->submit(staging->acquire()));
  ASSERT_TRUE(staging->isDeviceLost());
  const auto exported = main->submit(main->acquire());
  EXPECT_GE(exportFenceFd(*main, exported), 0);
  // The staging pool's fence has not signaled, and no time is left for the main pool's checkpoint.
  gQueue.waitOverride = VK_TIMEOUT;
  EXPECT_EQ(ctx.certifyAll(0), VK_TIMEOUT);
  EXPECT_LT(retiredThrough(*main), submissions(*main));
}

TEST_F(VulkanImmediateCommandsTest, ContextFenceExportGoesThroughThePool) {
#if IGL_PLATFORM_ANDROID
  // A device whose pools create exportable fences, with the extension that exports them.
  iglDev_.reset();
  vulkan::VulkanContextConfig config;
  config.exportableFences = true;
  auto context = vulkan::HWDevice::createContext(config, nullptr);
  ASSERT_NE(context, nullptr);
  Result ret;
  const auto devices =
      vulkan::HWDevice::queryDevices(*context, HWDeviceQueryDesc(HWDeviceType::Unknown), &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_FALSE(devices.empty());
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* const physicalDevice = reinterpret_cast<VkPhysicalDevice>(devices[0].guid);
  uint32_t numExtensions = 0;
  context->vf_.vkEnumerateDeviceExtensionProperties(
      physicalDevice, nullptr, &numExtensions, nullptr);
  std::vector<VkExtensionProperties> extensions(numExtensions);
  context->vf_.vkEnumerateDeviceExtensionProperties(
      physicalDevice, nullptr, &numExtensions, extensions.data());
  if (std::none_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& e) {
        return strcmp(e.extensionName, VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME) == 0;
      })) {
    GTEST_SKIP() << "The physical device does not support VK_KHR_external_fence_fd";
  }
  std::array<const char*, 1> extraExtensions = {VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME};
  const auto device = vulkan::HWDevice::create(std::move(context),
                                               devices[0],
                                               0,
                                               0,
                                               extraExtensions.size(),
                                               extraExtensions.data(),
                                               nullptr,
                                               "Fence export test",
                                               &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(device, nullptr);
  auto& ctx = device->getVulkanContext();
  ASSERT_NE(ctx.vf_.vkGetFenceFdKHR, nullptr);

  // Every fence the context's pools poll or wait on is recorded.
  const ScopedTableEntry<PFN_vkWaitForFences> waits(
      ctx.vf_, &VulkanFunctionTable::vkWaitForFences, recordingWaitForFences);
  gRealWaitForFences = waits.saved();
  gWaitedFences.clear();
  Commands& pool = *ctx.immediate_;
  const auto handle = pool.submit(pool.acquire());
  const VkFence fence = pool.getVkFenceFromSubmitHandle(handle);
  ASSERT_NE(fence, VK_NULL_HANDLE);
  const int fd = ctx.getFenceFdFromSubmitHandle(handle.handle());
  // A sync file descriptor is -1 when the fence already signaled.
  EXPECT_GE(fd, -1);
  if (fd >= 0) {
    close(fd);
  }
  // The pool marked the row: the exported fence is never polled or waited on, and a checkpoint
  // proves that the submission finished.
  EXPECT_FALSE(fenceTrusted(pool, handle.bufferIndex));
  EXPECT_EQ(pool.wait(handle, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(checkpointFences(pool), 1u);
  EXPECT_EQ(std::count(gWaitedFences.begin(), gWaitedFences.end(), handleKey(fence)), 0);
#else
  GTEST_SKIP() << "VulkanContext exports fences only on Android";
#endif // IGL_PLATFORM_ANDROID
}

TEST_F(VulkanImmediateCommandsTest, AContextThatCannotExportFencesReturnsMinusOne) {
  // The shared test device does not load `VK_KHR_external_fence_fd`, and only Android exports.
  auto& ctx = getVulkanContext();
  Commands& pool = *ctx.immediate_;
  const auto handle = pool.submit(pool.acquire());
  EXPECT_EQ(ctx.getFenceFdFromSubmitHandle(handle.handle()), -1);
  // Nothing was exported: the row's fence is still used.
  EXPECT_TRUE(fenceTrusted(pool, handle.bufferIndex));
  EXPECT_EQ(pool.wait(handle, kTestTimeout), VK_SUCCESS);
}

TEST_F(VulkanImmediateCommandsTest, ContextCertifyAllCoversBothPools) {
  auto& ctx = getVulkanContext();
  auto& staging = *ctx.stagingDevice_->immediate;
  const auto stagingHandle = staging.submit(staging.acquire());
  const auto mainHandle = ctx.immediate_->submit(ctx.immediate_->acquire());
  EXPECT_EQ(ctx.certifyAll(kTestTimeout), VK_SUCCESS);
  EXPECT_STREQ(rowState(staging, stagingHandle.bufferIndex), "Free");
  EXPECT_STREQ(rowState(*ctx.immediate_, mainHandle.bufferIndex), "Free");
  EXPECT_TRUE(staging.isReady(stagingHandle));
  EXPECT_TRUE(ctx.immediate_->isReady(mainHandle));

  // A finite budget is shared by both pools.
  static_cast<void>(staging.submit(staging.acquire()));
  static_cast<void>(ctx.immediate_->submit(ctx.immediate_->acquire()));
  EXPECT_EQ(ctx.certifyAll(5'000'000'000), VK_SUCCESS);
}

TEST_F(VulkanImmediateCommandsTest,
       ARecordingThatCannotRunIsNotSubmittedWithTheOneAfterItCallTrace) {
  auto& ctx = getVulkanContext();
  const VkCommandPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = ctx.deviceQueues_.graphicsQueueFamilyIndex,
  };
  VkCommandPool scratchPool = VK_NULL_HANDLE;
  ASSERT_EQ(ctx.vf_.vkCreateCommandPool(ctx.getVkDevice(), &poolInfo, nullptr, &scratchPool),
            VK_SUCCESS);
  // The calls of the submission of a recording with a primary command buffer after it.
  const auto traceOf = [&](VkResult endResult, const std::vector<VkResult>& submitResults) {
    const VulkanFunctionTable vf = tracedTable(ctx.vf_);
    vulkan::VulkanImmediateCommands commands(vf,
                                             ctx.getVkDevice(),
                                             ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                             false,
                                             false,
                                             false,
                                             "Next command buffer trace test");
    // A primary command buffer that runs after the row's own in the same batch, as the recording
    // of a reusable command buffer does. The traced submissions never reach the device.
    const VkCommandBufferAllocateInfo allocateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = scratchPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer next = VK_NULL_HANDLE;
    EXPECT_EQ(vf.vkAllocateCommandBuffers(ctx.getVkDevice(), &allocateInfo, &next), VK_SUCCESS);
    const auto first = commands.submit(commands.acquire());
    gTrace.lines.clear();
    gTrace.endResult = endResult;
    gTrace.submitResults = submitResults;
    const auto handle = submitWithNext(commands, commands.acquire(), next);
    gTrace.submitResults.clear();
    const std::vector<std::string> lines = gTrace.lines;
    const bool dropped = endResult != VK_SUCCESS || submitResults.size() > 1;
    EXPECT_EQ(droppedTheRecording(commands), dropped);
    EXPECT_EQ(commands.wait(first), VK_SUCCESS);
    EXPECT_EQ(commands.isReady(handle), !dropped);
    ctx.vf_.vkFreeCommandBuffers(ctx.getVkDevice(), scratchPool, 1, &next);
    return lines;
  };
  const std::vector<std::string> ran = traceOf(VK_SUCCESS, {});
  const std::string withBoth =
      "vkQueueSubmit cb1+cb32 fence1 waits=1 signals=1 wait:sem0 signal:sem1";
  ASSERT_EQ(std::count(ran.begin(), ran.end(), withBoth), 1) << joinLines(ran);

  // A recording that cannot be ended is not submitted, nor is the command buffer after it.
  std::vector<std::string> expected = ran;
  expected.erase(std::find(expected.begin(), expected.end(), withBoth));
  const std::vector<std::string> notEnded = traceOf(VK_ERROR_OUT_OF_DEVICE_MEMORY, {});
  EXPECT_EQ(notEnded, expected) << joinLines(notEnded);

  // Out of memory once: the same batch, with both command buffers, is submitted again. Out of
  // memory twice: the same two tries, and then nothing.
  expected = ran;
  expected.insert(std::find(expected.begin(), expected.end(), withBoth), withBoth);
  const std::vector<std::string> outOfMemory = traceOf(VK_SUCCESS, {VK_ERROR_OUT_OF_HOST_MEMORY});
  EXPECT_EQ(outOfMemory, expected) << joinLines(outOfMemory);
  const std::vector<std::string> outOfMemoryTwice =
      traceOf(VK_SUCCESS, {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY});
  EXPECT_EQ(outOfMemoryTwice, expected) << joinLines(outOfMemoryTwice);

  ctx.vf_.vkDestroyCommandPool(ctx.getVkDevice(), scratchPool, nullptr);
}

TEST_F(VulkanImmediateCommandsTest,
       ARecordingThatIsNotSubmittedDropsItsInjectedSemaphoresCallTrace) {
  auto& ctx = getVulkanContext();
  // An opaque non-null handle: the traced submissions never reach the device.
  const auto signalSemaphore = opaqueHandle<VkSemaphore>(0x20);
  const VulkanFunctionTable vf = tracedTable(ctx.vf_);
  // Without the chain, and with an injected signal but no injected wait: nothing orders later work
  // after the recording, so the next submission is not stopped.
  vulkan::VulkanImmediateCommands commands(vf,
                                           ctx.getVkDevice(),
                                           ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                           false,
                                           false,
                                           false,
                                           "Dropped semaphores trace test",
                                           false);
  const auto first = commands.submit(commands.acquire());
  gTrace.lines.clear();
  EXPECT_TRUE(commands.signalSemaphore(signalSemaphore, 0));
  gTrace.endResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  const auto notEnded = commands.submit(commands.acquire());
  EXPECT_TRUE(droppedTheRecording(commands));
  EXPECT_FALSE(queueStopped(commands));
  const auto next = commands.submit(commands.acquire());
  EXPECT_FALSE(droppedTheRecording(commands));
  // The injected signal went with the recording that was not submitted: the next batch signals
  // nothing, and waits on nothing.
  const std::vector<std::string> expected = {
      "vkBeginCommandBuffer cb1 flags=1",
      "vkEndCommandBuffer cb1",
      "vkBeginCommandBuffer cb2 flags=1",
      "vkEndCommandBuffer cb2",
      "vkQueueSubmit cb2 fence2 waits=0 signals=0",
  };
  EXPECT_EQ(gTrace.lines, expected) << joinLines(gTrace.lines);
  EXPECT_EQ(commands.wait(next), VK_SUCCESS);
  EXPECT_TRUE(commands.isReady(first));
  EXPECT_FALSE(commands.isReady(notEnded));
}

TEST_F(VulkanImmediateCommandsTest, ADroppedRecordingIsNotMeasured) {
  if (!iglDev_->hasFeature(DeviceFeatures::Timers)) {
    GTEST_SKIP() << "Timers are not supported";
  }
  auto& ctx = getVulkanContext();
  Result ret;
  const auto timer = iglDev_->createTimer(&ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(timer, nullptr);
  const auto queue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_NE(queue, nullptr);
  const CommandBufferDesc desc{.timer = timer};

  const auto measured = queue->createCommandBuffer(desc, &ret);
  ASSERT_NE(measured, nullptr);
  EXPECT_NE(queue->submit(*measured), 0u);
  measured->waitUntilCompleted();
  ASSERT_TRUE(timer->resultsAvailable());
  const auto measurement = timer->getElapsedTimeNanos();

  const ScopedTableEntry<PFN_vkGetQueryPoolResults> countReads(
      ctx.vf_, &VulkanFunctionTable::vkGetQueryPoolResults, countingGetQueryPoolResults);
  gRealGetQueryPoolResults = countReads.saved();
  const auto dropped = queue->createCommandBuffer(desc, &ret);
  ASSERT_NE(dropped, nullptr);
  {
    const ScopedTableEntry<PFN_vkEndCommandBuffer> endFails(
        ctx.vf_, &VulkanFunctionTable::vkEndCommandBuffer, failingEndCommandBuffer);
    // It is not submitted: the caller gets its handle, which never completes.
    EXPECT_EQ(queue->submit(*dropped), ctx.immediate_->getLastSubmitHandle().handle());
  }
  EXPECT_TRUE(droppedTheRecording(*ctx.immediate_));

  // The dropped recording never wrote its timestamps: the timer keeps the last measurement and does
  // not read the dropped recording's queries.
  gQueryReads = 0;
  EXPECT_TRUE(timer->resultsAvailable());
  EXPECT_EQ(timer->getElapsedTimeNanos(), measurement);
  EXPECT_EQ(gQueryReads, 0u);
}

TEST_F(VulkanImmediateCommandsTest, ARecordingThatCouldNotBeSubmittedIsNotMeasured) {
  if (!iglDev_->hasFeature(DeviceFeatures::Timers)) {
    GTEST_SKIP() << "Timers are not supported";
  }
  auto& ctx = getVulkanContext();
  Result ret;
  const auto timer = iglDev_->createTimer(&ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(timer, nullptr);
  const auto queue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_NE(queue, nullptr);
  const CommandBufferDesc desc{.timer = timer};

  const auto measured = queue->createCommandBuffer(desc, &ret);
  ASSERT_NE(measured, nullptr);
  const SubmitHandle measuredHandle = queue->submit(*measured);
  EXPECT_NE(measuredHandle, 0u);
  measured->waitUntilCompleted();
  ASSERT_TRUE(timer->resultsAvailable());
  const auto measurement = timer->getElapsedTimeNanos();

  const ScopedTableEntry<PFN_vkGetQueryPoolResults> countReads(
      ctx.vf_, &VulkanFunctionTable::vkGetQueryPoolResults, countingGetQueryPoolResults);
  gRealGetQueryPoolResults = countReads.saved();
  const auto notSubmitted = queue->createCommandBuffer(desc, &ret);
  ASSERT_NE(notSubmitted, nullptr);
  {
    // Both tries to submit it run out of memory.
    const ScopedTableEntry<PFN_vkQueueSubmit> submitFails(
        ctx.vf_, &VulkanFunctionTable::vkQueueSubmit, outOfMemoryQueueSubmit);
    const ScopedTableEntry<PFN_vkQueueSubmit2KHR> submit2Fails(
        ctx.vf_, &VulkanFunctionTable::vkQueueSubmit2KHR, outOfMemoryQueueSubmit2);
    // Nothing is submitted: the caller gets the recording's own handle, not the earlier
    // submission's.
    const SubmitHandle handle = queue->submit(*notSubmitted);
    EXPECT_NE(handle, 0u);
    EXPECT_NE(handle, measuredHandle);
    EXPECT_EQ(handle, ctx.immediate_->getLastSubmitHandle().handle());
  }
  EXPECT_TRUE(droppedTheRecording(*ctx.immediate_));

  // Nothing ran: the timer keeps the last measurement and does not read the recording's queries.
  gQueryReads = 0;
  EXPECT_TRUE(timer->resultsAvailable());
  EXPECT_EQ(timer->getElapsedTimeNanos(), measurement);
  EXPECT_EQ(gQueryReads, 0u);
}

TEST_F(VulkanImmediateCommandsTest, AStagingCopyThatIsNotSubmittedFails) {
  auto& ctx = getVulkanContext();
  // The staging device waits with the context's fence timeout: bounded here, so that a regression
  // fails instead of hanging.
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, kTestTimeout);
  auto& staging = *ctx.stagingDevice_;
  // Without the chain, so that a copy that is not submitted does not stop the later ones.
  const ScopedValue<bool> unchained(chained(*staging.immediate), false);
  const ScopedValue<bool> unchainedContext(chained(*ctx.immediate_), false);
  Result ret;
  // A small upload is recorded as a buffer update; a large one is copied through staging blocks.
  for (const size_t size : {size_t{256}, size_t{256} * 1024}) {
    const auto buffer =
        ctx.createBuffer(size,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &ret,
                         "Dropped staging copy test");
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(buffer, nullptr);
    ASSERT_FALSE(buffer->isMapped());
    const std::vector<uint8_t> original(size, 0x5a);
    ASSERT_TRUE(staging.bufferSubData(*buffer, 0, size, original.data())) << size;
    const std::vector<uint8_t> uploaded(size, 0x3c);
    std::vector<uint8_t> readBack(size, 0xa5);
    {
      const ScopedTableEntry<PFN_vkEndCommandBuffer> endFails(
          ctx.vf_, &VulkanFunctionTable::vkEndCommandBuffer, failingEndCommandBuffer);
      EXPECT_FALSE(staging.bufferSubData(*buffer, 0, size, uploaded.data())) << size;
      const ScopedTableEntry<PFN_vkWaitForFences> waits(
          ctx.vf_, &VulkanFunctionTable::vkWaitForFences, recordingWaitForFences);
      gRealWaitForFences = waits.saved();
      gWaitedFences.clear();
      EXPECT_FALSE(staging.getBufferSubData(*buffer, 0, size, readBack.data())) << size;
      // The readback does not wait for its copy, which was not submitted.
      const VkFence droppedFence =
          staging.immediate->getVkFenceFromSubmitHandle(staging.immediate->getLastSubmitHandle());
      ASSERT_NE(droppedFence, VK_NULL_HANDLE) << size;
      EXPECT_EQ(std::count(gWaitedFences.begin(), gWaitedFences.end(), handleKey(droppedFence)), 0)
          << size;
    }
    // Nothing was read back, and the buffer keeps what it held. Copies that run succeed.
    EXPECT_EQ(std::count(readBack.begin(), readBack.end(), 0xa5), static_cast<ptrdiff_t>(size))
        << size;
    EXPECT_TRUE(staging.getBufferSubData(*buffer, 0, size, readBack.data())) << size;
    EXPECT_EQ(readBack, original) << size;
    EXPECT_TRUE(staging.bufferSubData(*buffer, 0, size, uploaded.data())) << size;
    EXPECT_TRUE(staging.getBufferSubData(*buffer, 0, size, readBack.data())) << size;
    EXPECT_EQ(readBack, uploaded) << size;
  }

  // A multi-planar image upload.
  constexpr uint32_t kPlanarSize = 4;
  const auto planar =
      ctx.createImage(VK_IMAGE_TYPE_2D,
                      VkExtent3D{.width = kPlanarSize, .height = kPlanarSize, .depth = 1},
                      vulkan::textureFormatToVkFormat(TextureFormat::YUV_NV12),
                      1,
                      1,
                      VK_IMAGE_TILING_OPTIMAL,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      0,
                      VK_SAMPLE_COUNT_1_BIT,
                      &ret,
                      "Dropped staging copy test");
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const auto planarProperties = TextureFormatProperties::fromTextureFormat(TextureFormat::YUV_NV12);
  const auto planarRange = TextureRangeDesc::new2D(0, 0, kPlanarSize, kPlanarSize);
  const std::vector<uint8_t> planarPixels(kPlanarSize * kPlanarSize * 3 / 2, 0x3c);
  {
    const ScopedTableEntry<PFN_vkEndCommandBuffer> endFails(
        ctx.vf_, &VulkanFunctionTable::vkEndCommandBuffer, failingEndCommandBuffer);
    EXPECT_FALSE(staging.imageData(planar,
                                   TextureType::TwoD,
                                   planarRange,
                                   planarProperties,
                                   0,
                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                   planarPixels.data()));
  }
  // The recording that was not submitted changed no layout.
  EXPECT_EQ(planar.imageLayout_, VK_IMAGE_LAYOUT_UNDEFINED);
  EXPECT_TRUE(staging.imageData(planar,
                                TextureType::TwoD,
                                planarRange,
                                planarProperties,
                                0,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                planarPixels.data()));
  EXPECT_EQ(planar.imageLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // A texture upload, and a readback of the texture.
  const TextureDesc textureDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                     16,
                                                     16,
                                                     TextureDesc::TextureUsageBits::Sampled |
                                                         TextureDesc::TextureUsageBits::Attachment);
  const auto texture = iglDev_->createTexture(textureDesc, &ret);
  ASSERT_NE(texture, nullptr);
  const auto range = TextureRangeDesc::new2D(0, 0, 16, 16);
  const std::vector<uint8_t> originalPixels(16 * 16 * 4, 0x5a);
  ASSERT_TRUE(texture->upload(range, originalPixels.data()).isOk());
  FramebufferDesc framebufferDesc;
  framebufferDesc.colorAttachments[0].texture = texture;
  const auto framebuffer = iglDev_->createFramebuffer(framebufferDesc, &ret);
  ASSERT_NE(framebuffer, nullptr);
  const auto queue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_NE(queue, nullptr);
  const std::vector<uint8_t> pixels(16 * 16 * 4, 0x3c);
  {
    const ScopedTableEntry<PFN_vkEndCommandBuffer> endFails(
        ctx.vf_, &VulkanFunctionTable::vkEndCommandBuffer, failingEndCommandBuffer);
    EXPECT_FALSE(texture->upload(range, pixels.data()).isOk());
  }
  // The texture keeps what it held.
  std::vector<uint8_t> pixelsBack(pixels.size(), 0xa5);
  framebuffer->copyBytesColorAttachment(*queue, 0, pixelsBack.data(), range);
  EXPECT_EQ(pixelsBack, originalPixels);
  ASSERT_TRUE(texture->upload(range, pixels.data()).isOk());

  // A texture that was never used is in the undefined layout until an upload runs.
  const auto sampled = iglDev_->createTexture(
      TextureDesc::new2D(
          TextureFormat::RGBA_UNorm8, 16, 16, TextureDesc::TextureUsageBits::Sampled),
      &ret);
  ASSERT_NE(sampled, nullptr);
  const VkImageLayout& sampledLayout =
      static_cast<vulkan::Texture&>(*sampled).getVulkanTexture().image.imageLayout_;
  ASSERT_EQ(sampledLayout, VK_IMAGE_LAYOUT_UNDEFINED);
  {
    const ScopedTableEntry<PFN_vkEndCommandBuffer> endFails(
        ctx.vf_, &VulkanFunctionTable::vkEndCommandBuffer, failingEndCommandBuffer);
    EXPECT_FALSE(sampled->upload(range, pixels.data()).isOk());
  }
  EXPECT_EQ(sampledLayout, VK_IMAGE_LAYOUT_UNDEFINED);
  ASSERT_TRUE(sampled->upload(range, pixels.data()).isOk());
  EXPECT_EQ(sampledLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  pixelsBack.assign(pixels.size(), 0xa5);
  {
    const ScopedTableEntry<PFN_vkEndCommandBuffer> endFails(
        ctx.vf_, &VulkanFunctionTable::vkEndCommandBuffer, failingEndCommandBuffer);
    framebuffer->copyBytesColorAttachment(*queue, 0, pixelsBack.data(), range);
  }
  EXPECT_EQ(std::count(pixelsBack.begin(), pixelsBack.end(), 0xa5),
            static_cast<ptrdiff_t>(pixelsBack.size()));
  framebuffer->copyBytesColorAttachment(*queue, 0, pixelsBack.data(), range);
  EXPECT_EQ(pixelsBack, pixels);
}

TEST_F(VulkanImmediateCommandsTest, AReadbackCopiesAndRestoresTheLayoutInOneRecording) {
  auto& ctx = getVulkanContext();
  // The staging device waits with the context's fence timeout: bounded here, so that a regression
  // fails instead of hanging.
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, kTestTimeout);
  Result ret;
  const TextureDesc textureDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                     16,
                                                     16,
                                                     TextureDesc::TextureUsageBits::Sampled |
                                                         TextureDesc::TextureUsageBits::Attachment);
  const auto texture = iglDev_->createTexture(textureDesc, &ret);
  ASSERT_NE(texture, nullptr);
  const auto range = TextureRangeDesc::new2D(0, 0, 16, 16);
  const std::vector<uint8_t> pixels(16 * 16 * 4, 0x3c);
  ASSERT_TRUE(texture->upload(range, pixels.data()).isOk());
  FramebufferDesc framebufferDesc;
  framebufferDesc.colorAttachments[0].texture = texture;
  const auto framebuffer = iglDev_->createFramebuffer(framebufferDesc, &ret);
  ASSERT_NE(framebuffer, nullptr);
  const auto queue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_NE(queue, nullptr);
  const VkImageLayout& trackedLayout =
      static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image.imageLayout_;
  const VkImageLayout layout = trackedLayout;
  Commands& staging = *ctx.stagingDevice_->immediate;

  const ScopedTableEntry<PFN_vkEndCommandBuffer> ends(
      ctx.vf_, &VulkanFunctionTable::vkEndCommandBuffer, stagingEndCommandBuffer);
  const ScopedTableEntry<PFN_vkQueueSubmit> submits(
      ctx.vf_, &VulkanFunctionTable::vkQueueSubmit, stagingQueueSubmit);
  const ScopedTableEntry<PFN_vkQueueSubmit2KHR> submits2(
      ctx.vf_, &VulkanFunctionTable::vkQueueSubmit2KHR, stagingQueueSubmit2);
  const ScopedTableEntry<PFN_vkCmdPipelineBarrier> barriers(
      ctx.vf_, &VulkanFunctionTable::vkCmdPipelineBarrier, stagingCmdPipelineBarrier);
  const auto readBack = [&](uint32_t failingEnd, uint32_t firstFailingSubmit) {
    gStaging = {
        .realEnd = ends.saved(),
        .realSubmit = submits.saved(),
        .realSubmit2 = submits2.saved(),
        .realBarrier = barriers.saved(),
        .failingEnd = failingEnd,
        .firstFailingSubmit = firstFailingSubmit,
    };
    std::vector<uint8_t> pixelsBack(pixels.size(), 0xa5);
    framebuffer->copyBytesColorAttachment(*queue, 0, pixelsBack.data(), range);
    return pixelsBack;
  };

  // The transition to the transfer layout, the copy and the transition back are one recording
  // and one submission.
  EXPECT_EQ(readBack(0, 0), pixels);
  EXPECT_EQ(gStaging.ends, 1u);
  EXPECT_EQ(gStaging.submits, 1u);
  EXPECT_EQ(gStaging.barriers, 2u);

  // So no second recording or submission can fail on its own: failing the second end, or the
  // second submission and its retry, changes nothing.
  EXPECT_EQ(readBack(2, 0), pixels);
  EXPECT_FALSE(droppedTheRecording(staging));
  EXPECT_EQ(readBack(0, 2), pixels);
  EXPECT_FALSE(droppedTheRecording(staging));
  EXPECT_EQ(trackedLayout, layout);

  // Without the chain from here, so that a readback that is not submitted does not stop the later
  // ones (each recording then starts with a barrier that orders it after earlier work).
  const ScopedValue<bool> unchained(chained(staging), false);
  // When that recording is not submitted because its end fails, nothing is read and the image
  // keeps its layout; a readback that runs afterwards reads the image from the layout IGL tracks.
  const std::vector<uint8_t> untouched(pixels.size(), 0xa5);
  EXPECT_EQ(readBack(1, 0), untouched);
  EXPECT_TRUE(droppedTheRecording(staging));
  EXPECT_EQ(trackedLayout, layout);
  EXPECT_EQ(readBack(0, 0), pixels);

  // The same when both tries to submit it run out of memory.
  EXPECT_EQ(readBack(0, 1), untouched);
  EXPECT_TRUE(droppedTheRecording(staging));
  EXPECT_EQ(trackedLayout, layout);
}

TEST_F(VulkanImmediateCommandsTest, AReadbackReadsOnlyACopyKnownToHaveRunAndFinished) {
  auto& ctx = getVulkanContext();
  // The staging device waits with the context's fence timeout: finite here, so that a fence that
  // never answers ends the wait.
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, kTestTimeout);
  auto& staging = *ctx.stagingDevice_;
  Result ret;
  constexpr size_t kSize = size_t{64} * 1024;
  const auto buffer =
      ctx.createBuffer(kSize,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       &ret,
                       "Unproven readback test");
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  const std::vector<uint8_t> content(kSize, 0x3c);
  ASSERT_TRUE(staging.bufferSubData(*buffer, 0, kSize, content.data()));
  const auto texture =
      iglDev_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                16,
                                                16,
                                                TextureDesc::TextureUsageBits::Sampled |
                                                    TextureDesc::TextureUsageBits::Attachment),
                             &ret);
  ASSERT_NE(texture, nullptr);
  const auto range = TextureRangeDesc::new2D(0, 0, 16, 16);
  const std::vector<uint8_t> pixels(16 * 16 * 4, 0x3c);
  ASSERT_TRUE(texture->upload(range, pixels.data()).isOk());
  FramebufferDesc framebufferDesc;
  framebufferDesc.colorAttachments[0].texture = texture;
  const auto framebuffer = iglDev_->createFramebuffer(framebufferDesc, &ret);
  ASSERT_NE(framebuffer, nullptr);
  const auto queue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_NE(queue, nullptr);

  const ScopedTableEntry<PFN_vkQueueSubmit> submits(
      ctx.vf_, &VulkanFunctionTable::vkQueueSubmit, stagingQueueSubmit);
  const ScopedTableEntry<PFN_vkQueueSubmit2KHR> submits2(
      ctx.vf_, &VulkanFunctionTable::vkQueueSubmit2KHR, stagingQueueSubmit2);
  const ScopedTableEntry<PFN_vkWaitForFences> waits(
      ctx.vf_, &VulkanFunctionTable::vkWaitForFences, stagingWaitForFences);
  const auto fault = [&](VkResult fenceAnswer, uint32_t unknownSubmit) {
    gStaging = {
        .realSubmit = submits.saved(),
        .realSubmit2 = submits2.saved(),
        .realWait = waits.saved(),
        .unknownSubmit = unknownSubmit,
        .fenceAnswer = fenceAnswer,
    };
  };
  // Without faults, every block whose copy retired is returned and every staging buffer that is
  // free again is released.
  const auto settle = [&] {
    fault(VK_SUCCESS, 0);
    staging.immediate->waitAll();
    for (int pass = 0; pass < 4; ++pass) {
      staging.mergeRegionsAndFreeBuffers();
    }
    EXPECT_EQ(staging.getFreeStagingBufferSize(), 0u);
  };
  const auto readBuffer = [&](std::vector<uint8_t>& out) {
    out.assign(kSize, 0xa5);
    return staging.getBufferSubData(*buffer, 0, kSize, out.data());
  };
  const auto readImage = [&] {
    std::vector<uint8_t> out(pixels.size(), 0xa5);
    framebuffer->copyBytesColorAttachment(*queue, 0, out.data(), range);
    return out;
  };
  const std::vector<uint8_t> untouched(kSize, 0xa5);
  const std::vector<uint8_t> untouchedPixels(pixels.size(), 0xa5);
  std::vector<uint8_t> bufferBack;

  // A readback whose copy finished returns its block to a new staging buffer.
  settle();
  ASSERT_TRUE(readBuffer(bufferBack));
  EXPECT_EQ(bufferBack, content);
  const VkDeviceSize freeAfterReadback = staging.getFreeStagingBufferSize();

  // When fence waits and polls time out or fail, the copy may still run: nothing is read, and its
  // block is not free again, not even after a merge, until the copy retires.
  for (const VkResult fenceAnswer : {VK_TIMEOUT, VK_ERROR_OUT_OF_HOST_MEMORY}) {
    settle();
    fault(fenceAnswer, 0);
    EXPECT_FALSE(readBuffer(bufferBack)) << fenceAnswer;
    EXPECT_EQ(bufferBack, untouched) << fenceAnswer;
    EXPECT_EQ(staging.getFreeStagingBufferSize(), freeAfterReadback - kSize) << fenceAnswer;
    staging.mergeRegionsAndFreeBuffers();
    EXPECT_EQ(staging.getFreeStagingBufferSize(), freeAfterReadback - kSize) << fenceAnswer;
    EXPECT_EQ(readImage(), untouchedPixels) << fenceAnswer;
  }

  // A submission that reports VK_ERROR_UNKNOWN may or may not have run, even once a checkpoint
  // proves that it retired: nothing is read, and the block is free again.
  settle();
  fault(VK_SUCCESS, 1);
  EXPECT_FALSE(readBuffer(bufferBack));
  EXPECT_EQ(bufferBack, untouched);
  EXPECT_EQ(staging.getFreeStagingBufferSize(), freeAfterReadback);
  fault(VK_SUCCESS, 1);
  EXPECT_EQ(readImage(), untouchedPixels);

  // Copies known to have run and finished are read again.
  settle();
  EXPECT_TRUE(readBuffer(bufferBack));
  EXPECT_EQ(bufferBack, content);
  EXPECT_EQ(readImage(), pixels);
}

TEST_F(VulkanImmediateCommandsTest, AReadbackAfterADeviceLossReadsNothing) {
  auto& ctx = getVulkanContext();
  // The staging device waits with the context's fence timeout: bounded here, so that a regression
  // fails instead of hanging.
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, kTestTimeout);
  auto& staging = *ctx.stagingDevice_;
  Result ret;
  constexpr size_t kSize = size_t{64} * 1024;
  const auto buffer =
      ctx.createBuffer(kSize,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       &ret,
                       "Device loss readback test");
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  const std::vector<uint8_t> content(kSize, 0x3c);
  ASSERT_TRUE(staging.bufferSubData(*buffer, 0, kSize, content.data()));
  const auto texture =
      iglDev_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                16,
                                                16,
                                                TextureDesc::TextureUsageBits::Sampled |
                                                    TextureDesc::TextureUsageBits::Attachment),
                             &ret);
  ASSERT_NE(texture, nullptr);
  const auto range = TextureRangeDesc::new2D(0, 0, 16, 16);
  const std::vector<uint8_t> pixels(16 * 16 * 4, 0x3c);
  ASSERT_TRUE(texture->upload(range, pixels.data()).isOk());
  FramebufferDesc framebufferDesc;
  framebufferDesc.colorAttachments[0].texture = texture;
  const auto framebuffer = iglDev_->createFramebuffer(framebufferDesc, &ret);
  ASSERT_NE(framebuffer, nullptr);
  const auto queue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_NE(queue, nullptr);

  // The readback's fence reports a device loss: its submission retired, but nothing it copied can
  // be trusted.
  const ScopedTableEntry<PFN_vkWaitForFences> waits(
      ctx.vf_, &VulkanFunctionTable::vkWaitForFences, stagingWaitForFences);
  gStaging = {.realWait = waits.saved(), .fenceAnswer = VK_ERROR_DEVICE_LOST};
  std::vector<uint8_t> bufferBack(kSize, 0xa5);
  EXPECT_FALSE(staging.getBufferSubData(*buffer, 0, kSize, bufferBack.data()));
  EXPECT_EQ(bufferBack, std::vector<uint8_t>(kSize, 0xa5));
  EXPECT_TRUE(staging.immediate->isDeviceLost());

  // The loss is remembered: a later readback is not submitted, and reads nothing either.
  gStaging.fenceAnswer = VK_SUCCESS;
  std::vector<uint8_t> pixelsBack(pixels.size(), 0xa5);
  framebuffer->copyBytesColorAttachment(*queue, 0, pixelsBack.data(), range);
  EXPECT_EQ(pixelsBack, std::vector<uint8_t>(pixels.size(), 0xa5));
}

TEST_F(VulkanImmediateCommandsTest,
       ARecordingOpenAcrossASubmitThatSubmittedNothingDoesNotAnswerForTheRecordingThatTakesItsRow) {
  const auto pool = makeFakePool("Submit call cohort test");
  // Without the chain, so that the recording that is not submitted does not stop the later ones.
  chained(*pool) = false;
  const auto notSubmittedTwice =
      std::vector<VkResult>{VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};

  // Control: no submit() call between the acquisitions of a released recording and of the one that
  // takes its row, which had the same submit id: the released one answers for the open one.
  const auto pending = pool->submit(pool->acquire());
  const auto& released = pool->acquire();
  const auto releasedHandle = released.handle;
  pool->release(released);
  const auto& taker = pool->acquire();
  ASSERT_EQ(taker.handle.bufferIndex, releasedHandle.bufferIndex);
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(pending));
  EXPECT_FALSE(pool->isReady(releasedHandle));
  pool->discard(taker);
  EXPECT_TRUE(pool->isReady(releasedHandle));

  // A submit() call that submitted nothing came in between: as with any submit() call, the two
  // never shared a submit id, so the released recording answers for its own prefix only, even while
  // the recording that took its row is open.
  const auto inFlight = pool->submit(pool->acquire());
  const auto& open = pool->acquire();
  const auto openHandle = open.handle;
  gQueue.submitResults = notSubmittedTwice;
  static_cast<void>(pool->submit(pool->acquire()));
  ASSERT_TRUE(droppedTheRecording(*pool));
  pool->release(open);
  const auto& secondTaker = pool->acquire();
  ASSERT_EQ(secondTaker.handle.bufferIndex, openHandle.bufferIndex);
  EXPECT_FALSE(pool->isReady(openHandle));
  completeSubmissions(2);
  EXPECT_TRUE(pool->isReady(inFlight));
  EXPECT_TRUE(pool->isReady(openHandle));
  pool->discard(secondTaker);

  // Since that call, more submit() calls were made than submissions: with no submit() call between
  // the acquisitions, a released recording still answers for the one that takes its row.
  const auto& releasedAgain = pool->acquire();
  const auto releasedAgainHandle = releasedAgain.handle;
  pool->release(releasedAgain);
  const auto& thirdTaker = pool->acquire();
  ASSERT_EQ(thirdTaker.handle.bufferIndex, releasedAgainHandle.bufferIndex);
  completeSubmissions(gQueue.submitted.size());
  EXPECT_FALSE(pool->isReady(releasedAgainHandle));
  pool->discard(thirdTaker);
  EXPECT_TRUE(pool->isReady(releasedAgainHandle));
}

TEST_F(VulkanImmediateCommandsTest,
       ARecordingThatIsNotSubmittedKeepsItsOwnHandleWhichNeverCompletes) {
  const auto pool = makeFakePool("Not submitted handle test");
  const auto before = pool->submit(pool->acquire());
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto& closed = pool->acquire();
  const auto own = closed.handle;
  const VkFence fence = closed.fence.vkFence_;
  EXPECT_EQ(pool->submit(closed), own);
  EXPECT_EQ(pool->getLastSubmitHandle(), own);
  EXPECT_EQ(lastAccepted(*pool), before);
  EXPECT_STREQ(rowState(*pool, own.bufferIndex), "Dropped");
  EXPECT_TRUE(rowAgrees(*pool, own.bufferIndex));
  EXPECT_EQ(submissions(*pool), 1u);
  EXPECT_EQ(timesSubmitted(fence), 0u);
  // Its fence is the row's, as the fence of a submission that never reached the queue; nothing
  // signals it, so it cannot be exported.
  EXPECT_EQ(pool->getVkFenceFromSubmitHandle(own), fence);
  EXPECT_EQ(exportFenceFd(*pool, own), -1);
  EXPECT_EQ(gQueue.exports, 0u);

  // Once everything submitted before it finished it retired, as it never runs, but it never
  // completes.
  EXPECT_FALSE(isRetired(*pool, own));
  EXPECT_EQ(waitRetired(*pool, own, 0), VK_TIMEOUT);
  completeSubmissions(gQueue.submitted.size());
  EXPECT_TRUE(pool->isReady(before));
  EXPECT_TRUE(isRetired(*pool, own));
  EXPECT_FALSE(pool->isReady(own));
  EXPECT_EQ(pool->wait(own, 0), VK_TIMEOUT);
  EXPECT_EQ(pool->wait(own, 1'000'000), VK_TIMEOUT);
  // A bounded certification proves everything submitted finished: the recording submitted nothing.
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(retiredThrough(*pool), submissions(*pool));
  // The row stays out of circulation: it is freed only once a device loss is certified.
  for (uint32_t i = 0; i != 2 * Commands::kMaxCommandBuffers; ++i) {
    const auto& wrapper = pool->acquire();
    EXPECT_NE(wrapper.handle.bufferIndex, own.bufferIndex);
    pool->discard(wrapper);
  }
  EXPECT_FALSE(recycleDroppedRows(*pool));
  EXPECT_STREQ(rowState(*pool, own.bufferIndex), "Dropped");

  // A wait without a time limit waits on its fence as on one that nothing signals (the fake queue
  // fails such a wait, and answers as if the fence were signaled). With no device loss reported,
  // that answer proves nothing: the wait fails, and the recording does not read as finished.
  VkResult unbounded = VK_SUCCESS;
  EXPECT_NONFATAL_FAILURE(unbounded = pool->wait(own),
                          "Unbounded wait on a fence that has no pending signal");
  EXPECT_EQ(unbounded, VK_ERROR_UNKNOWN);
  EXPECT_FALSE(pool->isDeviceLost());
  EXPECT_FALSE(pool->isReady(own));
  // An answer that the device is lost ends the wait; the recording never reads as finished.
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  EXPECT_EQ(pool->wait(own), VK_ERROR_DEVICE_LOST);
  EXPECT_TRUE(pool->isDeviceLost());
  EXPECT_TRUE(pool->isReady(own));
  EXPECT_EQ(pool->wait(own, 0), VK_ERROR_DEVICE_LOST);
}

TEST_F(VulkanImmediateCommandsTest,
       TheFenceOfARecordingThatWasNotSubmittedProvesALossByTheBackoffRule) {
  // The answers of a wait on the fence of a recording that was not submitted, which nothing
  // signals: an answer that the device is lost; a success once a loss was reported (here by the
  // creation of a checkpoint fence); a success with no loss reported, which proves nothing.
  enum class Answer : uint8_t { DeviceLost, SuccessAfterAReportedLoss, Success };
  for (const Answer answer :
       {Answer::DeviceLost, Answer::SuccessAfterAReportedLoss, Answer::Success}) {
    SCOPED_TRACE(static_cast<int>(answer));
    const auto pool = makeFakePool("Dropped fence answer test");
    // Exported work that only a checkpoint can prove finished, then a recording that cannot run.
    const auto exported = pool->submit(pool->acquire());
    ASSERT_GE(exportFenceFd(*pool, exported), 0);
    gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
    const auto own = pool->submit(pool->acquire());
    ASSERT_TRUE(droppedTheRecording(*pool));
    if (answer == Answer::SuccessAfterAReportedLoss) {
      gQueue.createFenceResult = VK_ERROR_DEVICE_LOST;
      ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_ERROR_UNKNOWN);
      ASSERT_TRUE(pool->isDeviceLost());
    }
    ASSERT_FALSE(pool->isReady(exported));
    gQueue.blockingWaitResults = {answer == Answer::DeviceLost ? VK_ERROR_DEVICE_LOST : VK_SUCCESS};
    const VkResult result = pool->wait(own, kTestTimeout);
    if (answer == Answer::Success) {
      EXPECT_EQ(result, VK_ERROR_UNKNOWN);
      EXPECT_FALSE(pool->isReady(own));
      EXPECT_FALSE(pool->isReady(exported));
      EXPECT_FALSE(pool->isDeviceLost());
    } else {
      // The loss ends every submission of the pool: the exported work no longer runs either.
      EXPECT_EQ(result, VK_ERROR_DEVICE_LOST);
      EXPECT_TRUE(pool->isReady(own));
      EXPECT_EQ(retiredThrough(*pool), submissions(*pool));
      EXPECT_EQ(pool->wait(exported, 0), VK_ERROR_DEVICE_LOST);
    }
    EXPECT_EQ(gQueue.checkpoints, 0u);
  }
}

TEST_F(VulkanImmediateCommandsTest,
       WaitAllTakesTheAnswerOfARecordingThatWasNotSubmittedByTheSameRule) {
  enum class Answer : uint8_t { DeviceLost, SuccessAfterAReportedLoss, Success };
  for (const Answer answer :
       {Answer::DeviceLost, Answer::SuccessAfterAReportedLoss, Answer::Success}) {
    SCOPED_TRACE(static_cast<int>(answer));
    const auto pool = makeFakePool("Dropped fence waitAll test");
    if (answer == Answer::SuccessAfterAReportedLoss) {
      // A fence reset that reports the loss: it is remembered, and certifies nothing.
      static_cast<void>(pool->submit(pool->acquire()));
      completeSubmissions(gQueue.submitted.size());
      gQueue.resetResults = {VK_ERROR_DEVICE_LOST};
      const DebugAborts aborts;
      ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
      ASSERT_TRUE(pool->isDeviceLost());
    } else {
      gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
    }
    // A recording that cannot run, or that comes after a reported loss, is not submitted.
    const auto own = pool->submit(pool->acquire());
    ASSERT_TRUE(droppedTheRecording(*pool));
    if (answer == Answer::Success) {
      // A success with no loss reported proves nothing: the wait goes on, backing off, until a
      // back-off reports the loss.
      gQueue.blockingWaitResults = {VK_SUCCESS, VK_SUCCESS, VK_SUCCESS};
      gQueue.backoffResults = {VK_TIMEOUT, VK_TIMEOUT, VK_ERROR_DEVICE_LOST};
    } else {
      gQueue.blockingWaitResults = {answer == Answer::DeviceLost ? VK_ERROR_DEVICE_LOST
                                                                 : VK_SUCCESS};
    }
    pool->waitAll();
    EXPECT_TRUE(gQueue.blockingWaitResults.empty());
    EXPECT_EQ(gQueue.backoffWaits.size(), answer == Answer::Success ? 3u : 0u);
    EXPECT_TRUE(pool->isReady(own));
    EXPECT_EQ(pool->wait(own, 0), VK_ERROR_DEVICE_LOST);
    EXPECT_STREQ(rowState(*pool, own.bufferIndex), "Dropped");
  }
}

TEST_F(VulkanImmediateCommandsTest,
       AFailedWaitOrResetOfARecordingThatWasNotSubmittedAssertsAsItAlwaysDid) {
  const DebugAborts aborts;
  const auto pool = makeFakePool("Dropped row assertion test");
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto own = pool->submit(pool->acquire());
  ASSERT_TRUE(droppedTheRecording(*pool));
  // A wait on its fence that fails asserts, as `waitAll()` always did; the row stays out.
  gQueue.blockingWaitResults = {VK_ERROR_OUT_OF_HOST_MEMORY};
  pool->waitAll();
  EXPECT_EQ(aborts.in("waitAll"), kDebugAbortsPerFailure);
  EXPECT_STREQ(rowState(*pool, own.bufferIndex), "Dropped");
  // Once a loss is certified the row is reset: each reset that fails asserts, and the row stays
  // out until both resets work.
  certifyQueueIdle(*pool, true);
  gQueue.resetCommandBufferResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  EXPECT_FALSE(recycleDroppedRows(*pool));
  EXPECT_EQ(aborts.in("recycleDroppedRows"), kDebugAbortsPerFailure);
  EXPECT_STREQ(rowState(*pool, own.bufferIndex), "Dropped");
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  EXPECT_FALSE(recycleDroppedRows(*pool));
  EXPECT_EQ(aborts.in("recycleDroppedRows"), 2 * kDebugAbortsPerFailure);
  EXPECT_STREQ(rowState(*pool, own.bufferIndex), "Dropped");
  EXPECT_TRUE(recycleDroppedRows(*pool));
  EXPECT_STREQ(rowState(*pool, own.bufferIndex), "Free");
}

TEST_F(VulkanImmediateCommandsTest,
       WaitAllWaitsForARecordingThatIsNotSubmittedButTheDestructorDoesNot) {
  {
    const auto pool = makeFakePool("Not submitted wait all test");
    static_cast<void>(pool->submit(pool->acquire()));
    gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
    static_cast<void>(pool->submit(pool->acquire()));
    ASSERT_TRUE(droppedTheRecording(*pool));
    completeSubmissions(gQueue.submitted.size());
    // As on the fence of a submission that never reached the queue: without a time limit. The fake
    // queue answers that wait as signaled, which proves nothing; the wait goes on until a back-off
    // reports a device loss.
    gQueue.backoffResults = {VK_ERROR_DEVICE_LOST};
    EXPECT_NONFATAL_FAILURE(pool->waitAll(),
                            "Unbounded wait on a fence that has no pending signal");
    EXPECT_EQ(gQueue.backoffWaits.size(), 1u);
  }
  // The destructor waits only for what was submitted, which finished: it makes no wait without a
  // time limit, which the fake queue would fail.
  {
    const auto pool = makeFakePool("Not submitted destructor test");
    static_cast<void>(pool->submit(pool->acquire()));
    gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
    static_cast<void>(pool->submit(pool->acquire()));
    ASSERT_TRUE(droppedTheRecording(*pool));
    completeSubmissions(gQueue.submitted.size());
  }
}

TEST_F(VulkanImmediateCommandsTest, AReusableRunThatIsNotSubmittedNeverCompletes) {
  auto& ctx = getVulkanContext();
  // The context's command buffers are submitted to the fake queue, so that run A stays in flight
  // until a blocking wait completes it.
  const FakeContextPool fakePool(*this, ctx.immediate_, "Reusable run test");
  ReusableRunsWithADroppedRun runs;
  ASSERT_NO_FATAL_FAILURE(startReusableRunsWithADroppedRun(*fakePool, runs));

  // Waiting for the reusable command buffer waits for run B, as on the fence of a submission that
  // never reached the queue: without a time limit, which the fake queue fails. Run A is not what
  // it waits for.
  EXPECT_NONFATAL_FAILURE(runs.reusable->waitUntilCompleted(),
                          "Unbounded wait on a fence that has no pending signal");
  EXPECT_EQ(gQueue.completed, 0u);
}

TEST_F(VulkanImmediateCommandsTest, AReusableRunThatIsNotSubmittedFreesItsSlotWithTheRunBeforeIt) {
  auto& ctx = getVulkanContext();
  const FakeContextPool fakePool(*this, ctx.immediate_, "Reusable slot test");
  ReusableRunsWithADroppedRun runs;
  ASSERT_NO_FATAL_FAILURE(startReusableRunsWithADroppedRun(*fakePool, runs));
  // Every slot runs again, run B's included: a run first waits for the slot's last run, which for
  // run B's slot is run A (a wait on run B would have no time limit, which the fake queue fails).
  // The runs are not submitted, as the queue stopped after run B.
  const RenderPassDesc renderPass{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}}};
  ASSERT_EQ(ctx.config_.maxResourceCount, 3u);
  for (uint32_t i = 0; i != ctx.config_.maxResourceCount; ++i) {
    // The third slot has no recording yet; the first two run theirs again.
    if (i == 0) {
      Result ret;
      auto encoder = runs.reusable->createRenderCommandEncoder(renderPass, runs.framebuffer, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      encoder->endEncoding();
    }
    EXPECT_NE(runs.queue->submit(*runs.reusable), 0u) << i;
  }
  EXPECT_EQ(gQueue.completed, 1u);
}

TEST_F(VulkanImmediateCommandsTest,
       AReusableCommandBufferWhoseLastRunIsNotSubmittedOutlivesTheRunBeforeIt) {
  auto& ctx = getVulkanContext();
  const FakeContextPool fakePool(*this, ctx.immediate_, "Reusable destruction test");
  ReusableRunsWithADroppedRun runs;
  ASSERT_NO_FATAL_FAILURE(startReusableRunsWithADroppedRun(*fakePool, runs));
  const ScopedTableEntry<PFN_vkDestroyCommandPool> countDestroys(
      ctx.vf_, &VulkanFunctionTable::vkDestroyCommandPool, countingDestroyCommandPool);
  gRealDestroyCommandPool = countDestroys.saved();
  gCommandPoolsDestroyed = 0;

  // Run A may still execute the recordings: their command pool outlives the reusable command
  // buffer until run A finished. Run B, which was not submitted, does not hold them longer.
  runs.reusable.reset();
  EXPECT_EQ(gCommandPoolsDestroyed, 0u);
  completeSubmissions(gQueue.submitted.size());
  // A submission (here not submitted, as the queue stopped) runs the deferred tasks whose work
  // retired.
  runs.queue->submit(*runs.ordinary);
  EXPECT_EQ(gCommandPoolsDestroyed, 1u);
}

TEST_F(VulkanImmediateCommandsTest, AFirstStagingCopyThatIsNotSubmittedLeavesItsBlockFree) {
  auto& ctx = getVulkanContext();
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, kTestTimeout);
  // A staging device of its own, whose pool has submitted nothing yet; without the chain, so that
  // the copies after the one that is not submitted run.
  auto staging = std::make_unique<igl::vulkan::VulkanStagingDevice>(ctx);
  // Its destructor waits without a time limit for everything submitted: when the test fails, the
  // staging device is leaked instead, so that the failure cannot hang the test.
  struct LeakOnFailure {
    explicit LeakOnFailure(std::unique_ptr<igl::vulkan::VulkanStagingDevice>& staging) :
      staging(staging) {}
    std::unique_ptr<igl::vulkan::VulkanStagingDevice>& staging;
    LeakOnFailure(const LeakOnFailure&) = delete;
    LeakOnFailure& operator=(const LeakOnFailure&) = delete;
    ~LeakOnFailure() {
      if (::testing::Test::HasFailure()) {
        [[maybe_unused]] const igl::vulkan::VulkanStagingDevice* const leaked = staging.release();
      }
    }
  };
  const LeakOnFailure leakOnFailure{staging};
  chained(*staging->immediate) = false;
  ASSERT_TRUE(staging->immediate->getLastSubmitHandle().empty());
  Result ret;
  constexpr size_t kSize = size_t{256} * 1024;
  const auto target =
      ctx.createBuffer(kSize,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       &ret,
                       "Buffer: first staging copy");
  ASSERT_NE(target, nullptr);
  const std::vector<uint8_t> data(kSize, 0x5a);
  {
    // Both tries to submit the copy run out of memory: it is not submitted.
    const ScopedTableEntry<PFN_vkQueueSubmit> submitFails(
        ctx.vf_, &VulkanFunctionTable::vkQueueSubmit, outOfMemoryQueueSubmit);
    const ScopedTableEntry<PFN_vkQueueSubmit2KHR> submit2Fails(
        ctx.vf_, &VulkanFunctionTable::vkQueueSubmit2KHR, outOfMemoryQueueSubmit2);
    EXPECT_FALSE(staging->bufferSubData(*target, 0, data.size(), data.data()));
  }
  ASSERT_TRUE(droppedTheRecording(*staging->immediate));
  ASSERT_TRUE(lastAccepted(*staging->immediate).empty());
  ASSERT_EQ(submissions(*staging->immediate), 0u);
  // The block is free again: a merge recovers the whole staging buffer, releases it, and leaves the
  // free size at 0 (it would wrap if the block were not counted as free).
  const VkDeviceSize freeAfterTheFailure = staging->getFreeStagingBufferSize();
  ASSERT_GE(freeAfterTheFailure, kSize);
  staging->mergeRegionsAndFreeBuffers();
  ASSERT_EQ(staging->getFreeStagingBufferSize(), 0u);

  // Copies that run afterwards upload and read back the data.
  ASSERT_TRUE(staging->bufferSubData(*target, 0, data.size(), data.data()));
  std::vector<uint8_t> back(kSize, 0);
  ASSERT_TRUE(staging->getBufferSubData(*target, 0, back.size(), back.data()));
  EXPECT_EQ(back, data);
  // Everything submitted is proven finished, within a time limit.
  EXPECT_EQ(certifyAll(*staging->immediate, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(retiredThrough(*staging->immediate), submissions(*staging->immediate));
}

TEST_F(VulkanImmediateCommandsTest,
       ADeferredTaskOfARecordingThatIsNotSubmittedRunsOnceTheWorkBeforeItRetired) {
  auto& ctx = getVulkanContext();
  const FakeContextPool fakePool(*this, ctx.immediate_, "Not submitted deferred task test");
  const auto queue = iglDev_->createCommandQueue({}, nullptr);
  ASSERT_NE(queue, nullptr);
  const auto before = ctx.immediate_->submit(ctx.immediate_->acquire());
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto dropped = ctx.immediate_->submit(ctx.immediate_->acquire());
  ASSERT_TRUE(droppedTheRecording(*ctx.immediate_));
  bool ran = false;
  ctx.deferredTask(std::packaged_task<void()>([&ran] { ran = true; }), dropped);

  // A frame runs the deferred tasks whose work retired (the frame is not submitted: the queue
  // stopped).
  const auto frame = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(frame, nullptr);
  queue->submit(*frame);
  EXPECT_FALSE(ran);
  completeSubmissions(1);
  EXPECT_TRUE(ctx.immediate_->isReady(before));
  const auto nextFrame = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(nextFrame, nullptr);
  queue->submit(*nextFrame);
  // The recording never runs, and never completes: its task runs once the work before it retired.
  EXPECT_TRUE(ran);
  EXPECT_FALSE(ctx.immediate_->isReady(dropped));
}

TEST_F(VulkanImmediateCommandsTest, WaitDeferredTasksKeepsTheTasksOfWorkNotProvenFinished) {
  auto& ctx = getVulkanContext();
  const FakeContextPool fakePool(*this, ctx.immediate_, "Deferred task wait test");
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, 1'000'000);
  const auto done = ctx.immediate_->submit(ctx.immediate_->acquire());
  completeSubmissions(1);
  ASSERT_TRUE(ctx.immediate_->isReady(done));
  const auto pending = ctx.immediate_->submit(ctx.immediate_->acquire());
  std::vector<int> order;
  ctx.deferredTask(std::packaged_task<void()>([&order] { order.push_back(1); }), pending);
  ctx.deferredTask(std::packaged_task<void()>([&order] { order.push_back(2); }), done);

  // The first task's work does not finish within the fence timeout: it stays queued, and so does
  // the task after it, whose work finished.
  gQueue.blockingWaitsComplete = false;
  ctx.waitDeferredTasks();
  EXPECT_TRUE(order.empty());
  gQueue.blockingWaitsComplete = true;
  ctx.waitDeferredTasks();
  EXPECT_EQ(order, (std::vector<int>{1, 2}));
}

TEST_F(VulkanImmediateCommandsTest, ADeferredTaskCanQueueAnotherOne) {
  auto& ctx = getVulkanContext();
  const FakeContextPool fakePool(*this, ctx.immediate_, "Nested deferred task test");
  const auto done = ctx.immediate_->submit(ctx.immediate_->acquire());
  completeSubmissions(1);
  std::vector<int> order;
  ctx.deferredTask(std::packaged_task<void()>([&ctx, &order, done] {
                     order.push_back(1);
                     ctx.deferredTask(std::packaged_task<void()>([&order] { order.push_back(2); }),
                                      done);
                   }),
                   done);
  ctx.waitDeferredTasks();
  EXPECT_EQ(order, (std::vector<int>{1, 2}));
}

TEST_F(VulkanImmediateCommandsTest,
       AnExhaustedAcquireReusesTheRowsOfDroppedRecordingsOnceALossIsCertified) {
  const auto pool = makeFakePool("Dropped rows reuse test");
  // The first recording cannot be ended; with the chain, every later one is not submitted either.
  gQueue.endResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    static_cast<void>(pool->submit(pool->acquire()));
    ASSERT_TRUE(droppedTheRecording(*pool));
  }
  // Every row holds a recording that was not submitted: none is freed before a loss is certified.
  EXPECT_FALSE(tryFreeRow(*pool));
  certifyQueueIdle(*pool, true);
  // Then one try of an exhausted acquisition frees them, as lost.
  EXPECT_TRUE(tryFreeRow(*pool));
  for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
    EXPECT_STREQ(rowState(*pool, row), "Free") << row;
    EXPECT_TRUE(outcomesLost(*pool, row)) << row;
  }
}

TEST_F(VulkanImmediateCommandsTest,
       AnAcquireAfterARememberedDeviceLossEndsOnceABackoffReportsTheLoss) {
  const auto pool = makeFakePool("Remembered loss test");
  // A submission reports the loss, which the pool remembers; its fence then answers as signaled, as
  // a lost device may: its row is freed as finished, and no loss is certified.
  gQueue.submitResults = {VK_ERROR_DEVICE_LOST};
  const auto lost = pool->submit(pool->acquire());
  ASSERT_TRUE(pool->isDeviceLost());
  completeSubmissions(gQueue.submitted.size());
  EXPECT_EQ(pool->wait(lost, 0), VK_SUCCESS);
  // Every later recording is not submitted. Once they hold every row, an acquisition frees a row
  // only once a loss is certified: the second back-off reports it.
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    static_cast<void>(pool->submit(pool->acquire()));
    ASSERT_TRUE(droppedTheRecording(*pool));
  }
  gQueue.backoffResults = {VK_TIMEOUT, VK_SUCCESS};
  const auto& wrapper = pool->acquire();
  EXPECT_EQ(gQueue.backoffWaits, backoffs(2));
  EXPECT_TRUE(outcomesLost(*pool, wrapper.handle.bufferIndex));
  pool->discard(wrapper);
}

TEST_F(VulkanImmediateCommandsTest, EverySubmitCallThatFailsAssertsAsItAlwaysDid) {
  const DebugAborts aborts;
  const auto pool = makeFakePool("Submit assertion test");
  chained(*pool) = false;
  // Out of memory once: the retry submits the recording, after the failed call asserted.
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY};
  static_cast<void>(pool->submit(pool->acquire()));
  EXPECT_FALSE(droppedTheRecording(*pool));
  EXPECT_EQ(aborts.in("::submit("), kDebugAbortsPerFailure);
  // Out of memory twice: both calls assert, and the recording is not submitted.
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  static_cast<void>(pool->submit(pool->acquire()));
  EXPECT_TRUE(droppedTheRecording(*pool));
  EXPECT_EQ(aborts.in("::submit("), 3 * kDebugAbortsPerFailure);
  // A failed end asserts too.
  gQueue.endResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  static_cast<void>(pool->submit(pool->acquire()));
  EXPECT_TRUE(droppedTheRecording(*pool));
  EXPECT_EQ(aborts.in("::submit("), 4 * kDebugAbortsPerFailure);
  completeSubmissions(gQueue.submitted.size());
}

TEST_F(VulkanImmediateCommandsTest, WaitingForTheRetirementOfAnOpenRecordingAssertsAsAWaitDoes) {
  const auto pool = makeFakePool("Open retirement wait test");
  const auto& open = pool->acquire();
  const DebugAborts aborts;
  EXPECT_EQ(waitRetired(*pool, open.handle, kTestTimeout), VK_ERROR_UNKNOWN);
  EXPECT_EQ(aborts.in("waitRetired"), kDebugAbortsPerFailure);
  EXPECT_EQ(pool->wait(open.handle, kTestTimeout), VK_ERROR_UNKNOWN);
  EXPECT_EQ(aborts.in("::wait("), kDebugAbortsPerFailure);
  pool->discard(open);
}

TEST_F(VulkanImmediateCommandsTest, AnOlderHandleOfARowReusedAfterALossIsNotExported) {
  const DebugAborts aborts;
  const auto pool = makeFakePool("Lost row export test");
  chained(*pool) = false;
  gQueue.endResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  const auto dropped = pool->submit(pool->acquire());
  ASSERT_TRUE(droppedTheRecording(*pool));
  certifyQueueIdle(*pool, true);
  // The other rows stay open, so the next acquisition reuses the dropped recording's row, with a
  // later submit id.
  std::vector<const Commands::CommandBufferWrapper*> open;
  for (uint32_t i = 1; i != Commands::kMaxCommandBuffers; ++i) {
    open.push_back(&pool->acquire());
  }
  const auto& reused = pool->acquire();
  ASSERT_EQ(reused.handle.bufferIndex, dropped.bufferIndex);
  ASSERT_NE(reused.handle.submitId, dropped.submitId);
  // The handle reads lost, and is refused without the assertion on a handle whose row was reused.
  const size_t abortsBefore = gDebugAborts.size();
  EXPECT_EQ(exportFenceFd(*pool, dropped), -1);
  EXPECT_EQ(gDebugAborts.size(), abortsBefore);
  EXPECT_EQ(pool->getVkFenceFromSubmitHandle(dropped), VK_NULL_HANDLE);
  EXPECT_EQ(gQueue.exports, 0u);
  pool->discard(reused);
  for (const auto* wrapper : open) {
    pool->release(*wrapper);
  }
}

TEST_F(VulkanImmediateCommandsTest,
       ABindlessUpdateAfterARecordingThatIsNotSubmittedDoesNotWaitForIt) {
  vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
  config.enableDescriptorIndexing = true;
  const auto device = util::device::vulkan::createTestDevice(config);
  ASSERT_NE(device, nullptr);
  auto& ctx = device->getVulkanContext();
  if (!ctx.config_.enableDescriptorIndexing) {
    GTEST_SKIP() << "Descriptor indexing is not supported";
  }
  Result ret;
  const auto queue = device->createCommandQueue({}, &ret);
  ASSERT_NE(queue, nullptr);
  const auto first = queue->createCommandBuffer({}, &ret);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(queue->submit(*first), 0u);
  first->waitUntilCompleted();
  // A texture created since the last update of the bindless descriptor set: the next encoder
  // updates it, after the last submission finished.
  const auto texture = device->createTexture(
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Sampled),
      &ret);
  ASSERT_NE(texture, nullptr);
  const auto dropped = queue->createCommandBuffer({}, &ret);
  ASSERT_NE(dropped, nullptr);
  {
    const ScopedTableEntry<PFN_vkEndCommandBuffer> endFails(
        ctx.vf_, &VulkanFunctionTable::vkEndCommandBuffer, failingEndCommandBuffer);
    static_cast<void>(queue->submit(*dropped));
  }
  ASSERT_TRUE(droppedTheRecording(*ctx.immediate_));
  const VkFence droppedFence =
      ctx.immediate_->getVkFenceFromSubmitHandle(ctx.immediate_->getLastSubmitHandle());
  ASSERT_NE(droppedFence, VK_NULL_HANDLE);
  // Bounded, so that a regression fails instead of hanging.
  const ScopedValue<uint64_t> timeout(ctx.config_.fenceTimeoutNanoseconds, kTestTimeout);
  const ScopedTableEntry<PFN_vkWaitForFences> waits(
      ctx.vf_, &VulkanFunctionTable::vkWaitForFences, recordingWaitForFences);
  gRealWaitForFences = waits.saved();
  gWaitedFences.clear();
  const auto next = queue->createCommandBuffer({}, &ret);
  ASSERT_NE(next, nullptr);
  auto encoder = next->createComputeCommandEncoder();
  ASSERT_NE(encoder, nullptr);
  encoder->endEncoding();
  // The update waited for the last submission that reached the queue, not for the dropped one.
  EXPECT_EQ(std::count(gWaitedFences.begin(), gWaitedFences.end(), handleKey(droppedFence)), 0);
}

TEST_F(VulkanImmediateCommandsTest,
       ARecordingThatIsNotSubmittedReadsLostOnceADeviceLossIsCertified) {
  const auto pool = makeFakePool("Not submitted recording loss test");
  // A submission that finished, whose row then takes a recording that is not submitted.
  const auto earlier = pool->submit(pool->acquire());
  completeSubmissions(1);
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto& closed = pool->acquire();
  ASSERT_EQ(closed.handle.bufferIndex, earlier.bufferIndex);
  const auto dropped = pool->submit(closed);
  ASSERT_TRUE(droppedTheRecording(*pool));
  const uint64_t droppedAcquisition = acquisitionOf(*pool, dropped.bufferIndex);
  EXPECT_FALSE(pool->isReady(dropped));

  // The queue reports a device loss: the recording reads as lost, never as finished.
  certifyQueueIdle(*pool, true);
  EXPECT_TRUE(isRetired(*pool, dropped));
  EXPECT_TRUE(pool->isReady(dropped));
  EXPECT_EQ(pool->wait(dropped, kTestTimeout), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(pool->wait(earlier, kTestTimeout), VK_SUCCESS);

  // After the loss nothing is submitted: every row ends up holding a recording that was not
  // submitted, and then they are all reused. The row of the first one is taken by an acquisition
  // whose submit id wrapped to the same value.
  for (uint32_t row = 1; row != Commands::kMaxCommandBuffers; ++row) {
    static_cast<void>(pool->submit(pool->acquire()));
    ASSERT_TRUE(droppedTheRecording(*pool));
  }
  setAcquisitionCounter(*pool, droppedAcquisition + (uint64_t{1} << 32) - 1);
  const auto& again = pool->acquire();
  ASSERT_EQ(again.handle, dropped);
  EXPECT_TRUE(outcomesLost(*pool, dropped.bufferIndex));
  // The handle of the recording that was not submitted, which the new recording's handle equals,
  // reads as lost: it never reads as finished, nor gets the fence or the descriptor of the row's
  // new recording. The finished recording is 2^32 acquisitions older than the row's new one, past
  // the horizon of 2^32 - 1: its success is no longer proven, and it reads as lost (below the
  // horizon it keeps its success: see `AProvenSuccess*`).
  EXPECT_TRUE(pool->isReady(dropped));
  EXPECT_EQ(pool->wait(dropped, kTestTimeout), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(pool->wait(earlier, kTestTimeout), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(pool->getVkFenceFromSubmitHandle(dropped), VK_NULL_HANDLE);
  EXPECT_EQ(exportFenceFd(*pool, dropped), -1);
  // A value that is not an open descriptor: the pool never closes a stored descriptor here.
  constexpr int kStoredDescriptor = 1 << 20;
  pool->storeFDInSubmitHandle(again.handle, kStoredDescriptor);
  EXPECT_EQ(pool->cachedFDFromSubmitHandle(dropped), -1);
  pool->storeFDInSubmitHandle(again.handle, -1);
  pool->discard(again);
}

TEST_F(VulkanImmediateCommandsTest, AProvenSuccessKeepsItsAnswerWhenItsDroppedRowIsReused) {
  const auto pool = makeFakePool("Proven success after a drop test");
  const auto first = pool->submit(pool->acquire());
  completeSubmissions(1);
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  ASSERT_EQ(pool->wait(first, 0), VK_SUCCESS);
  // The row's next recording cannot run: it is not submitted.
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto& again = pool->acquire();
  ASSERT_EQ(again.handle.bufferIndex, first.bufferIndex);
  const auto dropped = pool->submit(again);
  ASSERT_TRUE(droppedTheRecording(*pool));
  // A device loss is certified, and the row of the dropped recording is reused.
  certifyQueueIdle(*pool, true);
  ASSERT_TRUE(recycleDroppedRows(*pool));
  ASSERT_TRUE(outcomesLost(*pool, first.bufferIndex));
  // The success keeps its answer; the dropped recording never reads as finished.
  EXPECT_TRUE(pool->isReady(first));
  EXPECT_EQ(pool->wait(first, 0), VK_SUCCESS);
  EXPECT_EQ(pool->wait(dropped, 0), VK_ERROR_DEVICE_LOST);
  // That answer is about the work that finished, not about the device: it stays lost, a new
  // recording is not submitted, and no fence or descriptor is handed out for the old handle.
  EXPECT_TRUE(pool->isDeviceLost());
  const size_t submitCalls = gQueue.submitCalls;
  const auto later = pool->submit(pool->acquire());
  EXPECT_TRUE(droppedTheRecording(*pool));
  EXPECT_EQ(gQueue.submitCalls, submitCalls);
  EXPECT_EQ(pool->wait(later, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(pool->wait(dropped, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(pool->wait(first, 0), VK_SUCCESS);
  EXPECT_EQ(pool->getVkFenceFromSubmitHandle(first), VK_NULL_HANDLE);
  EXPECT_EQ(exportFenceFd(*pool, first), -1);
}

TEST_F(VulkanImmediateCommandsTest,
       ARecordingThatIsNotSubmittedStopsTheQueueAtTheNextSubmissionOfItsPool) {
  const auto pool = makeFakePool("Queue stop test");
  const auto other = makeFakePool("Queue stop test, other pool");
  shareQueueStop(*other, *pool);
  const auto first = pool->submit(pool->acquire());
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto dropped = pool->submit(pool->acquire());
  ASSERT_TRUE(droppedTheRecording(*pool));
  EXPECT_TRUE(chainPoisoned(*pool));
  EXPECT_FALSE(queueStopped(*pool));

  // Until this pool submits again, the other pool's submissions run.
  const size_t submitCalls = gQueue.submitCalls;
  const auto otherRuns = other->submit(other->acquire());
  EXPECT_FALSE(droppedTheRecording(*other));
  EXPECT_EQ(gQueue.submitCalls, submitCalls + 1);

  // The next submission of this pool would wait on the chain semaphore of the recording that was
  // not submitted, which nothing signals: it is not submitted either (even if it would also have
  // run out of memory), and the queue stops for both pools.
  const auto trigger = pool->submit(pool->acquire());
  EXPECT_TRUE(droppedTheRecording(*pool));
  EXPECT_TRUE(queueStopped(*other));
  const auto otherStopped = other->submit(other->acquire());
  EXPECT_TRUE(droppedTheRecording(*other));
  EXPECT_EQ(gQueue.submitCalls, submitCalls + 1);
  EXPECT_STREQ(rowState(*other, otherStopped.bufferIndex), "Dropped");
  completeSubmissions(gQueue.submitted.size());
  EXPECT_TRUE(pool->isReady(first));
  EXPECT_TRUE(other->isReady(otherRuns));
  EXPECT_FALSE(pool->isReady(dropped));
  EXPECT_FALSE(pool->isReady(trigger));
  EXPECT_FALSE(other->isReady(otherStopped));
}

TEST_F(VulkanImmediateCommandsTest, AnUnchainedRecordingThatIsNotSubmittedDoesNotStopTheQueue) {
  const auto pool = makeFakePool("Unchained queue stop test");
  chained(*pool) = false;
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  static_cast<void>(pool->submit(pool->acquire()));
  ASSERT_TRUE(droppedTheRecording(*pool));
  // Nothing waits on the recording that was not submitted: the next one runs.
  EXPECT_FALSE(chainPoisoned(*pool));
  const auto next = pool->submit(pool->acquire());
  EXPECT_FALSE(droppedTheRecording(*pool));
  EXPECT_EQ(gQueue.submitted.size(), 1u);
  completeSubmissions(1);
  EXPECT_TRUE(pool->isReady(next));
}

TEST_F(VulkanImmediateCommandsTest, ARecordingWithAnInjectedWaitThatIsNotSubmittedStopsTheQueue) {
  const auto waitSemaphore = opaqueHandle<VkSemaphore>(0x10);
  // Without the chain, an injected wait still orders the next submission after the recording.
  const auto pool = makeFakePool("Injected wait queue stop test");
  chained(*pool) = false;
  ASSERT_TRUE(pool->waitSemaphore(waitSemaphore, 0));
  gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
  static_cast<void>(pool->submit(pool->acquire()));
  ASSERT_TRUE(droppedTheRecording(*pool));
  EXPECT_TRUE(chainPoisoned(*pool));
  const size_t submitCalls = gQueue.submitCalls;
  static_cast<void>(pool->submit(pool->acquire()));
  EXPECT_TRUE(droppedTheRecording(*pool));
  EXPECT_EQ(gQueue.submitCalls, submitCalls);
  EXPECT_TRUE(queueStopped(*pool));
}

TEST_F(VulkanImmediateCommandsTest,
       ADoubleOutOfMemoryWithAnInjectedWaitStopsTheQueueOfBothPoolsAtOnce) {
  const auto waitSemaphore = opaqueHandle<VkSemaphore>(0x10);
  for (const bool isChained : {false, true}) {
    SCOPED_TRACE(isChained);
    const auto pool = makeFakePool("Double out of memory queue stop test");
    const auto other = makeFakePool("Double out of memory queue stop test, other pool");
    shareQueueStop(*other, *pool);
    chained(*pool) = isChained;
    ASSERT_TRUE(pool->waitSemaphore(waitSemaphore, 0));
    gQueue.submitResults = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY};
    static_cast<void>(pool->submit(pool->acquire()));
    ASSERT_TRUE(droppedTheRecording(*pool));
    // Later work of either pool must not run before what the injected wait orders: the queue stops
    // before the other pool submits anything.
    EXPECT_TRUE(queueStopped(*pool));
    const size_t submitCalls = gQueue.submitCalls;
    static_cast<void>(other->submit(other->acquire()));
    EXPECT_TRUE(droppedTheRecording(*other));
    EXPECT_EQ(gQueue.submitCalls, submitCalls);
  }
}

TEST_F(VulkanImmediateCommandsTest, AFailedEndWithAnInjectedWaitStopsTheQueueAtOnce) {
  const auto waitSemaphore = opaqueHandle<VkSemaphore>(0x10);
  for (const bool isChained : {true, false}) {
    const auto pool = makeFakePool("Failed end queue stop test");
    const auto other = makeFakePool("Failed end queue stop test, other pool");
    shareQueueStop(*other, *pool);
    chained(*pool) = isChained;
    ASSERT_TRUE(pool->waitSemaphore(waitSemaphore, 0));
    gQueue.endResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    static_cast<void>(pool->submit(pool->acquire()));
    ASSERT_TRUE(droppedTheRecording(*pool)) << isChained;
    // Later work of either pool must not run before what the injected wait orders: the queue stops
    // before anything else is submitted.
    EXPECT_TRUE(queueStopped(*pool)) << isChained;
    static_cast<void>(other->submit(other->acquire()));
    EXPECT_TRUE(droppedTheRecording(*other)) << isChained;
    EXPECT_EQ(gQueue.submitCalls, 0u) << isChained;
  }
}

TEST_F(VulkanImmediateCommandsTest, AFailedEndStopsTheQueueAtTheNextSubmissionOfItsPool) {
  const auto pool = makeFakePool("Failed end chain test");
  const auto other = makeFakePool("Failed end chain test, other pool");
  shareQueueStop(*other, *pool);
  gQueue.endResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  static_cast<void>(pool->submit(pool->acquire()));
  ASSERT_TRUE(droppedTheRecording(*pool));
  EXPECT_FALSE(queueStopped(*pool));
  static_cast<void>(other->submit(other->acquire()));
  EXPECT_FALSE(droppedTheRecording(*other));
  static_cast<void>(pool->submit(pool->acquire()));
  EXPECT_TRUE(droppedTheRecording(*pool));
  static_cast<void>(other->submit(other->acquire()));
  EXPECT_TRUE(droppedTheRecording(*other));
  EXPECT_EQ(gQueue.submitCalls, 1u);
  completeSubmissions(1);
}

TEST_F(VulkanImmediateCommandsTest, ACachedDeviceLossAloneCertifiesNothing) {
  const auto pool = makeFakePool("Device loss latch test");
  const auto pending = pool->submit(pool->acquire());
  gQueue.submitResults = {VK_ERROR_DEVICE_LOST};
  const auto lostSubmit = pool->submit(pool->acquire());
  ASSERT_TRUE(pool->isDeviceLost());
  // No fence, checkpoint or idle answered device loss: the cached loss proves nothing.
  EXPECT_FALSE(pool->isReady(pending));
  EXPECT_FALSE(pool->isReady(lostSubmit));
  gQueue.blockingWaitsComplete = false;
  EXPECT_EQ(pool->wait(pending, 1'000'000), VK_TIMEOUT);
  EXPECT_EQ(certifyAll(*pool, 1'000'000), VK_TIMEOUT);
  EXPECT_EQ(retiredThrough(*pool), 0u);
  gQueue.blockingWaitsComplete = true;
  completeSubmissions(2);
  EXPECT_TRUE(pool->isReady(pending));
  EXPECT_TRUE(pool->isReady(lostSubmit));
}

TEST_F(VulkanImmediateCommandsTest, AQueueIdleCertifiesBothPools) {
  auto& ctx = getVulkanContext();
  auto& staging = *ctx.stagingDevice_->immediate;
  // One submission in each pool whose fence the pool does not poll, as after an export.
  const auto mainHandle = ctx.immediate_->submit(ctx.immediate_->acquire());
  distrustFence(*ctx.immediate_, mainHandle.bufferIndex);
  const auto stagingHandle = staging.submit(staging.acquire());
  distrustFence(staging, stagingHandle.bufferIndex);
  EXPECT_FALSE(ctx.immediate_->isReady(mainHandle));
  EXPECT_FALSE(staging.isReady(stagingHandle));

  const ScopedTableEntry<PFN_vkQueueWaitIdle> idle(
      ctx.vf_, &VulkanFunctionTable::vkQueueWaitIdle, answeringQueueWaitIdle);
  gRealQueueWaitIdle = idle.saved();
  // An idle that fails proves nothing.
  gIdleResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  EXPECT_FALSE(ctx.waitIdle().isOk());
  EXPECT_FALSE(ctx.immediate_->isReady(mainHandle));
  EXPECT_FALSE(staging.isReady(stagingHandle));

  // A successful idle certifies everything both pools submitted, without a checkpoint.
  gIdleResult = VK_SUCCESS;
  EXPECT_TRUE(ctx.waitIdle().isOk());
  EXPECT_TRUE(ctx.immediate_->isReady(mainHandle));
  EXPECT_TRUE(staging.isReady(stagingHandle));
  EXPECT_EQ(retiredThrough(*ctx.immediate_), submissions(*ctx.immediate_));
  EXPECT_EQ(retiredThrough(staging), submissions(staging));
  EXPECT_EQ(checkpointFences(*ctx.immediate_), 0u);
  EXPECT_EQ(checkpointFences(staging), 0u);
}

TEST_F(VulkanImmediateCommandsTest, AQueueIdleThatLosesTheDeviceCertifiesBothPoolsAsLost) {
  auto& ctx = getVulkanContext();
  auto& staging = *ctx.stagingDevice_->immediate;
  const auto mainHandle = ctx.immediate_->submit(ctx.immediate_->acquire());
  distrustFence(*ctx.immediate_, mainHandle.bufferIndex);
  const auto stagingHandle = staging.submit(staging.acquire());
  distrustFence(staging, stagingHandle.bufferIndex);

  const ScopedTableEntry<PFN_vkQueueWaitIdle> idle(
      ctx.vf_, &VulkanFunctionTable::vkQueueWaitIdle, answeringQueueWaitIdle);
  gRealQueueWaitIdle = idle.saved();
  gIdleResult = VK_ERROR_DEVICE_LOST;
  EXPECT_FALSE(ctx.waitIdle().isOk());
  gIdleResult = VK_SUCCESS;
  EXPECT_TRUE(ctx.immediate_->isReady(mainHandle));
  EXPECT_TRUE(staging.isReady(stagingHandle));
  EXPECT_EQ(ctx.immediate_->wait(mainHandle, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(staging.wait(stagingHandle, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_TRUE(ctx.immediate_->isDeviceLost());
  // A lost device may answer success: a later idle that succeeds, and the signaled fence of a later
  // submission, change no outcome.
  EXPECT_TRUE(ctx.waitIdle().isOk());
  const auto later = ctx.immediate_->submit(ctx.immediate_->acquire());
  EXPECT_EQ(ctx.immediate_->wait(later), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(ctx.immediate_->wait(mainHandle, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(staging.wait(stagingHandle, 0), VK_ERROR_DEVICE_LOST);
}

TEST_F(VulkanImmediateCommandsTest, NothingIsSubmittedBetweenTheContextIdleAndThePoolDestruction) {
  auto& ctx = getVulkanContext();
  auto& staging = *ctx.stagingDevice_->immediate;
  // Submissions that only a checkpoint or an idle can certify.
  const auto mainHandle = ctx.immediate_->submit(ctx.immediate_->acquire());
  distrustFence(*ctx.immediate_, mainHandle.bufferIndex);
  const auto stagingHandle = staging.submit(staging.acquire());
  distrustFence(staging, stagingHandle.bufferIndex);

  // The table goes away with the context: replace its entries without restoring them.
  auto& table = const_cast<VulkanFunctionTable&>(ctx.vf_);
  gIdleSeen = false;
  gIdleResult = VK_SUCCESS;
  gSubmitsAfterIdle = 0;
  gRealQueueWaitIdle = table.vkQueueWaitIdle;
  gRealContextSubmit = table.vkQueueSubmit;
  gRealContextSubmit2 = table.vkQueueSubmit2KHR;
  table.vkQueueWaitIdle = answeringQueueWaitIdle;
  table.vkQueueSubmit = countingQueueSubmit;
  if (table.vkQueueSubmit2KHR) {
    table.vkQueueSubmit2KHR = countingQueueSubmit2;
  }
  iglDev_.reset();
  EXPECT_TRUE(gIdleSeen);
  EXPECT_EQ(gSubmitsAfterIdle, 0u);
}

TEST_F(VulkanImmediateCommandsTest,
       ADeferredTaskNotProvenAtTeardownRunsBeforeTheDeviceIsDestroyed) {
  auto& ctx = getVulkanContext();
  const DebugAborts aborts;
  // A submission that only a checkpoint can prove finished, and a task queued behind it.
  const auto handle = ctx.immediate_->submit(ctx.immediate_->acquire());
  distrustFence(*ctx.immediate_, handle.bufferIndex);
  int runs = 0;
  bool ranAfterTheProof = false;
  bool ranBeforeTheDevice = false;
  ctx.deferredTask(std::packaged_task<void()>([&runs, &ranAfterTheProof, &ranBeforeTheDevice] {
                     ++runs;
                     ranAfterTheProof = gCheckpointFailures == 0;
                     ranBeforeTheDevice = !gDeviceDestroyed;
                   }),
                   handle);
  // At teardown the queue idle fails and the first checkpoints run out of memory: the task's work
  // is not proven finished when the context drains its tasks, only later, when the pool's
  // destructor submits a checkpoint.
  auto& table = const_cast<VulkanFunctionTable&>(ctx.vf_);
  gIdleSeen = false;
  gIdleResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  gRealQueueWaitIdle = table.vkQueueWaitIdle;
  table.vkQueueWaitIdle = answeringQueueWaitIdle;
  gRealContextSubmit = table.vkQueueSubmit;
  table.vkQueueSubmit = failingCheckpointSubmit;
  gCheckpointFailures = 3;
  gDeviceDestroyed = false;
  gRealDestroyDevice = table.vkDestroyDevice;
  table.vkDestroyDevice = recordingDestroyDevice;
  iglDev_.reset();
  EXPECT_TRUE(gDeviceDestroyed);
  EXPECT_EQ(gCheckpointFailures, 0u);
  EXPECT_EQ(runs, 1);
  EXPECT_TRUE(ranAfterTheProof);
  EXPECT_TRUE(ranBeforeTheDevice);
  gIdleResult = VK_SUCCESS;
}

TEST_F(VulkanImmediateCommandsTest,
       ATaskThatARetainedTaskQueuesAtTeardownRunsBeforeTheDeviceIsDestroyed) {
  auto& ctx = getVulkanContext();
  const DebugAborts aborts;
  // A submission that only a checkpoint can prove finished, and a task queued behind it that queues
  // another task, with the default handle, when it runs.
  const auto handle = ctx.immediate_->submit(ctx.immediate_->acquire());
  distrustFence(*ctx.immediate_, handle.bufferIndex);
  vulkan::VulkanContext* const context = &ctx;
  bool poolExisted = false;
  int innerRuns = 0;
  bool innerRanBeforeTheDevice = false;
  ctx.deferredTask(
      std::packaged_task<void()>([context, &poolExisted, &innerRuns, &innerRanBeforeTheDevice] {
        // `deferredTask()` reads the pool's next submit handle: the pool must still exist.
        poolExisted = context->immediate_ != nullptr;
        if (poolExisted) {
          context->deferredTask(std::packaged_task<void()>([&innerRuns, &innerRanBeforeTheDevice] {
            ++innerRuns;
            innerRanBeforeTheDevice = !gDeviceDestroyed;
          }));
        }
      }),
      handle);
  // As in the test above: the work is proven finished only by a checkpoint of the teardown's drain.
  auto& table = const_cast<VulkanFunctionTable&>(ctx.vf_);
  gIdleSeen = false;
  gIdleResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  gRealQueueWaitIdle = table.vkQueueWaitIdle;
  table.vkQueueWaitIdle = answeringQueueWaitIdle;
  gRealContextSubmit = table.vkQueueSubmit;
  table.vkQueueSubmit = failingCheckpointSubmit;
  gCheckpointFailures = 3;
  gDeviceDestroyed = false;
  gRealDestroyDevice = table.vkDestroyDevice;
  table.vkDestroyDevice = recordingDestroyDevice;
  iglDev_.reset();
  EXPECT_TRUE(gDeviceDestroyed);
  EXPECT_TRUE(poolExisted);
  EXPECT_EQ(innerRuns, 1);
  EXPECT_TRUE(innerRanBeforeTheDevice);
  gIdleResult = VK_SUCCESS;
}

TEST_F(VulkanImmediateCommandsTest, ABufferThatARetainedTaskReleasesAtTeardownIsDestroyedFirst) {
  auto& ctx = getVulkanContext();
  const DebugAborts aborts;
  Result ret;
  std::shared_ptr<vulkan::VulkanBuffer> buffer =
      ctx.createBuffer(256,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &ret,
                       "Teardown buffer");
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  // A task queued behind work that only a checkpoint can prove finished holds the last reference to
  // a buffer. Releasing it queues the buffer's destruction, with the default handle.
  const auto handle = ctx.immediate_->submit(ctx.immediate_->acquire());
  distrustFence(*ctx.immediate_, handle.bufferIndex);
  vulkan::VulkanContext* const context = &ctx;
  size_t queuedByTheRelease = 0;
  ctx.deferredTask(std::packaged_task<void()>(
                       [context, &queuedByTheRelease, held = std::move(buffer)]() mutable {
                         const size_t before = context->deferredTasks.size();
                         held.reset();
                         queuedByTheRelease = context->deferredTasks.size() - before;
                       }),
                   handle);
  auto& table = const_cast<VulkanFunctionTable&>(ctx.vf_);
  gIdleSeen = false;
  gIdleResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  gRealQueueWaitIdle = table.vkQueueWaitIdle;
  table.vkQueueWaitIdle = answeringQueueWaitIdle;
  gRealContextSubmit = table.vkQueueSubmit;
  table.vkQueueSubmit = failingCheckpointSubmit;
  gCheckpointFailures = 3;
  gDeviceDestroyed = false;
  gRealDestroyDevice = table.vkDestroyDevice;
  table.vkDestroyDevice = recordingDestroyDevice;
  gTeardownContext = &ctx;
  gTasksLeftAfterTheSweeps = 1;
  gRealDestroyPipelineCache = table.vkDestroyPipelineCache;
  table.vkDestroyPipelineCache = recordingDestroyPipelineCache;
  iglDev_.reset();
  gTeardownContext = nullptr;
  EXPECT_TRUE(gDeviceDestroyed);
  // The buffer's destruction was queued while the pool existed, and ran in the next sweep.
  EXPECT_EQ(queuedByTheRelease, 1u);
  EXPECT_EQ(gTasksLeftAfterTheSweeps, 0u);
  gIdleResult = VK_SUCCESS;
}

TEST_F(VulkanImmediateCommandsTest, ATaskThatKeepsQueueingTasksAtTeardownIsStopped) {
  auto& ctx = getVulkanContext();
  const DebugAborts aborts;
  const auto handle = ctx.immediate_->submit(ctx.immediate_->acquire());
  distrustFence(*ctx.immediate_, handle.bufferIndex);
  int runs = 0;
  gOwnedDestroyedAfterTheDevice = false;
  auto owned = std::make_shared<OwnedByTasks>();
  const std::weak_ptr<OwnedByTasks> watch = owned;
  ctx.deferredTask(std::packaged_task<void()>(
                       RequeueingTask{.context = &ctx, .runs = &runs, .owned = std::move(owned)}),
                   handle);
  auto& table = const_cast<VulkanFunctionTable&>(ctx.vf_);
  gIdleSeen = false;
  gIdleResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  gRealQueueWaitIdle = table.vkQueueWaitIdle;
  table.vkQueueWaitIdle = answeringQueueWaitIdle;
  gRealContextSubmit = table.vkQueueSubmit;
  table.vkQueueSubmit = failingCheckpointSubmit;
  gCheckpointFailures = 3;
  gDeviceDestroyed = false;
  gRealDestroyDevice = table.vkDestroyDevice;
  table.vkDestroyDevice = recordingDestroyDevice;
  iglDev_.reset();
  // The tasks it queued ran in later sweeps, a bounded number of times; the destruction went on.
  EXPECT_TRUE(gDeviceDestroyed);
  EXPECT_GT(runs, 2);
  EXPECT_LT(runs, 100);
  EXPECT_EQ(aborts.in("~VulkanContext"), kDebugAbortsPerFailure);
  // The task left is leaked: what it owns is never destroyed, and so never after the device.
  EXPECT_FALSE(watch.expired());
  EXPECT_FALSE(gOwnedDestroyedAfterTheDevice);
  gIdleResult = VK_SUCCESS;
}

TEST_F(VulkanImmediateCommandsTest,
       ATaskThatKeepsQueueingTasksIsStoppedAtTheTeardownOfAHealthyContext) {
  auto& ctx = getVulkanContext();
  const DebugAborts aborts;
  // Queued with the default handle on a context with nothing open and no failure injected: the
  // queue idles at teardown, and the task's work is retired at once.
  int runs = 0;
  gOwnedDestroyedAfterTheDevice = false;
  auto owned = std::make_shared<OwnedByTasks>();
  const std::weak_ptr<OwnedByTasks> watch = owned;
  ctx.deferredTask(std::packaged_task<void()>(
      RequeueingTask{.context = &ctx, .runs = &runs, .owned = std::move(owned)}));
  auto& table = const_cast<VulkanFunctionTable&>(ctx.vf_);
  gDeviceDestroyed = false;
  gRealDestroyDevice = table.vkDestroyDevice;
  table.vkDestroyDevice = recordingDestroyDevice;
  iglDev_.reset();
  // It ran a bounded number of times, the destruction went on, and the task left is leaked.
  EXPECT_TRUE(gDeviceDestroyed);
  EXPECT_GT(runs, 2);
  EXPECT_LT(runs, 100);
  EXPECT_EQ(aborts.in("~VulkanContext"), kDebugAbortsPerFailure);
  EXPECT_FALSE(watch.expired());
  EXPECT_FALSE(gOwnedDestroyedAfterTheDevice);
}

TEST_F(VulkanImmediateCommandsTest, TheStagingPoolUsesTheMainPoolsQueueStateAndIsDestroyedFirst) {
  auto& ctx = getVulkanContext();
  const Commands& main = *ctx.immediate_;
  const Commands& staging = *ctx.stagingDevice_->immediate;
  // The main pool owns the queue state; the staging pool uses it.
  EXPECT_EQ(sharedQueueState(main), ownQueueState(main));
  EXPECT_EQ(sharedQueueState(staging), ownQueueState(main));
  // The context destroys the staging pool before the main pool, so nothing uses the state after it
  // is gone (the ASAN suite runs this test too).
  const VkCommandPool mainPool = commandPoolOf(main);
  const VkCommandPool stagingPool = commandPoolOf(staging);
  auto& table = const_cast<VulkanFunctionTable&>(ctx.vf_);
  gDestroyedCommandPools.clear();
  gRealDestroyCommandPool = table.vkDestroyCommandPool;
  table.vkDestroyCommandPool = recordingDestroyCommandPool;
  iglDev_.reset();
  const auto stagingAt =
      std::find(gDestroyedCommandPools.begin(), gDestroyedCommandPools.end(), stagingPool);
  const auto mainAt =
      std::find(gDestroyedCommandPools.begin(), gDestroyedCommandPools.end(), mainPool);
  ASSERT_NE(stagingAt, gDestroyedCommandPools.end());
  ASSERT_NE(mainAt, gDestroyedCommandPools.end());
  EXPECT_LT(stagingAt, mainAt);
}

TEST_F(VulkanImmediateCommandsTest, AQueueIdleReleasesPendingAndAbandonedCheckpoints) {
  const auto pool = makeFakePool("Idle checkpoint test");
  // An exported submission whose checkpoint is abandoned (an unknown result)...
  const auto first = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, first), 0);
  gQueue.submitResults = {VK_ERROR_UNKNOWN};
  EXPECT_EQ(pool->wait(first, kTestTimeout), VK_ERROR_UNKNOWN);
  // ...and one whose checkpoint is still pending.
  const auto second = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, second), 0);
  gQueue.blockingWaitsComplete = false;
  EXPECT_EQ(pool->wait(second, 1'000'000), VK_TIMEOUT);
  ASSERT_EQ(checkpointFences(*pool), 2u);
  EXPECT_STREQ(checkpointState(*pool, 0), "Abandoned");
  EXPECT_STREQ(checkpointState(*pool, 1), "Pending");

  // The queue is idle.
  completeSubmissions(gQueue.submitted.size());
  certifyQueueIdle(*pool, false);
  EXPECT_TRUE(pool->isReady(first));
  EXPECT_TRUE(pool->isReady(second));
  EXPECT_STREQ(checkpointState(*pool, 0), "Signaled");
  EXPECT_STREQ(checkpointState(*pool, 1), "Signaled");

  // After a device loss, an idle certifies as lost.
  gQueue.blockingWaitsComplete = true;
  const auto third = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, third), 0);
  certifyQueueIdle(*pool, true);
  EXPECT_TRUE(pool->isReady(third));
  EXPECT_EQ(pool->wait(third, 0), VK_ERROR_DEVICE_LOST);
  EXPECT_TRUE(pool->isDeviceLost());
}

TEST_F(VulkanImmediateCommandsTest, ExportingAHandleWithoutAFenceReturnsMinusOne) {
  const auto pool = makeFakePool("Export without fence test");
  const auto handle = pool->submit(pool->acquire());
  completeSubmissions(1);
  pool->waitAll();
  const auto& reuse = pool->acquire();
  ASSERT_EQ(reuse.handle.bufferIndex, handle.bufferIndex);
  ASSERT_EQ(pool->getVkFenceFromSubmitHandle(handle), VK_NULL_HANDLE);
  EXPECT_EQ(exportFenceFd(*pool, handle), -1);
  EXPECT_EQ(gQueue.exports, 0u);
  // A null handle has no fence either.
  EXPECT_EQ(pool->getVkFenceFromSubmitHandle({}), VK_NULL_HANDLE);
  pool->discard(reuse);
}

TEST_F(VulkanImmediateCommandsTest, ThePoolWaitsForAPendingCheckpointBeforeDestroyingIt) {
  const DebugAborts aborts;
  auto pool = makeFakePool("Pending checkpoint teardown test");
  // Exported work, whose checkpoint is still pending.
  const auto exported = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, exported), 0);
  gQueue.blockingWaitsComplete = false;
  ASSERT_EQ(certifyAll(*pool, 1'000'000), VK_TIMEOUT);
  gQueue.blockingWaitsComplete = true;
  ASSERT_STREQ(checkpointState(*pool, 0), "Pending");
  // A later fence proves everything finished. A poll of the checkpoint that fails asserts, and one
  // that does not see its signal yet leaves it pending.
  static_cast<void>(pool->submit(pool->acquire()));
  completeSubmissions(gQueue.submitted.size());
  gQueue.pollResults = {VK_ERROR_OUT_OF_HOST_MEMORY};
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(aborts.in("pollPendingCheckpoints"), kDebugAbortsPerFailure);
  EXPECT_STREQ(checkpointState(*pool, 0), "Pending");
  gQueue.pollResults = {VK_TIMEOUT};
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_STREQ(checkpointState(*pool, 0), "Pending");
  // A fence cannot be destroyed while its signal is pending: the destructor waits for it.
  gQueue.pollResults = {VK_TIMEOUT};
  const size_t blockingWaits = gQueue.blockingWaits;
  pool.reset();
  EXPECT_EQ(gQueue.blockingWaits, blockingWaits + 1);
}

TEST_F(VulkanImmediateCommandsTest, AnAcquireWithEveryRowQuarantinedBacksOffUntilOneIsRecovered) {
  const DebugAborts aborts;
  const auto pool = makeFakePool("Every row quarantined test");
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    static_cast<void>(pool->submit(pool->acquire()));
  }
  completeSubmissions(gQueue.submitted.size());
  // No finished row's fence can be reset: every row is quarantined.
  gQueue.resetResults.assign(Commands::kMaxCommandBuffers, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
    ASSERT_TRUE(quarantined(*pool, row));
  }
  // The first try recovers no row (no command buffer can be reset), and no submitted row is left to
  // wait on: the acquisition backs off, then recovers a row.
  gQueue.resetCommandBufferResults.assign(Commands::kMaxCommandBuffers,
                                          VK_ERROR_OUT_OF_DEVICE_MEMORY);
  const auto& wrapper = pool->acquire();
  EXPECT_EQ(gQueue.backoffWaits.size(), 1u);
  EXPECT_TRUE(gQueue.resetCommandBufferResults.empty());
  EXPECT_FALSE(quarantined(*pool, wrapper.handle.bufferIndex));
  pool->discard(wrapper);
}

TEST_F(VulkanImmediateCommandsTest, AZeroTimeCertifyAllOnlyPolls) {
  const auto pool = makeFakePool("Zero time certify all test");
  const auto exported = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, exported), 0);
  // Only a checkpoint could prove that the exported submission finished: none is submitted.
  EXPECT_EQ(certifyAll(*pool, 0), VK_TIMEOUT);
  EXPECT_EQ(gQueue.checkpoints, 0u);
  const auto trusted = pool->submit(pool->acquire());
  // Nothing finished: no checkpoint is submitted.
  EXPECT_EQ(certifyAll(*pool, 0), VK_TIMEOUT);
  EXPECT_EQ(gQueue.checkpoints, 0u);
  // Everything finished: the fences are polled, but nothing is reset or recycled.
  completeSubmissions(2);
  EXPECT_EQ(certifyAll(*pool, 0), VK_SUCCESS);
  EXPECT_EQ(gQueue.checkpoints, 0u);
  EXPECT_STREQ(rowState(*pool, exported.bufferIndex), "Submitted");
  EXPECT_STREQ(rowState(*pool, trusted.bufferIndex), "Submitted");
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_STREQ(rowState(*pool, exported.bufferIndex), "Free");
  EXPECT_STREQ(rowState(*pool, trusted.bufferIndex), "Free");
}

TEST_F(VulkanImmediateCommandsTest,
       ARowWhoseFenceCannotBeResetIsNotReusedWhileAnotherRowCanBeFreed) {
  const auto pool = makeFakePool("Row reset failure test");
  // An exported submission, covered by a later one.
  const auto exported = pool->submit(pool->acquire());
  const VkFence exportedFence = pool->getVkFenceFromSubmitHandle(exported);
  ASSERT_GE(exportFenceFd(*pool, exported), 0);
  static_cast<void>(pool->submit(pool->acquire()));
  completeSubmissions(2);
  // Its fence cannot be reset, and stays signaled.
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_STREQ(rowState(*pool, exported.bufferIndex), "Submitted");
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  // While another row can be freed, its row is not acquired again, and its fence is never
  // submitted again.
  for (uint32_t i = 0; i != 2 * Commands::kMaxCommandBuffers; ++i) {
    const auto handle = pool->submit(pool->acquire());
    EXPECT_NE(handle.bufferIndex, exported.bufferIndex);
    completeSubmissions(gQueue.submitted.size());
  }
  EXPECT_EQ(timesSubmitted(exportedFence), 1u);
}

TEST_F(VulkanImmediateCommandsTest, AnExhaustedAcquireRecoversAQuarantinedRowBeforeAnyWait) {
  const DebugAborts aborts;
  for (const bool replaceFence : {false, true}) {
    SCOPED_TRACE(replaceFence);
    const auto pool = makeFakePool("Quarantine before wait test");
    // A finished row whose fence could not be reset, and every other row busy with work that does
    // not finish: a wait on any of them would block (the fake fails the test if one happens).
    const auto first = pool->submit(pool->acquire());
    const VkFence firstFence = pool->getVkFenceFromSubmitHandle(first);
    completeSubmissions(1);
    gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
    ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
    ASSERT_TRUE(quarantined(*pool, first.bufferIndex));
    fillOtherRows(*pool, first.bufferIndex);
    gQueue.blockingWaitsComplete = false;
    // Its fence can be reset now, or else gets replaced: the row is taken without any wait.
    if (replaceFence) {
      gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
    }
    const size_t blockingWaits = gQueue.blockingWaits;
    const auto& again = pool->acquire();
    EXPECT_EQ(again.handle.bufferIndex, first.bufferIndex);
    EXPECT_EQ(again.fence.vkFence_ == firstFence, !replaceFence);
    EXPECT_EQ(gQueue.blockingWaits, blockingWaits);
    EXPECT_TRUE(gQueue.backoffWaits.empty());
    pool->discard(again);
    gQueue.blockingWaitsComplete = true;
    completeSubmissions(gQueue.submitted.size());
  }
}

TEST_F(VulkanImmediateCommandsTest, AnExhaustedAcquireFreesAnUntrustedRowThatALaterAnswerCovers) {
  for (const bool byCheckpoint : {false, true}) {
    SCOPED_TRACE(byCheckpoint);
    const auto pool = makeFakePool("Untrusted row revisit test");
    // The lowest row holds exported work, which the later rows prove finished: their fences, or,
    // when they are exported too, a checkpoint that is still pending.
    for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
      const auto handle = pool->submit(pool->acquire());
      if (row == 0 || byCheckpoint) {
        ASSERT_GE(exportFenceFd(*pool, handle), 0);
      }
    }
    if (byCheckpoint) {
      gQueue.blockingWaitsComplete = false;
      ASSERT_EQ(certifyAll(*pool, 1'000'000), VK_TIMEOUT);
      gQueue.blockingWaitsComplete = true;
      ASSERT_EQ(checkpointFences(*pool), 1u);
      ASSERT_STREQ(checkpointState(*pool, 0), "Pending");
    }
    completeSubmissions(gQueue.submitted.size());
    // One pass without waiting learns what covers every row, then frees them all.
    const size_t blockingWaits = gQueue.blockingWaits;
    const auto& next = pool->acquire();
    EXPECT_EQ(numAvailable(*pool), Commands::kMaxCommandBuffers - 1);
    EXPECT_EQ(gQueue.blockingWaits, blockingWaits);
    EXPECT_TRUE(gQueue.backoffWaits.empty());
    pool->discard(next);
  }
}

TEST_F(VulkanImmediateCommandsTest, ACertifyAllThatSeesADeviceLossSubmitsNoCheckpoint) {
  for (const uint64_t budget : {kTestTimeout, UINT64_MAX}) {
    SCOPED_TRACE(budget);
    const auto pool = makeFakePool("No checkpoint after a loss test");
    // A submission whose fence is waited on, then an exported one, which only a checkpoint could
    // prove finished.
    const auto fenced = pool->submit(pool->acquire());
    const auto exported = pool->submit(pool->acquire());
    ASSERT_GE(exportFenceFd(*pool, exported), 0);
    const size_t submitCalls = gQueue.submitCalls;
    gQueue.waitOverride = VK_ERROR_DEVICE_LOST;
    EXPECT_EQ(certifyAll(*pool, budget), VK_ERROR_DEVICE_LOST);
    gQueue.waitOverride.reset();
    // The loss ends both: no checkpoint is submitted to the lost device, nor waited for.
    EXPECT_EQ(gQueue.submitCalls, submitCalls);
    EXPECT_EQ(checkpointFences(*pool), 0u);
    EXPECT_EQ(pool->wait(fenced, 0), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(pool->wait(exported, 0), VK_ERROR_DEVICE_LOST);
  }
}

TEST_F(VulkanImmediateCommandsTest, ASuccessAfterACertifiedLossDoesNotMakeLostWorkSucceed) {
  const auto pool = makeFakePool("Frozen success test");
  // The first submission is proven finished; a loss is then certified.
  const auto succeeded = pool->submit(pool->acquire());
  completeSubmissions(1);
  ASSERT_EQ(pool->wait(succeeded, kTestTimeout), VK_SUCCESS);
  const auto lost = pool->submit(pool->acquire());
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  ASSERT_EQ(pool->wait(lost, kTestTimeout), VK_ERROR_DEVICE_LOST);
  // A lost device may still answer success: a queue idle that succeeds changes no outcome, and a
  // later recording is not submitted.
  certifyQueueIdle(*pool, false);
  const auto later = pool->submit(pool->acquire());
  EXPECT_TRUE(droppedTheRecording(*pool));
  completeSubmissions(gQueue.submitted.size());
  for (int i = 0; i != 2; ++i) {
    EXPECT_EQ(pool->wait(succeeded, 0), VK_SUCCESS);
    EXPECT_EQ(pool->wait(lost, 0), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(pool->wait(later, 0), VK_ERROR_DEVICE_LOST);
  }
}

TEST_F(VulkanImmediateCommandsTest, AProvenSuccessKeepsItsAnswerWhenItsRowIsReused) {
  // The row's later recording is lost, and its row is freed by recycling, or, once its fence could
  // not be reset, by quarantine recovery.
  for (const bool quarantine : {false, true}) {
    SCOPED_TRACE(quarantine);
    const DebugAborts aborts;
    const auto pool = makeFakePool("Proven success test");
    const auto first = pool->submit(pool->acquire());
    completeSubmissions(1);
    ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
    ASSERT_EQ(pool->wait(first, 0), VK_SUCCESS);
    // The row's next recording is submitted, and its fence answers that the device is lost.
    const auto& again = pool->acquire();
    ASSERT_EQ(again.handle.bufferIndex, first.bufferIndex);
    const auto lost = pool->submit(again);
    if (quarantine) {
      gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
    }
    gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
    ASSERT_EQ(pool->wait(lost, kTestTimeout), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(pool->wait(first, 0), VK_SUCCESS);
    ASSERT_EQ(quarantined(*pool, first.bufferIndex), quarantine);
    if (quarantine) {
      fillOtherRows(*pool, first.bufferIndex);
    }
    // The row is acquired a third time, and not submitted: its previous recording is the lost one.
    const auto& third = pool->acquire();
    ASSERT_EQ(third.handle.bufferIndex, first.bufferIndex);
    EXPECT_TRUE(pool->isReady(first));
    EXPECT_EQ(pool->wait(first, 0), VK_SUCCESS);
    EXPECT_EQ(pool->wait(first, kTestTimeout), VK_SUCCESS);
    // The success covers neither the later recording, which stays lost, nor the device.
    EXPECT_EQ(pool->wait(lost, 0), VK_ERROR_DEVICE_LOST);
    EXPECT_TRUE(pool->isDeviceLost());
    // Neither older recording gets the row's fence.
    EXPECT_EQ(pool->getVkFenceFromSubmitHandle(first), VK_NULL_HANDLE);
    EXPECT_EQ(pool->getVkFenceFromSubmitHandle(lost), VK_NULL_HANDLE);
    pool->discard(third);
  }
}

TEST_F(VulkanImmediateCommandsTest, AProvenSuccessKeepsItsAnswerWhenSubmitIdsWrapThroughZero) {
  const auto pool = makeFakePool("Proven success wrap test");
  // The first recording takes the last submit id before the ids wrap; 0 is skipped.
  setAcquisitionCounter(*pool, (uint64_t{1} << 32) - 2);
  const auto first = pool->submit(pool->acquire());
  ASSERT_EQ(first.submitId, 0xFFFFFFFFu);
  completeSubmissions(1);
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  const auto& again = pool->acquire();
  ASSERT_EQ(again.handle.bufferIndex, first.bufferIndex);
  ASSERT_EQ(again.handle.submitId, 1u);
  const auto lost = pool->submit(again);
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  ASSERT_EQ(pool->wait(lost, kTestTimeout), VK_ERROR_DEVICE_LOST);
  const auto& third = pool->acquire();
  ASSERT_EQ(third.handle.bufferIndex, first.bufferIndex);
  ASSERT_EQ(third.handle.submitId, 2u);
  EXPECT_EQ(pool->wait(first, 0), VK_SUCCESS);
  EXPECT_EQ(pool->wait(lost, 0), VK_ERROR_DEVICE_LOST);
  pool->discard(third);
}

TEST_F(VulkanImmediateCommandsTest, AProvenSuccessIsKeptForHandlesFewerThan2To32Minus1Older) {
  // A finished recording, then a lost one on the same row; the row's third acquisition comes 2^32 -
  // 2 acquisitions after the finished one (the ordinal 2^32, whose submit id would be 0, is
  // skipped), or 2^32 - 1: then it has the finished one's submit id.
  for (const bool atTheHorizon : {false, true}) {
    SCOPED_TRACE(atTheHorizon);
    const auto pool = makeFakePool("Proven success horizon test");
    setAcquisitionCounter(*pool, 1);
    const auto first = pool->submit(pool->acquire());
    ASSERT_EQ(acquisitionOf(*pool, first.bufferIndex), 2u);
    completeSubmissions(1);
    ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
    const auto& second = pool->acquire();
    ASSERT_EQ(second.handle.bufferIndex, first.bufferIndex);
    const auto lost = pool->submit(second);
    gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
    ASSERT_EQ(pool->wait(lost, kTestTimeout), VK_ERROR_DEVICE_LOST);
    setAcquisitionCounter(*pool, (uint64_t{1} << 32) + (atTheHorizon ? 1 : 0));
    const auto& third = pool->acquire();
    ASSERT_EQ(third.handle.bufferIndex, first.bufferIndex);
    ASSERT_EQ(third.handle.submitId == first.submitId, atTheHorizon);
    const auto current = pool->submit(third);
    completeSubmissions(gQueue.submitted.size());
    ASSERT_EQ(pool->wait(current, kTestTimeout), VK_ERROR_DEVICE_LOST);
    // Below the horizon the success is kept, though the row's previous recording was lost; at it,
    // the handle answers as the row's current recording, which came after the loss.
    EXPECT_EQ(pool->wait(first, 0), atTheHorizon ? VK_ERROR_DEVICE_LOST : VK_SUCCESS);
  }
}

TEST_F(VulkanImmediateCommandsTest, AFenceThatAnswersADeviceLossEndsEverySubmission) {
  // A trusted submission, an exported one, then another trusted one; the first fence answers that
  // the device is lost, when polled, when waited on, and when an exhausted acquisition polls it.
  for (const int how : {0, 1, 2}) {
    SCOPED_TRACE(how);
    const auto pool = makeFakePool("Fresh loss test");
    const auto first = pool->submit(pool->acquire());
    const auto exported = pool->submit(pool->acquire());
    ASSERT_GE(exportFenceFd(*pool, exported), 0);
    const auto last = pool->submit(pool->acquire());
    const size_t submitCalls = gQueue.submitCalls;
    if (how == 0) {
      gQueue.pollResults = {VK_ERROR_DEVICE_LOST};
      EXPECT_TRUE(pool->isReady(first));
    } else if (how == 1) {
      gQueue.pollResults = {VK_TIMEOUT};
      gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
      EXPECT_EQ(pool->wait(first, kTestTimeout), VK_ERROR_DEVICE_LOST);
    } else {
      for (uint32_t row = 3; row != Commands::kMaxCommandBuffers; ++row) {
        static_cast<void>(pool->submit(pool->acquire()));
      }
      gQueue.pollResults = {VK_ERROR_DEVICE_LOST};
      const auto& next = pool->acquire();
      EXPECT_EQ(numAvailable(*pool), Commands::kMaxCommandBuffers - 1);
      pool->discard(next);
    }
    EXPECT_EQ(retiredThrough(*pool), submissions(*pool));
    EXPECT_EQ(pool->wait(exported, 0), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(pool->wait(last, 0), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(gQueue.submitCalls, submitCalls + (how == 2 ? Commands::kMaxCommandBuffers - 3 : 0));
    EXPECT_EQ(checkpointFences(*pool), 0u);
  }
}

TEST_F(VulkanImmediateCommandsTest,
       AFinishedRowWhoseFenceCannotBeResetIsNotReusedWhileAnotherRowCanBeFreed) {
  const auto pool = makeFakePool("Finished row reset failure test");
  const auto first = pool->submit(pool->acquire());
  const VkFence firstFence = pool->getVkFenceFromSubmitHandle(first);
  completeSubmissions(1);
  // Its fence answers that the submission finished, but cannot be reset, and stays signaled.
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_STREQ(rowState(*pool, first.bufferIndex), "Submitted");
  EXPECT_TRUE(pool->isReady(first));
  // While another row can be freed, its row is not acquired again, and its fence is never
  // submitted again.
  for (uint32_t i = 0; i != 2 * Commands::kMaxCommandBuffers; ++i) {
    const auto handle = pool->submit(pool->acquire());
    EXPECT_NE(handle.bufferIndex, first.bufferIndex);
    completeSubmissions(gQueue.submitted.size());
  }
  EXPECT_EQ(timesSubmitted(firstFence), 1u);
}

TEST_F(VulkanImmediateCommandsTest, AQuarantinedRowIsNeverWaitedOn) {
  const auto pool = makeFakePool("Quarantined row wait test");
  const auto first = pool->submit(pool->acquire());
  const VkFence firstFence = pool->getVkFenceFromSubmitHandle(first);
  completeSubmissions(1);
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  ASSERT_STREQ(rowState(*pool, first.bufferIndex), "Submitted");
  // After the failed reset the fence's state is undefined; here it never signals again. The
  // submission finished before the reset, so nothing waits on that fence: a bounded call succeeds,
  // and an unbounded one returns (the fake queue fails the test on an unbounded wait it cannot
  // end).
  gQueue.fenceSubmission.erase(handleKey(firstFence));
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  pool->waitAll();
  EXPECT_TRUE(pool->isReady(first));
}

TEST_F(VulkanImmediateCommandsTest, AQuarantinedRowIsReusedWhenNoOtherRowIsFree) {
  const auto pool = makeFakePool("Quarantined row recovery test");
  const auto first = pool->submit(pool->acquire());
  const VkFence firstFence = pool->getVkFenceFromSubmitHandle(first);
  completeSubmissions(1);
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  ASSERT_TRUE(quarantined(*pool, first.bufferIndex));

  // Every other row holds a submission that does not finish: the acquisition resets the
  // quarantined row's fence again, which works now, and takes the row with the same fence.
  fillOtherRows(*pool, first.bufferIndex);
  gQueue.blockingWaitsComplete = false;
  const auto& again = pool->acquire();
  EXPECT_EQ(again.handle.bufferIndex, first.bufferIndex);
  EXPECT_EQ(again.fence.vkFence_, firstFence);
  EXPECT_FALSE(quarantined(*pool, first.bufferIndex));
  pool->discard(again);
  gQueue.blockingWaitsComplete = true;
}

TEST_F(VulkanImmediateCommandsTest, AQuarantinedRowGetsANewFenceWhenItsFenceStillCannotBeReset) {
  const auto pool = makeFakePool("Quarantined row fence replacement test", true);
  const auto first = pool->submit(pool->acquire());
  const VkFence firstFence = pool->getVkFenceFromSubmitHandle(first);
  completeSubmissions(1);
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  fillOtherRows(*pool, first.bufferIndex);
  gQueue.blockingWaitsComplete = false;

  // The fence still cannot be reset when the row is taken: the row gets a new fence, exportable
  // like the old one, and the old fence is destroyed.
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const size_t exportableCreates = gQueue.exportableFenceCreates;
  const auto& again = pool->acquire();
  EXPECT_EQ(again.handle.bufferIndex, first.bufferIndex);
  EXPECT_NE(again.fence.vkFence_, firstFence);
  EXPECT_TRUE(again.fence.exportable());
  EXPECT_EQ(gQueue.exportableFenceCreates, exportableCreates + 1);
  pool->discard(again);
  gQueue.blockingWaitsComplete = true;
}

TEST_F(VulkanImmediateCommandsTest, AQuarantinedRowStaysOutWhileNoNewFenceCanBeCreated) {
  const auto pool = makeFakePool("Quarantined row creation failure test");
  const auto first = pool->submit(pool->acquire());
  const VkFence firstFence = pool->getVkFenceFromSubmitHandle(first);
  completeSubmissions(1);
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  ASSERT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  // Every other row holds exported work, which only a checkpoint can prove finished, and no
  // checkpoint can be submitted: no other row can be freed.
  for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
    if (row != first.bufferIndex) {
      ASSERT_GE(exportFenceFd(*pool, pool->submit(pool->acquire())), 0);
    }
  }
  gQueue.submitResults.assign(100, VK_ERROR_OUT_OF_HOST_MEMORY);

  // At the first try neither a reset nor a new fence works: the row stays quarantined, and the
  // acquisition tries again after a pause, when the reset works.
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  gQueue.createFenceResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  const auto& again = pool->acquire();
  EXPECT_EQ(gQueue.backoffWaits, backoffs(1));
  EXPECT_FALSE(gQueue.createFenceResult.has_value());
  EXPECT_EQ(again.handle.bufferIndex, first.bufferIndex);
  EXPECT_EQ(again.fence.vkFence_, firstFence);
  pool->discard(again);
  gQueue.submitResults.clear();
}

TEST_F(VulkanImmediateCommandsTest, ARowWhoseCommandBufferCannotBeResetIsNotReusedUntilItCanBe) {
  const auto pool = makeFakePool("Command buffer reset failure test");
  const auto first = pool->submit(pool->acquire());
  completeSubmissions(1);
  gQueue.resetCommandBufferResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  // The submission finished, but its row stays out of circulation.
  EXPECT_TRUE(pool->isReady(first));
  EXPECT_TRUE(quarantined(*pool, first.bufferIndex));
  const auto& next = pool->acquire();
  EXPECT_NE(next.handle.bufferIndex, first.bufferIndex);
  pool->discard(next);

  // A recording discarded when its command buffer cannot be reset is kept out the same way. Its
  // handle waits for the submission before it, never on its own fence, which was never submitted.
  const auto pending = pool->submit(pool->acquire());
  const auto& discarded = pool->acquire();
  const auto discardedHandle = discarded.handle;
  const uint32_t discardedRow = discardedHandle.bufferIndex;
  gQueue.resetCommandBufferResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  pool->discard(discarded);
  EXPECT_TRUE(quarantined(*pool, discardedRow));
  EXPECT_STRNE(rowState(*pool, discardedRow), "Free");
  EXPECT_FALSE(pool->isReady(discardedHandle));
  EXPECT_EQ(pool->wait(discardedHandle, kTestTimeout), VK_SUCCESS);
  EXPECT_TRUE(pool->isReady(pending));

  // Once no other row is free, the rows are reset again. The first one's command buffer still
  // cannot be reset, so it stays out and the other one is reused.
  for (uint32_t row = 0; row != Commands::kMaxCommandBuffers; ++row) {
    if (row != first.bufferIndex && row != discardedRow) {
      static_cast<void>(pool->submit(pool->acquire()));
    }
  }
  gQueue.blockingWaitsComplete = false;
  gQueue.resetCommandBufferResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  const auto& again = pool->acquire();
  EXPECT_EQ(again.handle.bufferIndex, std::max(first.bufferIndex, discardedRow));
  EXPECT_TRUE(quarantined(*pool, std::min(first.bufferIndex, discardedRow)));
  pool->discard(again);
  gQueue.blockingWaitsComplete = true;
}

TEST_F(VulkanImmediateCommandsTest, ARowWhoseFenceAnswersADeviceLossIsReused) {
  const auto pool = makeFakePool("Lost row reuse test");
  std::vector<Commands::SubmitHandle> handles;
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    handles.push_back(pool->submit(pool->acquire()));
  }
  // Nothing finished, and every fence now answers a device loss: the exhausted acquisition frees
  // every row whose fence answers, as lost, and reuses the oldest.
  gQueue.waitOverride = VK_ERROR_DEVICE_LOST;
  const auto& wrapper = pool->acquire();
  EXPECT_EQ(wrapper.handle.bufferIndex, handles.front().bufferIndex);
  EXPECT_EQ(pool->wait(handles.front(), kTestTimeout), VK_ERROR_DEVICE_LOST);
  EXPECT_STREQ(rowState(*pool, handles.back().bufferIndex), "Free");
  EXPECT_EQ(pool->wait(handles.back(), kTestTimeout), VK_ERROR_DEVICE_LOST);
  pool->discard(wrapper);
  // Nothing is left to wait for.
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  gQueue.waitOverride.reset();
}
TEST_F(VulkanImmediateCommandsTest, TheOldestRowIsReusedAtOnceWhenItsWaitAnswersADeviceLoss) {
  const auto pool = makeFakePool("Lost oldest row test");
  std::vector<Commands::SubmitHandle> handles;
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    handles.push_back(pool->submit(pool->acquire()));
  }
  // Nothing has finished when polled, and the wait on the oldest submission answers a device loss:
  // that certifies it as lost, and its row is reused at once, without a back-off.
  gQueue.blockingWaitResults = {VK_ERROR_DEVICE_LOST};
  const auto& wrapper = pool->acquire();
  EXPECT_EQ(wrapper.handle.bufferIndex, handles.front().bufferIndex);
  EXPECT_TRUE(gQueue.backoffWaits.empty());
  EXPECT_EQ(pool->wait(handles.front(), 0), VK_ERROR_DEVICE_LOST);
  // The loss ends every submission, not only the one waited for.
  EXPECT_EQ(retiredThrough(*pool), submissions(*pool));
  EXPECT_EQ(pool->wait(handles.back(), 0), VK_ERROR_DEVICE_LOST);
  pool->discard(wrapper);
}

TEST_F(VulkanImmediateCommandsTest, AnAcquireThatCannotFreeARowBacksOffInsteadOfSpinning) {
  const auto pool = makeFakePool("Acquire back-off test");
  // Every row holds an exported submission, so only a checkpoint can prove one finished.
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    const auto handle = pool->submit(pool->acquire());
    ASSERT_GE(exportFenceFd(*pool, handle), 0);
  }
  // The first checkpoints cannot be submitted for lack of memory.
  constexpr size_t kFailures = 20;
  gQueue.submitResults.assign(kFailures, VK_ERROR_OUT_OF_HOST_MEMORY);
  const auto& wrapper = pool->acquire();
  // Each failed try is followed by a wait on the back-off fence, 1 ms, then twice as long each
  // time up to 16 ms; the checkpoint that could be submitted freed a row.
  EXPECT_EQ(gQueue.backoffWaits, backoffs(kFailures));
  EXPECT_EQ(gQueue.checkpoints, kFailures + 1);
  EXPECT_TRUE(wrapper.isEncoding);
  EXPECT_FALSE(pool->isDeviceLost());
  pool->discard(wrapper);
}

TEST_F(VulkanImmediateCommandsTest, WaitAllKeepsTryingUntilTheWorkIsProvenFinished) {
  const auto pool = makeFakePool("Wait all retry test");
  const auto handle = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, handle), 0);
  constexpr size_t kFailures = 20;
  gQueue.submitResults.assign(kFailures, VK_ERROR_OUT_OF_HOST_MEMORY);
  pool->waitAll();
  // It returns only once a checkpoint proved the submission finished, backing off after each failed
  // try.
  EXPECT_EQ(gQueue.backoffWaits, backoffs(kFailures));
  EXPECT_TRUE(pool->isReady(handle));
  EXPECT_EQ(pool->wait(handle, 0), VK_SUCCESS);
  EXPECT_EQ(gQueue.checkpoints, kFailures + 1);

  // The back-off starts again at 1 ms in every call.
  const auto later = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, later), 0);
  gQueue.submitResults.assign(2, VK_ERROR_OUT_OF_HOST_MEMORY);
  gQueue.backoffWaits.clear();
  pool->waitAll();
  EXPECT_EQ(gQueue.backoffWaits, backoffs(2));
  EXPECT_TRUE(pool->isReady(later));
}

TEST_F(VulkanImmediateCommandsTest, ABackoffThatReportsADeviceLossEndsTheWorkAsLost) {
  // Nothing on a working device signals the back-off fence: an answer that the device is lost ends
  // the work, and so does a success once a loss was already reported, here by the creation of a
  // checkpoint fence.
  for (const VkResult answer : {VK_ERROR_DEVICE_LOST, VK_SUCCESS}) {
    gQueue = {};
    const auto pool = makeFakePool("Back-off device loss test");
    // Every row holds an exported submission, so only a checkpoint can prove one finished, and none
    // can be submitted.
    std::vector<Commands::SubmitHandle> handles;
    for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
      handles.push_back(pool->submit(pool->acquire()));
      ASSERT_GE(exportFenceFd(*pool, handles.back()), 0);
    }
    gQueue.submitResults.assign(100, VK_ERROR_OUT_OF_HOST_MEMORY);
    if (answer == VK_SUCCESS) {
      gQueue.createFenceResult = VK_ERROR_DEVICE_LOST;
    }
    gQueue.backoffResults = {VK_TIMEOUT, VK_TIMEOUT, answer};
    const auto& wrapper = pool->acquire();
    // The third back-off reported the loss: every submission ended, as lost, and its row is reused
    // without another checkpoint.
    EXPECT_EQ(gQueue.backoffWaits, backoffs(3)) << answer;
    EXPECT_EQ(gQueue.checkpoints, answer == VK_SUCCESS ? 2u : 3u) << answer;
    EXPECT_TRUE(pool->isDeviceLost()) << answer;
    for (const auto handle : handles) {
      EXPECT_TRUE(pool->isReady(handle)) << answer;
      EXPECT_EQ(pool->wait(handle, 0), VK_ERROR_DEVICE_LOST) << answer;
    }
    pool->discard(wrapper);
    gQueue.submitResults.clear();
  }
}

TEST_F(VulkanImmediateCommandsTest,
       ABackoffFenceThatAnswersSuccessWithNoLossReportedProvesNothing) {
  const auto pool = makeFakePool("Back-off success without a loss test");
  std::vector<Commands::SubmitHandle> handles;
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    handles.push_back(pool->submit(pool->acquire()));
    ASSERT_GE(exportFenceFd(*pool, handles.back()), 0);
  }
  gQueue.submitResults.assign(3, VK_ERROR_OUT_OF_HOST_MEMORY);
  // The back-off fence answers as signaled at once, and no loss was reported: that proves nothing.
  // The pool waits as long on the host instead, and tries again until a checkpoint is submitted.
  gQueue.backoffResults.assign(3, VK_SUCCESS);
  const auto start = std::chrono::steady_clock::now();
  const auto& wrapper = pool->acquire();
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(1 + 2 + 4));
  EXPECT_EQ(gQueue.backoffWaits, backoffs(3));
  EXPECT_EQ(gQueue.checkpoints, 4u);
  EXPECT_FALSE(pool->isDeviceLost());
  for (const auto handle : handles) {
    EXPECT_EQ(pool->wait(handle, 0), VK_SUCCESS);
  }
  pool->discard(wrapper);
}

TEST_F(VulkanImmediateCommandsTest, WaitAllReturnsOnceABackoffReportsADeviceLoss) {
  const auto pool = makeFakePool("Wait all back-off device loss test");
  const auto handle = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, handle), 0);
  gQueue.submitResults.assign(100, VK_ERROR_OUT_OF_HOST_MEMORY);
  gQueue.backoffResults = {VK_TIMEOUT, VK_ERROR_DEVICE_LOST};
  pool->waitAll();
  EXPECT_EQ(gQueue.backoffWaits, backoffs(2));
  EXPECT_EQ(gQueue.checkpoints, 2u);
  EXPECT_EQ(pool->wait(handle, 0), VK_ERROR_DEVICE_LOST);
  gQueue.submitResults.clear();
}

TEST_F(VulkanImmediateCommandsTest, ABackoffWhoseFenceWaitFailsWaitsOnTheHost) {
  const auto pool = makeFakePool("Back-off failed wait test");
  for (uint32_t i = 0; i != Commands::kMaxCommandBuffers; ++i) {
    ASSERT_GE(exportFenceFd(*pool, pool->submit(pool->acquire())), 0);
  }
  gQueue.submitResults.assign(3, VK_ERROR_OUT_OF_HOST_MEMORY);
  // The back-off fence waits fail at once: an error proves nothing, and the pool waits as long on
  // the host instead of trying again at once.
  gQueue.backoffResults.assign(3, VK_ERROR_OUT_OF_HOST_MEMORY);
  const auto start = std::chrono::steady_clock::now();
  const auto& wrapper = pool->acquire();
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(1 + 2 + 4));
  EXPECT_EQ(gQueue.backoffWaits, backoffs(3));
  EXPECT_EQ(gQueue.checkpoints, 4u);
  EXPECT_FALSE(pool->isDeviceLost());
  pool->discard(wrapper);
}

TEST_F(VulkanImmediateCommandsTest, ACheckpointWhoseFenceCannotBeResetGetsANewFence) {
  const auto pool = makeFakePool("Checkpoint reset failure test");
  const auto first = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, first), 0);
  EXPECT_EQ(pool->wait(first, kTestTimeout), VK_SUCCESS);
  ASSERT_EQ(checkpointFences(*pool), 1u);
  ASSERT_STREQ(checkpointState(*pool, 0), "Signaled");
  const VkFence checkpointFence = gQueue.submitted.back();

  // Reusing the checkpoint needs a reset, which fails: the checkpoint gets a new fence, and the old
  // one is never submitted again.
  const auto second = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, second), 0);
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  EXPECT_EQ(pool->wait(second, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(checkpointFences(*pool), 1u);
  const VkFence newFence = gQueue.submitted.back();
  EXPECT_NE(newFence, checkpointFence);
  EXPECT_EQ(timesSubmitted(checkpointFence), 1u);

  // When no fence can be created either, the exact result is reported and the checkpoint is
  // quarantined; the next checkpoint gives it a new fence.
  const auto third = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, third), 0);
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  gQueue.createFenceResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  EXPECT_EQ(pool->wait(third, kTestTimeout), VK_ERROR_OUT_OF_DEVICE_MEMORY);
  EXPECT_STREQ(checkpointState(*pool, 0), "Quarantined");
  EXPECT_EQ(pool->wait(third, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(checkpointFences(*pool), 1u);
  EXPECT_EQ(timesSubmitted(newFence), 1u);
}

TEST_F(VulkanImmediateCommandsTest, AWaitReportsItsOwnWorkWhenAnotherRowCannotBeReset) {
  for (const VkResult resetFailure : {VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_DEVICE_LOST}) {
    gQueue = {};
    const auto pool = makeFakePool("Wait reset failure test");
    // An exported submission, then one whose fence the wait blocks on.
    const auto exported = pool->submit(pool->acquire());
    ASSERT_GE(exportFenceFd(*pool, exported), 0);
    const auto waited = pool->submit(pool->acquire());
    // Freeing the exported submission's row after the wait fails to reset its fence: the row stays
    // out, and the wait reports the waited work, which finished.
    gQueue.resetResults = {resetFailure};
    EXPECT_EQ(pool->wait(waited, kTestTimeout), VK_SUCCESS) << resetFailure;
    EXPECT_TRUE(quarantined(*pool, exported.bufferIndex)) << resetFailure;
    EXPECT_STREQ(rowState(*pool, exported.bufferIndex), "Submitted") << resetFailure;
    EXPECT_TRUE(pool->isReady(exported)) << resetFailure;
    // A reported device loss is remembered, but does not make later work ready.
    EXPECT_EQ(pool->isDeviceLost(), resetFailure == VK_ERROR_DEVICE_LOST) << resetFailure;
    const auto later = pool->submit(pool->acquire());
    EXPECT_FALSE(pool->isReady(later)) << resetFailure;
    completeSubmissions(gQueue.submitted.size());
  }
}

TEST_F(VulkanImmediateCommandsTest, AFailedResetAssertsAsItAlwaysDid) {
  const DebugAborts aborts;
  const auto pool = makeFakePool("Reset assertion test");
  // A finished row whose fence cannot be reset, then one whose command buffer cannot be reset.
  static_cast<void>(pool->submit(pool->acquire()));
  completeSubmissions(gQueue.submitted.size());
  gQueue.resetResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(aborts.in("recycleCertifiedRow"), kDebugAbortsPerFailure);
  static_cast<void>(pool->submit(pool->acquire()));
  completeSubmissions(gQueue.submitted.size());
  gQueue.resetCommandBufferResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  EXPECT_EQ(certifyAll(*pool, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(aborts.in("recycleCertifiedRow"), 2 * kDebugAbortsPerFailure);
  // A discarded recording whose command buffer cannot be reset.
  gQueue.resetCommandBufferResults = {VK_ERROR_OUT_OF_DEVICE_MEMORY};
  pool->discard(pool->acquire());
  EXPECT_EQ(aborts.in("freeUnsubmittedRow"), kDebugAbortsPerFailure);
  EXPECT_EQ(gDebugAborts.size(), 3 * kDebugAbortsPerFailure);
}

TEST_F(VulkanImmediateCommandsTest, ACheckpointResetThatReportsDeviceLossIsRemembered) {
  const auto pool = makeFakePool("Checkpoint reset device loss test");
  const auto first = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, first), 0);
  EXPECT_EQ(pool->wait(first, kTestTimeout), VK_SUCCESS);
  ASSERT_STREQ(checkpointState(*pool, 0), "Signaled");
  const VkFence checkpointFence = gQueue.submitted.back();
  const auto second = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, second), 0);
  // The reset reports a loss, which is remembered but proves nothing, and no new fence can be
  // created: the work is not proven finished.
  gQueue.resetResults = {VK_ERROR_DEVICE_LOST};
  gQueue.createFenceResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  EXPECT_EQ(pool->wait(second, kTestTimeout), VK_ERROR_UNKNOWN);
  EXPECT_TRUE(pool->isDeviceLost());
  EXPECT_STREQ(checkpointState(*pool, 0), "Quarantined");
  EXPECT_FALSE(pool->isReady(second));
  // Then the checkpoint gets a new fence, which proves the submission finished.
  EXPECT_EQ(pool->wait(second, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(checkpointFences(*pool), 1u);
  EXPECT_EQ(timesSubmitted(checkpointFence), 1u);
}

TEST_F(VulkanImmediateCommandsTest, NoCheckpointIsSubmittedOnceDrainingTheOldestUsedTheTime) {
  const auto pool = makeFakePool("Checkpoint budget test");
  // Two pending checkpoints, covering the first and the second exported submission.
  gQueue.blockingWaitsComplete = false;
  for (uint32_t i = 0; i != 2; ++i) {
    const auto exported = pool->submit(pool->acquire());
    ASSERT_GE(exportFenceFd(*pool, exported), 0);
    ASSERT_EQ(pool->wait(exported, 1'000'000), VK_TIMEOUT);
  }
  ASSERT_EQ(gQueue.checkpoints, 2u);
  // A third needs a new checkpoint. Waiting for the oldest pending one uses the whole budget, so
  // no checkpoint is reset or submitted after it.
  const auto third = pool->submit(pool->acquire());
  ASSERT_GE(exportFenceFd(*pool, third), 0);
  gQueue.blockingWaitsComplete = true;
  EXPECT_EQ(pool->wait(third, 1), VK_TIMEOUT);
  EXPECT_EQ(gQueue.checkpoints, 2u);
  EXPECT_EQ(pool->wait(third, kTestTimeout), VK_SUCCESS);
  EXPECT_EQ(gQueue.checkpoints, 3u);
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
       AnExhaustedDescriptorPoolParkedUnderADroppedRecordingIsReusedOnceTheWorkBeforeItFinished) {
  auto& ctx = getVulkanContext();
  const auto dispatch = makeStorageBufferDispatch();
  ASSERT_NE(dispatch.pipeline, nullptr);
  const DescriptorPoolProbe probe(ctx.vf_);
  const FakeContextPool fakePool(*this, ctx.immediate_, "Dropped descriptor pool test");
  // Without the chain, recordings after the dropped one are submitted.
  chained(*fakePool) = false;
  const auto queue = iglDev_->createCommandQueue({}, nullptr);
  ASSERT_NE(queue, nullptr);
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
  // B's first set moves the arena to a second pool and parks the first one under B's handle. B
  // cannot be ended: it is not submitted, and never runs.
  const auto recordingB = queue->createCommandBuffer({}, nullptr);
  ASSERT_NE(recordingB, nullptr);
  recordDispatches(*recordingB, dispatch, 1);
  ASSERT_NE(gProbe.lastStorageBufferPool, firstPool);
  gQueue.endResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  queue->submit(*recordingB);
  ASSERT_TRUE(droppedTheRecording(*fakePool));

  // O uses up the second pool. A may still read the first pool's sets: a third pool is used.
  recordDispatches(*recordingO, dispatch, kSetsPerPool);
  EXPECT_NE(gProbe.lastStorageBufferPool, firstPool);
  // Once A finished, the next switch reuses the first pool, although B's handle never completes.
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
