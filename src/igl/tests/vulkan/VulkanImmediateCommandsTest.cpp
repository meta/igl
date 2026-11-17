/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanImmediateCommands.h>

#include "../util/TestDevice.h"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>
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
  return gTrace.waitResult;
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

  // The discarded recording and the one that took its row have the same handle, so the stale
  // handle names the open recording: not ready, not waitable, and its fence is the row's.
  EXPECT_EQ(second.handle, stale);
  EXPECT_FALSE(commands.isReady(stale));
  EXPECT_EQ(commands.wait(stale, 0), VK_ERROR_UNKNOWN);
  EXPECT_EQ(commands.getVkFenceFromSubmitHandle(stale), second.fence.vkFence_);

  const auto handle = commands.submit(second);
  EXPECT_EQ(handle, stale);
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
  // Only polls: no reset of the command buffer or of the fence.
  const std::vector<std::string> expected = {
      "vkWaitForFences all=1 timeout=0 fence0",
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

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
