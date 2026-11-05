/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanImmediateCommands.h>

#include "../util/TestDevice.h"

#include <memory>
#include <set>
#include <vector>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

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

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
