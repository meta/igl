/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <cstdint>
#include <future>
#include <igl/CommandBuffer.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanRenderPassBuilder.h>
#include <igl/vulkan/VulkanVma.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class VulkanContextExtendedTest : public ::testing::Test {
 public:
  VulkanContextExtendedTest() = default;

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

  uint32_t vmaAllocationCount() {
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(static_cast<VmaAllocator>(getVulkanContext().getVmaAllocator()), &stats);
    return stats.total.statistics.allocationCount;
  }

  static void submitAndWait(ICommandQueue& cmdQueue) {
    Result ret;
    auto cmdBuf = cmdQueue.createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk());
    cmdQueue.submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
  }

  // Staging buffers created with the device are freed within the first few submissions.
  static void settle(ICommandQueue& cmdQueue) {
    for (int i = 0; i < 4; ++i) {
      submitAndWait(cmdQueue);
    }
  }
};

TEST_F(VulkanContextExtendedTest, WaitIdle) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  cmdQueue->submit(*cmdBuf);

  auto result = ctx.waitIdle();
  EXPECT_TRUE(result.isOk()) << result.message.c_str();
}

TEST_F(VulkanContextExtendedTest, DeferredTaskExecution) {
  auto& ctx = getVulkanContext();

  bool taskExecuted = false;
  std::packaged_task<void()> task([&taskExecuted]() { taskExecuted = true; });

  ctx.deferredTask(std::move(task));

  Result ret;
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());
  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  cmdQueue->submit(*cmdBuf);

  ctx.waitDeferredTasks();
}

TEST_F(VulkanContextExtendedTest, ReleasedTextureIsFreedAfterTwoSubmitsAndWaits) {
  ASSERT_FALSE(getVulkanContext().config_.enableDescriptorIndexing);

  Result ret;
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());
  settle(*cmdQueue);
  const uint32_t allocationsBefore = vmaAllocationCount();

  {
    const TextureDesc desc = TextureDesc::new2D(
        TextureFormat::RGBA_UNorm8, 64, 64, TextureDesc::TextureUsageBits::Sampled);
    auto texture = iglDev_->createTexture(desc, &ret);
    ASSERT_TRUE(ret.isOk());
    ASSERT_NE(texture, nullptr);
    EXPECT_EQ(vmaAllocationCount(), allocationsBefore + 1);
  }

  submitAndWait(*cmdQueue);
  submitAndWait(*cmdQueue);
  EXPECT_EQ(vmaAllocationCount(), allocationsBefore);
}

TEST_F(VulkanContextExtendedTest, ReleasedTextureOutlivesAnOpenCommandBuffer) {
  ASSERT_FALSE(getVulkanContext().config_.enableDescriptorIndexing);

  Result ret;
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());
  settle(*cmdQueue);
  const uint32_t allocationsBefore = vmaAllocationCount();

  // Recording may have used the texture, so it has to stay alive until this buffer is submitted.
  auto openCmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  {
    const TextureDesc desc = TextureDesc::new2D(
        TextureFormat::RGBA_UNorm8, 64, 64, TextureDesc::TextureUsageBits::Sampled);
    auto texture = iglDev_->createTexture(desc, &ret);
    ASSERT_TRUE(ret.isOk());
    ASSERT_NE(texture, nullptr);
  }

  submitAndWait(*cmdQueue);
  submitAndWait(*cmdQueue);
  EXPECT_EQ(vmaAllocationCount(), allocationsBefore + 1);

  cmdQueue->submit(*openCmdBuf);
  openCmdBuf->waitUntilCompleted();
  submitAndWait(*cmdQueue);
  EXPECT_EQ(vmaAllocationCount(), allocationsBefore);
}

TEST_F(VulkanContextExtendedTest, TextureViewKeepsItsBaseTextureAlive) {
  ASSERT_FALSE(getVulkanContext().config_.enableDescriptorIndexing);

  Result ret;
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());
  settle(*cmdQueue);
  const uint32_t allocationsBefore = vmaAllocationCount();

  const TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 64, 64, TextureDesc::TextureUsageBits::Sampled);
  auto baseTexture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(baseTexture, nullptr);
  auto view = iglDev_->createTextureView(baseTexture, {.debugName = "view"}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(vmaAllocationCount(), allocationsBefore + 1);

  baseTexture.reset();
  submitAndWait(*cmdQueue);
  submitAndWait(*cmdQueue);
  EXPECT_EQ(vmaAllocationCount(), allocationsBefore + 1);

  view.reset();
  submitAndWait(*cmdQueue);
  submitAndWait(*cmdQueue);
  EXPECT_EQ(vmaAllocationCount(), allocationsBefore);
}

TEST_F(VulkanContextExtendedTest, GetPipelineCacheData) {
  auto& ctx = getVulkanContext();

  auto cacheData = ctx.getPipelineCacheData();
  // Cache data may be empty if no pipelines have been created, but the call must not crash
  SUCCEED();
}

TEST_F(VulkanContextExtendedTest, GetVkDevice) {
  auto& ctx = getVulkanContext();
  EXPECT_TRUE(ctx.getVkDevice() != nullptr);
}

TEST_F(VulkanContextExtendedTest, GetVkPhysicalDevice) {
  auto& ctx = getVulkanContext();
  EXPECT_TRUE(ctx.getVkPhysicalDevice() != nullptr);
}

TEST_F(VulkanContextExtendedTest, GetVkInstance) {
  auto& ctx = getVulkanContext();
  EXPECT_TRUE(ctx.getVkInstance() != nullptr);
}

TEST_F(VulkanContextExtendedTest, PhysicalDeviceProperties) {
  auto& ctx = getVulkanContext();
  const auto& props = ctx.getVkPhysicalDeviceProperties();

  EXPECT_NE(props.apiVersion, 0u);
  EXPECT_GT(props.limits.maxImageDimension2D, 0u);
}

TEST_F(VulkanContextExtendedTest, CurrentSyncIndex) {
  auto& ctx = getVulkanContext();
  EXPECT_LT(ctx.currentSyncIndex(), ctx.config_.maxResourceCount);
}

TEST_F(VulkanContextExtendedTest, FeaturesAccessor) {
  auto& ctx = getVulkanContext();
  const auto& features = ctx.features();

  EXPECT_EQ(features.vkPhysicalDeviceFeatures2.sType, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
}

TEST_F(VulkanContextExtendedTest, HasSwapchainWithoutInit) {
  auto& ctx = getVulkanContext();
  EXPECT_FALSE(ctx.hasSwapchain());
}

TEST_F(VulkanContextExtendedTest, GetFrameNumber) {
  auto& ctx = getVulkanContext();
  EXPECT_EQ(ctx.getFrameNumber(), 0u);
}

TEST_F(VulkanContextExtendedTest, FindRenderPassCached) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const auto rp1 = ctx.findRenderPass(builder);
  EXPECT_NE(rp1.pass, VK_NULL_HANDLE);

  const auto rp2 = ctx.findRenderPass(builder);
  EXPECT_EQ(rp1.pass, rp2.pass);
}

TEST_F(VulkanContextExtendedTest, ConfigDefaults) {
  const auto& ctx = getVulkanContext();
  EXPECT_GT(ctx.config_.maxResourceCount, 0u);
}

// Invariant: syncAcquireNext() advances the ring-buffer index by exactly one step modulo
// maxResourceCount, so the index never leaves [0, maxResourceCount) and returns to its starting
// value after exactly maxResourceCount acquisitions. A wrong modulus or a missing wrap would
// desynchronize per-frame resource reuse.
TEST_F(VulkanContextExtendedTest, SyncAcquireNextWrapsAroundModuloMaxResourceCount) {
  auto& ctx = getVulkanContext();

  const uint32_t maxResourceCount = ctx.config_.maxResourceCount;
  ASSERT_GT(maxResourceCount, 0u);

  const uint32_t startIndex = ctx.currentSyncIndex();
  ASSERT_LT(startIndex, maxResourceCount);

  for (uint32_t step = 1; step <= maxResourceCount; ++step) {
    ctx.syncAcquireNext();
    EXPECT_EQ(ctx.currentSyncIndex(), (startIndex + step) % maxResourceCount);
  }

  // A full lap of maxResourceCount acquisitions returns the index to where it started.
  EXPECT_EQ(ctx.currentSyncIndex(), startIndex);
}

// Invariant: two getOrCreateVkDescriptorSetLayout() calls with byte-identical bindings resolve to
// the SAME cached VkDescriptorSetLayout handle (cache hit via DescriptorSetLayoutCacheKey equality)
// instead of allocating a redundant second layout.
TEST_F(VulkanContextExtendedTest, GetOrCreateVkDescriptorSetLayoutDedupesIdenticalBindings) {
  auto& ctx = getVulkanContext();

  const VkDescriptorSetLayoutBinding binding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
  };

  const VkDescriptorSetLayout first = ctx.getOrCreateVkDescriptorSetLayout(
      /*flags=*/0, /*numBindings=*/1, &binding, /*bindingFlags=*/nullptr);
  const VkDescriptorSetLayout second = ctx.getOrCreateVkDescriptorSetLayout(
      /*flags=*/0, /*numBindings=*/1, &binding, /*bindingFlags=*/nullptr);

  EXPECT_NE(first, VK_NULL_HANDLE);
  EXPECT_EQ(first, second);
}

// Invariant: getOrCreateVkDescriptorSetLayout() must treat layouts that differ in ANY single
// binding field (binding index, descriptor type, descriptor count, or stage flags) as distinct,
// returning a different handle for each. If the cache-key equality dropped a field from its
// comparison, a mismatching layout would wrongly alias the baseline handle.
TEST_F(VulkanContextExtendedTest, GetOrCreateVkDescriptorSetLayoutDistinguishesDifferingBindings) {
  auto& ctx = getVulkanContext();

  const VkDescriptorSetLayoutBinding base{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
  };
  const VkDescriptorSetLayout baseHandle = ctx.getOrCreateVkDescriptorSetLayout(
      /*flags=*/0, /*numBindings=*/1, &base, /*bindingFlags=*/nullptr);
  ASSERT_NE(baseHandle, VK_NULL_HANDLE);

  // Each variant differs from `base` in exactly one field.
  VkDescriptorSetLayoutBinding differentBindingIndex = base;
  differentBindingIndex.binding = 1;
  VkDescriptorSetLayoutBinding differentType = base;
  differentType.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  VkDescriptorSetLayoutBinding differentCount = base;
  differentCount.descriptorCount = 2;
  VkDescriptorSetLayoutBinding differentStage = base;
  differentStage.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  EXPECT_NE(ctx.getOrCreateVkDescriptorSetLayout(
                /*flags=*/0, /*numBindings=*/1, &differentBindingIndex, /*bindingFlags=*/nullptr),
            baseHandle);
  EXPECT_NE(ctx.getOrCreateVkDescriptorSetLayout(
                /*flags=*/0, /*numBindings=*/1, &differentType, /*bindingFlags=*/nullptr),
            baseHandle);
  EXPECT_NE(ctx.getOrCreateVkDescriptorSetLayout(
                /*flags=*/0, /*numBindings=*/1, &differentCount, /*bindingFlags=*/nullptr),
            baseHandle);
  EXPECT_NE(ctx.getOrCreateVkDescriptorSetLayout(
                /*flags=*/0, /*numBindings=*/1, &differentStage, /*bindingFlags=*/nullptr),
            baseHandle);
}

// Invariant: getClosestDepthStencilFormat() walks the per-format compatibility list from closest to
// least-close and returns the FIRST device-supported entry. VK_FORMAT_D16_UNORM and
// VK_FORMAT_D32_SFLOAT are guaranteed by the Vulkan spec to support depth/stencil attachment, so
// they are always the highest-priority supported candidate for their requested formats.
TEST_F(VulkanContextExtendedTest,
       GetClosestDepthStencilFormatPrefersHighestPrioritySupportedFormat) {
  auto& ctx = getVulkanContext();

  EXPECT_EQ(ctx.getClosestDepthStencilFormat(TextureFormat::Z_UNorm16), VK_FORMAT_D16_UNORM);
  EXPECT_EQ(ctx.getClosestDepthStencilFormat(TextureFormat::Z_UNorm32), VK_FORMAT_D32_SFLOAT);
}

// checkAndUpdateDescriptorSets() runs once per pass, so a wait that re-arms the configured
// timeout after every failure turns a bounded wait into an unbounded stall. The first attempt
// must use the configured value and later ones must poll. No device needed -- the policy is
// pure, and the timeout branch it feeds cannot otherwise be reached deterministically (it
// needs a fence that never signals).
TEST(VulkanContextDescriptorSetWaitTimeoutTest, FirstAttemptUsesConfiguredTimeout) {
  constexpr uint64_t kConfigured = 30ULL * 1000000000ULL;

  EXPECT_EQ(igl::vulkan::VulkanContext::descriptorSetWaitTimeout(kConfigured,
                                                                 /*previouslyDegraded=*/false),
            kConfigured);
}

TEST(VulkanContextDescriptorSetWaitTimeoutTest, PollsAfterAPreviousTimeout) {
  constexpr uint64_t kConfigured = 30ULL * 1000000000ULL;

  EXPECT_EQ(igl::vulkan::VulkanContext::descriptorSetWaitTimeout(kConfigured,
                                                                 /*previouslyDegraded=*/true),
            0ULL);
}

// The IGL default is UINT64_MAX. Escalation must still apply, otherwise a stuck fence blocks
// forever on the very configuration that most needs bounding.
TEST(VulkanContextDescriptorSetWaitTimeoutTest, EscalatesEvenFromTheInfiniteDefault) {
  EXPECT_EQ(igl::vulkan::VulkanContext::descriptorSetWaitTimeout(UINT64_MAX,
                                                                 /*previouslyDegraded=*/false),
            UINT64_MAX);
  EXPECT_EQ(igl::vulkan::VulkanContext::descriptorSetWaitTimeout(UINT64_MAX,
                                                                 /*previouslyDegraded=*/true),
            0ULL);
}

TEST_F(VulkanContextExtendedTest, DummyTextureUploadFailureFailsContextInit) {
  if (!IGL_VULKAN_USE_VMA) {
    GTEST_SKIP() << "The allocation tests run on the VMA path";
  }
  using Faulting = util::device::vulkan::FaultingDevice;
  // The two dummy buffers are allowed; the first staging buffer, which the dummy texture's upload
  // needs, cannot be allocated. The fault is armed before create().
  const Faulting fd(util::device::vulkan::getContextConfig(), [](Faulting& self) {
    self.arm(Faulting::Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/2, UINT32_MAX);
  });

  EXPECT_EQ(fd.device(), nullptr);
  EXPECT_FALSE(fd.creationResult().isOk());
  EXPECT_NE(fd.creationResult().message.find("dummy texture"), std::string::npos)
      << fd.creationResult().message.c_str();
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
