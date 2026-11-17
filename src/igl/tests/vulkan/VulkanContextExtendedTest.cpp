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
#include <set>
#include <string>
#include <type_traits>
#include <igl/CommandBuffer.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanFence.h>
#include <igl/vulkan/VulkanImmediateCommands.h>
#include <igl/vulkan/VulkanRenderPassBuilder.h>
#include <igl/vulkan/VulkanSemaphore.h>
#include <igl/vulkan/VulkanStagingDevice.h>
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

namespace {

/// Wraps the creation and destruction calls of a `VulkanImmediateCommands` through a copy of the
/// function table: it counts the attempts, fails the chosen one, and tracks every live object so
/// that a test can assert that nothing is left behind or destroyed twice.
struct PoolFaults {
  VulkanFunctionTable real = {};
  int failPoolAt = -1;
  int failFenceAt = -1;
  int failSemaphoreAt = -1;
  int failAllocateAt = -1;
  VkResult injected = VK_ERROR_OUT_OF_HOST_MEMORY;
  int pools = 0;
  int fences = 0;
  int semaphores = 0;
  int allocations = 0;
  std::set<uint64_t> livePools;
  std::set<uint64_t> liveFences;
  std::set<uint64_t> liveSemaphores;
  int badDestroys = 0;
};
thread_local PoolFaults gPool;

template<typename T>
uint64_t keyOf(T handle) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
}

template<typename T>
T garbage() {
  if constexpr (std::is_pointer_v<T>) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<T>(static_cast<uintptr_t>(0xdeadbee0));
  } else {
    return static_cast<T>(0xdeadbeefdeadbee0ULL);
  }
}

/// Whether `handle` is a live object of the test; a null handle is a no-op, and any other handle is
/// a bad destroy (it is not forwarded to the driver).
template<typename T>
bool trackDestroy(std::set<uint64_t>& live, T handle) {
  if (handle == VK_NULL_HANDLE) {
    return false;
  }
  if (live.erase(keyOf(handle)) != 1) {
    ++gPool.badDestroys;
    return false;
  }
  return true;
}

VKAPI_ATTR VkResult VKAPI_CALL poolCreateCommandPool(VkDevice d,
                                                     const VkCommandPoolCreateInfo* i,
                                                     const VkAllocationCallbacks* a,
                                                     VkCommandPool* o) {
  if (gPool.pools++ == gPool.failPoolAt) {
    *o = garbage<VkCommandPool>();
    return gPool.injected;
  }
  const VkResult result = gPool.real.vkCreateCommandPool(d, i, a, o);
  gPool.livePools.insert(keyOf(*o));
  return result;
}

VKAPI_ATTR void VKAPI_CALL poolDestroyCommandPool(VkDevice d,
                                                  VkCommandPool p,
                                                  const VkAllocationCallbacks* a) {
  if (trackDestroy(gPool.livePools, p)) {
    gPool.real.vkDestroyCommandPool(d, p, a);
  }
}

VKAPI_ATTR VkResult VKAPI_CALL poolCreateFence(VkDevice d,
                                               const VkFenceCreateInfo* i,
                                               const VkAllocationCallbacks* a,
                                               VkFence* o) {
  if (gPool.fences++ == gPool.failFenceAt) {
    *o = garbage<VkFence>();
    return gPool.injected;
  }
  const VkResult result = gPool.real.vkCreateFence(d, i, a, o);
  gPool.liveFences.insert(keyOf(*o));
  return result;
}

VKAPI_ATTR void VKAPI_CALL poolDestroyFence(VkDevice d, VkFence f, const VkAllocationCallbacks* a) {
  if (trackDestroy(gPool.liveFences, f)) {
    gPool.real.vkDestroyFence(d, f, a);
  }
}

VKAPI_ATTR VkResult VKAPI_CALL poolCreateSemaphore(VkDevice d,
                                                   const VkSemaphoreCreateInfo* i,
                                                   const VkAllocationCallbacks* a,
                                                   VkSemaphore* o) {
  if (gPool.semaphores++ == gPool.failSemaphoreAt) {
    *o = garbage<VkSemaphore>();
    return gPool.injected;
  }
  const VkResult result = gPool.real.vkCreateSemaphore(d, i, a, o);
  gPool.liveSemaphores.insert(keyOf(*o));
  return result;
}

VKAPI_ATTR void VKAPI_CALL poolDestroySemaphore(VkDevice d,
                                                VkSemaphore s,
                                                const VkAllocationCallbacks* a) {
  if (trackDestroy(gPool.liveSemaphores, s)) {
    gPool.real.vkDestroySemaphore(d, s, a);
  }
}

VKAPI_ATTR VkResult VKAPI_CALL poolAllocateCommandBuffers(VkDevice d,
                                                          const VkCommandBufferAllocateInfo* i,
                                                          VkCommandBuffer* o) {
  if (gPool.allocations++ == gPool.failAllocateAt) {
    return gPool.injected;
  }
  return gPool.real.vkAllocateCommandBuffers(d, i, o);
}

} // namespace

TEST_F(VulkanContextExtendedTest, ImmediateCommandsConstructionReportsEachFailureAndLeaksNothing) {
  auto& ctx = getVulkanContext();
  enum class Step { Pool, Fence, Semaphore, Allocate };
  struct Case {
    Step step;
    int index;
  };
  const std::vector<Case> cases = {{.step = Step::Pool, .index = 0},
                                   {.step = Step::Fence, .index = 0},
                                   {.step = Step::Fence, .index = 1},
                                   {.step = Step::Fence, .index = 31},
                                   {.step = Step::Semaphore, .index = 0},
                                   {.step = Step::Semaphore, .index = 17},
                                   {.step = Step::Semaphore, .index = 31},
                                   {.step = Step::Allocate, .index = 0},
                                   {.step = Step::Allocate, .index = 5},
                                   {.step = Step::Allocate, .index = 31}};
  constexpr int kRows = static_cast<int>(vulkan::VulkanImmediateCommands::kMaxCommandBuffers);
  for (const auto& c : cases) {
    gPool = {};
    gPool.real = ctx.vf_;
    VulkanFunctionTable vf = ctx.vf_;
    vf.vkCreateCommandPool = poolCreateCommandPool;
    vf.vkDestroyCommandPool = poolDestroyCommandPool;
    vf.vkCreateFence = poolCreateFence;
    vf.vkDestroyFence = poolDestroyFence;
    vf.vkCreateSemaphore = poolCreateSemaphore;
    vf.vkDestroySemaphore = poolDestroySemaphore;
    vf.vkAllocateCommandBuffers = poolAllocateCommandBuffers;
    switch (c.step) {
    case Step::Pool:
      gPool.failPoolAt = c.index;
      break;
    case Step::Fence:
      gPool.failFenceAt = c.index;
      break;
    case Step::Semaphore:
      gPool.failSemaphoreAt = c.index;
      break;
    case Step::Allocate:
      gPool.failAllocateAt = c.index;
      break;
    }
    const std::string label =
        "step " + std::to_string(static_cast<int>(c.step)) + " index " + std::to_string(c.index);
    {
      vulkan::VulkanImmediateCommands commands(vf,
                                               ctx.getVkDevice(),
                                               ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                               false,
                                               false,
                                               false,
                                               "Failing pool test");
      EXPECT_FALSE(commands.valid()) << label;
      EXPECT_EQ(commands.initResult(), VK_ERROR_OUT_OF_HOST_MEMORY) << label;
    }
    // The construction stopped at the failure: nothing was attempted after it.
    const int row = c.index;
    switch (c.step) {
    case Step::Pool:
      EXPECT_EQ(gPool.fences, 0) << label;
      break;
    case Step::Fence:
      // A row's fence and semaphore are created together, so the row's semaphore exists too.
      EXPECT_EQ(gPool.fences, row + 1) << label;
      EXPECT_EQ(gPool.semaphores, row + 1) << label;
      EXPECT_EQ(gPool.allocations, row) << label;
      break;
    case Step::Semaphore:
      EXPECT_EQ(gPool.fences, row + 1) << label;
      EXPECT_EQ(gPool.semaphores, row + 1) << label;
      EXPECT_EQ(gPool.allocations, row) << label;
      break;
    case Step::Allocate:
      EXPECT_EQ(gPool.fences, row + 1) << label;
      EXPECT_EQ(gPool.semaphores, row + 1) << label;
      EXPECT_EQ(gPool.allocations, row + 1) << label;
      break;
    }
    // Everything that was created is destroyed, once.
    EXPECT_TRUE(gPool.livePools.empty()) << label;
    EXPECT_TRUE(gPool.liveFences.empty()) << label;
    EXPECT_TRUE(gPool.liveSemaphores.empty()) << label;
    EXPECT_EQ(gPool.badDestroys, 0) << label;
  }

  // A pool without a fault is valid, creates every row and destroys all of it.
  gPool = {};
  gPool.real = ctx.vf_;
  VulkanFunctionTable vf = ctx.vf_;
  vf.vkCreateCommandPool = poolCreateCommandPool;
  vf.vkDestroyCommandPool = poolDestroyCommandPool;
  vf.vkCreateFence = poolCreateFence;
  vf.vkDestroyFence = poolDestroyFence;
  vf.vkCreateSemaphore = poolCreateSemaphore;
  vf.vkDestroySemaphore = poolDestroySemaphore;
  vf.vkAllocateCommandBuffers = poolAllocateCommandBuffers;
  {
    vulkan::VulkanImmediateCommands commands(vf,
                                             ctx.getVkDevice(),
                                             ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                             false,
                                             false,
                                             false,
                                             "Pool without a fault");
    EXPECT_TRUE(commands.valid());
    EXPECT_EQ(commands.initResult(), VK_SUCCESS);
    EXPECT_EQ(gPool.livePools.size(), size_t{1});
    EXPECT_EQ(gPool.liveFences.size(), static_cast<size_t>(kRows));
    EXPECT_EQ(gPool.liveSemaphores.size(), static_cast<size_t>(kRows));
  }
  EXPECT_TRUE(gPool.livePools.empty());
  EXPECT_TRUE(gPool.liveFences.empty());
  EXPECT_TRUE(gPool.liveSemaphores.empty());
  EXPECT_EQ(gPool.badDestroys, 0);
}

TEST_F(VulkanContextExtendedTest, FenceAndSemaphoreRecordWhyTheyWereNotCreated) {
  auto& ctx = getVulkanContext();
  gPool = {};
  gPool.real = ctx.vf_;
  gPool.failFenceAt = 0;
  gPool.failSemaphoreAt = 0;
  VulkanFunctionTable vf = ctx.vf_;
  vf.vkCreateFence = poolCreateFence;
  vf.vkDestroyFence = poolDestroyFence;
  vf.vkCreateSemaphore = poolCreateSemaphore;
  vf.vkDestroySemaphore = poolDestroySemaphore;
  {
    const vulkan::VulkanFence failedFence(vf, ctx.getVkDevice(), VkFenceCreateFlags{});
    EXPECT_EQ(failedFence.initResult(), VK_ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_EQ(failedFence.vkFence_, VK_NULL_HANDLE);
    vulkan::VulkanFence fence(vf, ctx.getVkDevice(), VkFenceCreateFlags{});
    EXPECT_EQ(fence.initResult(), VK_SUCCESS);
    EXPECT_NE(fence.vkFence_, VK_NULL_HANDLE);

    const vulkan::VulkanSemaphore failedBinary(vf, ctx.getVkDevice());
    EXPECT_EQ(failedBinary.initResult(), VK_ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_EQ(failedBinary.getVkSemaphore(), VK_NULL_HANDLE);
    const vulkan::VulkanSemaphore binary(vf, ctx.getVkDevice());
    EXPECT_EQ(binary.initResult(), VK_SUCCESS);
    EXPECT_NE(binary.getVkSemaphore(), VK_NULL_HANDLE);

    // Moving carries the result with the handle.
    vulkan::VulkanFence moved(std::move(fence));
    EXPECT_EQ(moved.initResult(), VK_SUCCESS);
    EXPECT_NE(moved.vkFence_, VK_NULL_HANDLE);
  }
  gPool.semaphores = 0;
  gPool.failSemaphoreAt = 1;
  if (ctx.features().has_VK_KHR_timeline_semaphore) {
    // The timeline constructor, the second one of the semaphore class: the second creation fails.
    const vulkan::VulkanSemaphore ok(vf, ctx.getVkDevice());
    const vulkan::VulkanSemaphore failedTimeline(vf, ctx.getVkDevice(), uint64_t{0});
    EXPECT_EQ(failedTimeline.initResult(), VK_ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_EQ(failedTimeline.getVkSemaphore(), VK_NULL_HANDLE);
  }
  EXPECT_TRUE(gPool.liveFences.empty());
  EXPECT_TRUE(gPool.liveSemaphores.empty());
  EXPECT_EQ(gPool.badDestroys, 0);
}

TEST_F(VulkanContextExtendedTest, ContextInitFailsWhenACommandBufferPoolCannotBeCreated) {
  using Faulting = util::device::vulkan::FaultingDevice;
  using Counted = Faulting::Counted;
  struct Case {
    Faulting::Call call;
    uint32_t skip;
    const char* message;
  };
  // The first creation of each kind belongs to the context's own pool; `AllocateCommandBuffers`
  // at the last row ends its construction too. The second command pool is the staging device's.
  const std::vector<Case> cases = {
      {.call = Faulting::Call::CreateCommandPool, .skip = 0, .message = "command buffer pool"},
      {.call = Faulting::Call::CreateFence, .skip = 0, .message = "command buffer pool"},
      {.call = Faulting::Call::CreateSemaphore, .skip = 0, .message = "command buffer pool"},
      {.call = Faulting::Call::AllocateCommandBuffers, .skip = 0, .message = "command buffer pool"},
      {.call = Faulting::Call::AllocateCommandBuffers,
       .skip = 31,
       .message = "command buffer pool"},
      {.call = Faulting::Call::CreateCommandPool,
       .skip = 1,
       .message = "staging command buffer pool"},
  };
  for (const auto& c : cases) {
    const Faulting fd(util::device::vulkan::getContextConfig(), [&c](Faulting& self) {
      self.arm(c.call, VK_ERROR_OUT_OF_HOST_MEMORY, c.skip, /*count=*/1);
    });
    const std::string label =
        "call " + std::to_string(static_cast<int>(c.call)) + " skip " + std::to_string(c.skip);
    EXPECT_EQ(fd.device(), nullptr) << label;
    EXPECT_EQ(fd.failed(c.call), 1u) << label;
    EXPECT_EQ(fd.creationResult().code, Result::Code::RuntimeError) << label;
    EXPECT_NE(fd.creationResult().message.find(c.message), std::string::npos)
        << label << ": " << fd.creationResult().message.c_str();
    // The message names the failing result (VK_ERROR_OUT_OF_HOST_MEMORY is -1).
    EXPECT_NE(fd.creationResult().message.find("VkResult -1"), std::string::npos)
        << label << ": " << fd.creationResult().message.c_str();
    // A handle a failed creation left behind is never destroyed (the fixture counts such a destroy
    // and does not forward it to the driver).
    EXPECT_EQ(fd.counted(Counted::BadDestroy), 0u) << label;
    // Nothing the failed context created is left: every created object was destroyed.
    EXPECT_EQ(fd.reached(Faulting::Call::CreateFence) - fd.failed(Faulting::Call::CreateFence),
              fd.counted(Counted::DestroyFence))
        << label;
    EXPECT_EQ(
        fd.reached(Faulting::Call::CreateSemaphore) - fd.failed(Faulting::Call::CreateSemaphore),
        fd.counted(Counted::DestroySemaphore))
        << label;
    EXPECT_EQ(fd.reached(Faulting::Call::CreateCommandPool) -
                  fd.failed(Faulting::Call::CreateCommandPool),
              fd.counted(Counted::DestroyCommandPool))
        << label;
  }
}

TEST_F(VulkanContextExtendedTest, ContextWithoutFaultsHasValidCommandBufferPools) {
  using Faulting = util::device::vulkan::FaultingDevice;
  const Faulting fd(util::device::vulkan::getContextConfig());
  ASSERT_NE(fd.device(), nullptr) << fd.creationResult().message.c_str();
  EXPECT_TRUE(fd.creationResult().isOk());
  auto& ctx = fd.context();
  ASSERT_NE(ctx.immediate_, nullptr);
  EXPECT_TRUE(ctx.immediate_->valid());
  EXPECT_EQ(ctx.immediate_->initResult(), VK_SUCCESS);
  ASSERT_NE(ctx.stagingDevice_, nullptr);
  EXPECT_TRUE(ctx.stagingDevice_->immediate->valid());
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
