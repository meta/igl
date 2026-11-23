/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanFence.h>

#include "../util/TestDevice.h"

#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

// Bounded timeout for GPU-signaled fence waits so a missing signal (driver bug, resource
// exhaustion) fails the test cleanly instead of hanging the runner on the default UINT64_MAX wait.
constexpr uint64_t kFenceWaitTimeoutNs = 5'000'000'000; // 5 seconds

class VulkanFenceTest : public ::testing::Test {
 public:
  VulkanFenceTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;

  vulkan::VulkanContext& getVulkanContext() {
    auto& device = static_cast<vulkan::Device&>(*iglDev_);
    return device.getVulkanContext();
  }
};

TEST_F(VulkanFenceTest, CreateSignaledFence) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence fence(
      ctx.vf_, ctx.getVkDevice(), VK_FENCE_CREATE_SIGNALED_BIT, false, "testSignaledFence");

  EXPECT_NE(fence.vkFence_, VK_NULL_HANDLE);
  EXPECT_FALSE(fence.exportable());

  EXPECT_TRUE(fence.wait(0));
}

TEST_F(VulkanFenceTest, CreateUnsignaledFence) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence fence(ctx.vf_, ctx.getVkDevice(), 0, false, "testUnsignaledFence");

  EXPECT_NE(fence.vkFence_, VK_NULL_HANDLE);
  EXPECT_FALSE(fence.exportable());
}

TEST_F(VulkanFenceTest, ResetFence) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence fence(
      ctx.vf_, ctx.getVkDevice(), VK_FENCE_CREATE_SIGNALED_BIT, false, "testResetFence");

  ASSERT_NE(fence.vkFence_, VK_NULL_HANDLE);
  EXPECT_TRUE(fence.wait(0));

  EXPECT_TRUE(fence.reset());

  // After reset the fence is unsignaled: an immediate wait times out.
  EXPECT_FALSE(fence.wait(0));
}

TEST_F(VulkanFenceTest, ExportableFence) {
#if IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence fence(
      ctx.vf_, ctx.getVkDevice(), VK_FENCE_CREATE_SIGNALED_BIT, true, "testExportableFence");

  EXPECT_NE(fence.vkFence_, VK_NULL_HANDLE);
  EXPECT_TRUE(fence.exportable());
#else
  GTEST_SKIP() << "Exportable fences may not be supported on this platform.";
#endif
}

TEST_F(VulkanFenceTest, MoveConstruction) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence original(
      ctx.vf_, ctx.getVkDevice(), VK_FENCE_CREATE_SIGNALED_BIT, false, "testMoveCtor");

  VkFence originalHandle = original.vkFence_;
  ASSERT_NE(originalHandle, VK_NULL_HANDLE);

  vulkan::VulkanFence moved(std::move(original));

  EXPECT_EQ(moved.vkFence_, originalHandle);
  // The signaled state transfers with the move, not just the handle: an immediate wait succeeds.
  EXPECT_TRUE(moved.wait(0));
  // After move, original is in a valid but unspecified state - don't access it
}

TEST_F(VulkanFenceTest, SignalWithNullQueue) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence fence(ctx.vf_, ctx.getVkDevice(), 0, false, "testSignalNull");

  ASSERT_NE(fence.vkFence_, VK_NULL_HANDLE);
  EXPECT_FALSE(fence.signal(VK_NULL_HANDLE));
}

TEST_F(VulkanFenceTest, SignalAndWait) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence fence(ctx.vf_, ctx.getVkDevice(), 0, false, "testSignalAndWait");

  ASSERT_NE(fence.vkFence_, VK_NULL_HANDLE);

  const VkQueue graphicsQueue = ctx.deviceQueues_.graphicsQueue;
  // Use a boolean expression rather than ASSERT_NE(graphicsQueue, VK_NULL_HANDLE). VkQueue is a
  // dispatchable handle (a pointer on every ABI), but on 32-bit ABIs (VK_USE_64_BIT_PTR_DEFINES==0)
  // VK_NULL_HANDLE expands to the integer literal 0ULL, so gtest's two-type comparison macro fails
  // to compile (pointer vs integer). The boolean form lets the standard null-pointer conversion
  // apply instead.
  ASSERT_TRUE(graphicsQueue != VK_NULL_HANDLE);

  ASSERT_TRUE(fence.signal(graphicsQueue));
  EXPECT_TRUE(fence.wait(kFenceWaitTimeoutNs));
}

TEST_F(VulkanFenceTest, WaitUnsignaledTimesOut) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence fence(ctx.vf_, ctx.getVkDevice(), 0, false, "testWaitTimeout");

  ASSERT_NE(fence.vkFence_, VK_NULL_HANDLE);
  EXPECT_FALSE(fence.wait(0));
}

TEST_F(VulkanFenceTest, MoveAssignment) {
  auto& ctx = getVulkanContext();
  vulkan::VulkanFence original(
      ctx.vf_, ctx.getVkDevice(), VK_FENCE_CREATE_SIGNALED_BIT, false, "testMoveAssign1");
  vulkan::VulkanFence target(ctx.vf_, ctx.getVkDevice(), 0, false, "testMoveAssign2");

  VkFence originalHandle = original.vkFence_;
  ASSERT_NE(originalHandle, VK_NULL_HANDLE);

  target = std::move(original);

  EXPECT_EQ(target.vkFence_, originalHandle);
  // After move, original is in a valid but unspecified state - don't access it
}

// Verifies the common reuse cycle: signal on the GPU, wait, reset, then signal again. This is the
// pattern callers rely on when recycling a single fence across frames.
TEST_F(VulkanFenceTest, SignalResetSignalCycle) {
  auto& ctx = getVulkanContext();
  const VkQueue queue = ctx.deviceQueues_.graphicsQueue;
  // Use a boolean expression rather than ASSERT_NE(queue, VK_NULL_HANDLE). VkQueue is a
  // dispatchable handle (a pointer on every ABI), but on 32-bit ABIs (VK_USE_64_BIT_PTR_DEFINES==0)
  // VK_NULL_HANDLE expands to the integer literal 0ULL, so gtest's two-type comparison macro fails
  // to compile (pointer vs integer). The boolean form lets the standard null-pointer conversion
  // apply instead.
  ASSERT_TRUE(queue != VK_NULL_HANDLE);

  vulkan::VulkanFence fence(ctx.vf_, ctx.getVkDevice(), 0, false, "testSignalResetCycle");
  ASSERT_NE(fence.vkFence_, VK_NULL_HANDLE);

  // First cycle.
  ASSERT_TRUE(fence.signal(queue));
  ASSERT_TRUE(fence.wait(kFenceWaitTimeoutNs));

  // Reset back to unsignaled and confirm.
  ASSERT_TRUE(fence.reset());
  EXPECT_FALSE(fence.wait(0));

  // Second cycle reuses the same fence object.
  ASSERT_TRUE(fence.signal(queue));
  EXPECT_TRUE(fence.wait(kFenceWaitTimeoutNs));
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
