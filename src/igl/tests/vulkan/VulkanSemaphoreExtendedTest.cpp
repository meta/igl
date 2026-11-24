/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanSemaphore.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class VulkanSemaphoreExtendedTest : public ::testing::Test {
 public:
  VulkanSemaphoreExtendedTest() = default;

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

TEST_F(VulkanSemaphoreExtendedTest, BinaryCreation) {
  auto& ctx = getVulkanContext();

  auto semaphore = std::make_unique<vulkan::VulkanSemaphore>(
      ctx.vf_, ctx.getVkDevice(), false, "testBinarySemaphore");

  ASSERT_NE(semaphore, nullptr);
  EXPECT_NE(semaphore->getVkSemaphore(), VK_NULL_HANDLE);
  EXPECT_FALSE(semaphore->exportable_);
  EXPECT_EQ(semaphore->getFileDescriptor(), -1);
}

TEST_F(VulkanSemaphoreExtendedTest, TimelineCreation) {
  auto& ctx = getVulkanContext();

  auto semaphore = std::make_unique<vulkan::VulkanSemaphore>(
      ctx.vf_, ctx.getVkDevice(), 0, false, "testTimelineSemaphore");

  ASSERT_NE(semaphore, nullptr);
  EXPECT_NE(semaphore->getVkSemaphore(), VK_NULL_HANDLE);
  EXPECT_FALSE(semaphore->exportable_);
  EXPECT_EQ(semaphore->getFileDescriptor(), -1);
}

TEST_F(VulkanSemaphoreExtendedTest, ExportableSemaphore) {
#if IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID
  auto& ctx = getVulkanContext();

  auto semaphore = std::make_unique<vulkan::VulkanSemaphore>(
      ctx.vf_, ctx.getVkDevice(), true, "testExportableSemaphore");

  ASSERT_NE(semaphore, nullptr);
  EXPECT_NE(semaphore->getVkSemaphore(), VK_NULL_HANDLE);
  EXPECT_TRUE(semaphore->exportable_);
#else
  GTEST_SKIP() << "Exportable semaphores may not be supported on this platform.";
#endif
}

TEST_F(VulkanSemaphoreExtendedTest, MoveConstruction) {
  auto& ctx = getVulkanContext();

  vulkan::VulkanSemaphore original(ctx.vf_, ctx.getVkDevice(), false, "testMoveCtor");
  VkSemaphore originalHandle = original.getVkSemaphore();
  ASSERT_NE(originalHandle, VK_NULL_HANDLE);

  vulkan::VulkanSemaphore moved(std::move(original));
  EXPECT_EQ(moved.getVkSemaphore(), originalHandle);
  // After move, original is in a valid but unspecified state - don't access it
}

TEST_F(VulkanSemaphoreExtendedTest, MoveAssignment) {
  auto& ctx = getVulkanContext();

  vulkan::VulkanSemaphore original(ctx.vf_, ctx.getVkDevice(), false, "testMoveAssign1");
  vulkan::VulkanSemaphore target(ctx.vf_, ctx.getVkDevice(), false, "testMoveAssign2");

  VkSemaphore originalHandle = original.getVkSemaphore();
  ASSERT_NE(originalHandle, VK_NULL_HANDLE);

  target = std::move(original);
  EXPECT_EQ(target.getVkSemaphore(), originalHandle);
  // After move, original is in a valid but unspecified state - don't access it
}

TEST_F(VulkanSemaphoreExtendedTest, TimelineWithInitialValue) {
  auto& ctx = getVulkanContext();

  auto semaphore = std::make_unique<vulkan::VulkanSemaphore>(
      ctx.vf_, ctx.getVkDevice(), 42, false, "testTimelineInitial");

  ASSERT_NE(semaphore, nullptr);
  EXPECT_NE(semaphore->getVkSemaphore(), VK_NULL_HANDLE);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
