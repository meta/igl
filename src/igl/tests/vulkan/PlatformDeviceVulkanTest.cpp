/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <cstdint>
#include <memory>
#include <igl/CommandBuffer.h>
#include <igl/CommandQueue.h>
#include <igl/Device.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImmediateCommands.h>

namespace igl::tests {

namespace {

/// What the replacement functions below return, and what vkGetFenceFdKHR() was asked to export.
struct FakeVulkan {
  VkResult waitForFencesResult = VK_SUCCESS;
  VkResult getFenceFdResult = VK_SUCCESS;
  int fenceFd{-1};
  uint32_t numGetFenceFdCalls{};
  VkFence exportedFence = VK_NULL_HANDLE;
  VkExternalFenceHandleTypeFlagBits exportedHandleType{};
};

FakeVulkan& fakeVulkan() {
  static FakeVulkan fake;
  return fake;
}

VKAPI_ATTR VkResult VKAPI_CALL fakeWaitForFences(VkDevice /*device*/,
                                                 uint32_t /*fenceCount*/,
                                                 const VkFence* /*pFences*/,
                                                 VkBool32 /*waitAll*/,
                                                 uint64_t /*timeout*/) {
  return fakeVulkan().waitForFencesResult;
}

VKAPI_ATTR VkResult VKAPI_CALL fakeGetFenceFd(VkDevice /*device*/,
                                              const VkFenceGetFdInfoKHR* pGetFdInfo,
                                              int* pFd) {
  FakeVulkan& fake = fakeVulkan();
  ++fake.numGetFenceFdCalls;
  fake.exportedFence = pGetFdInfo->fence;
  fake.exportedHandleType = pGetFdInfo->handleType;
  *pFd = fake.fenceFd;
  return fake.getFenceFdResult;
}

/// Lets a test replace functions of the function table of a context, which VulkanContext exposes
/// as a const reference to a table that it owns. Puts the table back when it goes out of scope.
class ScopedFunctionTable final {
 public:
  explicit ScopedFunctionTable(vulkan::VulkanContext& ctx) :
    table_(const_cast<VulkanFunctionTable&>(ctx.vf_)), saved_(table_) {
    fakeVulkan() = {};
  }
  ~ScopedFunctionTable() {
    table_ = saved_;
  }
  ScopedFunctionTable(const ScopedFunctionTable&) = delete;
  ScopedFunctionTable& operator=(const ScopedFunctionTable&) = delete;

  VulkanFunctionTable* operator->() {
    return &table_;
  }

 private:
  VulkanFunctionTable& table_;
  const VulkanFunctionTable saved_;
};

class PlatformDeviceVulkanTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);

    device_ = util::createTestDevice();
    ASSERT_NE(device_, nullptr);
    ASSERT_EQ(device_->getBackendType(), BackendType::Vulkan);

    Result result;
    queue_ = device_->createCommandQueue({}, &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(queue_, nullptr);
  }

 protected:
  vulkan::VulkanContext& context() {
    return static_cast<vulkan::Device&>(*device_).getVulkanContext();
  }

  vulkan::PlatformDevice& platformDevice() {
    return *device_->getPlatformDevice<vulkan::PlatformDevice>();
  }

  /// Submits an empty command buffer, or returns 0 after reporting a failure.
  SubmitHandle submit() {
    Result result;
    const std::shared_ptr<ICommandBuffer> commandBuffer = queue_->createCommandBuffer({}, &result);
    if (!result.isOk() || commandBuffer == nullptr) {
      ADD_FAILURE() << "Cannot create a command buffer: " << result.message;
      return 0;
    }
    return queue_->submit(*commandBuffer);
  }

  int cachedFenceFd(SubmitHandle handle) {
    return context().immediate_->cachedFDFromSubmitHandle(
        vulkan::VulkanImmediateCommands::SubmitHandle(handle));
  }

  std::shared_ptr<IDevice> device_;
  std::shared_ptr<ICommandQueue> queue_;
};

TEST_F(PlatformDeviceVulkanTest, WaitOnSubmitHandleSucceedsOnceTheSubmissionCompletes) {
  const SubmitHandle handle = submit();
  ASSERT_NE(handle, 0u);

  Result result(Result::Code::RuntimeError);
  EXPECT_TRUE(platformDevice().waitOnSubmitHandle(handle, UINT64_MAX, &result));
  EXPECT_TRUE(result.isOk()) << result.message;
}

TEST_F(PlatformDeviceVulkanTest, WaitOnSubmitHandleFailsOnTimeout) {
  const SubmitHandle handle = submit();
  ASSERT_NE(handle, 0u);
  {
    ScopedFunctionTable table(context());
    table->vkWaitForFences = fakeWaitForFences;
    fakeVulkan().waitForFencesResult = VK_TIMEOUT;

    Result result;
    EXPECT_FALSE(platformDevice().waitOnSubmitHandle(handle, 0, &result));
    EXPECT_FALSE(result.isOk());
    EXPECT_EQ(result.message, "VK_TIMEOUT");
  }
  EXPECT_TRUE(platformDevice().waitOnSubmitHandle(handle));
}

TEST_F(PlatformDeviceVulkanTest, WaitOnSubmitHandleFailsWhenTheDeviceIsLost) {
  const SubmitHandle handle = submit();
  ASSERT_NE(handle, 0u);

  ScopedFunctionTable table(context());
  table->vkWaitForFences = fakeWaitForFences;
  fakeVulkan().waitForFencesResult = VK_ERROR_DEVICE_LOST;

  Result result;
  EXPECT_FALSE(platformDevice().waitOnSubmitHandle(handle, UINT64_MAX, &result));
  EXPECT_FALSE(result.isOk());
  EXPECT_EQ(result.message, "VK_ERROR_DEVICE_LOST");
  EXPECT_TRUE(context().isDeviceLost());
}

TEST_F(PlatformDeviceVulkanTest, WaitOnSubmitHandleFailsForANullHandle) {
  Result result;
  EXPECT_FALSE(platformDevice().waitOnSubmitHandle(0, 0, &result));
  EXPECT_EQ(result.code, Result::Code::ArgumentInvalid);
}

TEST_F(PlatformDeviceVulkanTest, FenceFdExportIsUnsupportedWithoutTheExportFunction) {
  const SubmitHandle handle = submit();
  ASSERT_NE(handle, 0u);

  ScopedFunctionTable table(context());
  table->vkGetFenceFdKHR = nullptr;

  Result result;
  EXPECT_EQ(platformDevice().getFenceFdFromSubmitHandle(handle, &result), -1);
  EXPECT_EQ(result.code, Result::Code::Unsupported);
}

TEST_F(PlatformDeviceVulkanTest, FenceFdExportReportsADriverErrorWithoutAnFd) {
  const SubmitHandle handle = submit();
  ASSERT_NE(handle, 0u);

  ScopedFunctionTable table(context());
  table->vkGetFenceFdKHR = fakeGetFenceFd;
  fakeVulkan().getFenceFdResult = VK_ERROR_TOO_MANY_OBJECTS;
  // A failed call leaves its output undefined: it must not be returned as a descriptor.
  fakeVulkan().fenceFd = 77;

  Result result;
  EXPECT_EQ(platformDevice().getFenceFdFromSubmitHandle(handle, &result), -1);
  EXPECT_FALSE(result.isOk());
  EXPECT_NE(result.code, Result::Code::Unsupported);
  EXPECT_EQ(result.message, "VK_ERROR_TOO_MANY_OBJECTS");
  EXPECT_EQ(fakeVulkan().numGetFenceFdCalls, 1u);
  EXPECT_EQ(cachedFenceFd(handle), -1);
}

TEST_F(PlatformDeviceVulkanTest, FenceFdExportOfASignaledFenceSucceedsWithoutAnFd) {
  const SubmitHandle handle = submit();
  ASSERT_NE(handle, 0u);

  ScopedFunctionTable table(context());
  table->vkGetFenceFdKHR = fakeGetFenceFd;
  fakeVulkan().fenceFd = -1;

  Result result(Result::Code::RuntimeError);
  EXPECT_EQ(platformDevice().getFenceFdFromSubmitHandle(handle, &result), -1);
  EXPECT_TRUE(result.isOk()) << result.message;
  EXPECT_EQ(fakeVulkan().numGetFenceFdCalls, 1u);
}

TEST_F(PlatformDeviceVulkanTest, FenceFdExportReturnsTheSyncFdOfTheSubmission) {
  const SubmitHandle handle = submit();
  ASSERT_NE(handle, 0u);
  const VkFence fence = platformDevice().getVkFenceFromSubmitHandle(handle);
  ASSERT_NE(fence, VK_NULL_HANDLE);

  ScopedFunctionTable table(context());
  table->vkGetFenceFdKHR = fakeGetFenceFd;
  // The descriptor is only compared, never used.
  fakeVulkan().fenceFd = 1234;

  Result result(Result::Code::RuntimeError);
  EXPECT_EQ(platformDevice().getFenceFdFromSubmitHandle(handle, &result), 1234);
  EXPECT_TRUE(result.isOk()) << result.message;
  EXPECT_EQ(fakeVulkan().exportedFence, fence);
  EXPECT_EQ(fakeVulkan().exportedHandleType, VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT);
  EXPECT_EQ(cachedFenceFd(handle), 1234);
}

TEST_F(PlatformDeviceVulkanTest, FenceFdExportFailsForANullHandle) {
  ScopedFunctionTable table(context());
  table->vkGetFenceFdKHR = fakeGetFenceFd;

  Result result;
  EXPECT_EQ(platformDevice().getFenceFdFromSubmitHandle(0, &result), -1);
  EXPECT_EQ(result.code, Result::Code::ArgumentInvalid);
  EXPECT_EQ(fakeVulkan().numGetFenceFdCalls, 0u);
}

TEST_F(PlatformDeviceVulkanTest, FenceFdExportFailsForAHandleWhoseSubmissionWasRecycled) {
  const SubmitHandle handle = submit();
  ASSERT_NE(handle, 0u);
  // The same command buffer with a later submit id (the upper 32 bits): its row no longer holds the
  // submission of `handle`.
  const SubmitHandle recycled = handle + (uint64_t{1} << 32);
  ASSERT_EQ(platformDevice().getVkFenceFromSubmitHandle(recycled), VK_NULL_HANDLE);

  ScopedFunctionTable table(context());
  table->vkGetFenceFdKHR = fakeGetFenceFd;

  Result result;
  EXPECT_EQ(platformDevice().getFenceFdFromSubmitHandle(recycled, &result), -1);
  EXPECT_EQ(result.code, Result::Code::ArgumentInvalid);
  EXPECT_EQ(fakeVulkan().numGetFenceFdCalls, 0u);
}

} // namespace

} // namespace igl::tests
