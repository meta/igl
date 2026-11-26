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

namespace igl::tests {

namespace {

struct FakeVulkan {
  VkResult waitForFencesResult = VK_SUCCESS;
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

  EXPECT_EQ(platformDevice().getFenceFdFromSubmitHandle(handle), -1);
}

} // namespace

} // namespace igl::tests
