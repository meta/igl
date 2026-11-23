/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <memory>
#include <igl/tests/util/device/TestDevice.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanTexture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX

namespace igl::tests {

namespace {
constexpr uint32_t kWidth = 256;
constexpr uint32_t kHeight = 256;
constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;
} // namespace

class VulkanMemorylessTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
    device_ = igl::tests::util::device::createTestDevice(BackendType::Vulkan);
    ASSERT_TRUE(device_ != nullptr);
    auto& device = static_cast<vulkan::Device&>(*device_);
    context_ = &device.getVulkanContext();
    ASSERT_TRUE(context_ != nullptr);
  }

 protected:
  std::shared_ptr<IDevice> device_;
  vulkan::VulkanContext* context_ = nullptr;
};

TEST_F(VulkanMemorylessTest, MemorylessUsageFlags) {
  Result ret;
  TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, kWidth, kHeight, TextureDesc::TextureUsageBits::Attachment);
  desc.storage = ResourceStorage::Memoryless;
  const auto texture = device_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(texture, nullptr);
  const VkImageUsageFlags usageFlags =
      static_cast<const vulkan::Texture&>(*texture).getVulkanTexture().image.getVkImageUsageFlags();

  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0u);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0u);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_SAMPLED_BIT, 0u);
}

TEST_F(VulkanMemorylessTest, InputAttachmentUsageFlag) {
  Result ret;
  TextureDesc desc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                        kWidth,
                                        kHeight,
                                        TextureDesc::TextureUsageBits::Attachment |
                                            TextureDesc::TextureUsageBits::InputAttachment);
  const auto texture = device_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(texture, nullptr);
  const VkImageUsageFlags usageFlags =
      static_cast<const vulkan::Texture&>(*texture).getVulkanTexture().image.getVkImageUsageFlags();

  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
}

TEST_F(VulkanMemorylessTest, MemorylessInputAttachmentUsageFlags) {
  Result ret;
  TextureDesc desc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                        kWidth,
                                        kHeight,
                                        TextureDesc::TextureUsageBits::Attachment |
                                            TextureDesc::TextureUsageBits::InputAttachment);
  desc.storage = ResourceStorage::Memoryless;
  const auto texture = device_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(texture, nullptr);
  const VkImageUsageFlags usageFlags =
      static_cast<const vulkan::Texture&>(*texture).getVulkanTexture().image.getVkImageUsageFlags();

  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0u);
  EXPECT_EQ(usageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0u);
}

TEST_F(VulkanMemorylessTest, FeatureMatchesLazilyAllocatedSupport) {
  // hasFeature(MemorylessAttachments) should match supportsLazilyAllocatedMemory()
  const bool hasMemorylessFeature = device_->hasFeature(DeviceFeatures::MemorylessAttachments);
  const bool supportsLazilyAllocated = context_->supportsLazilyAllocatedMemory();
  EXPECT_EQ(hasMemorylessFeature, supportsLazilyAllocated);
}

TEST_F(VulkanMemorylessTest, LazilyAllocatedFallbackSucceeds) {
  // Create a VulkanImage directly with VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT.
  // On Lavapipe (which doesn't support lazily-allocated memory), this should fall back to
  // DEVICE_LOCAL.
  vulkan::VulkanImage image(
      *context_,
      VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
      VK_IMAGE_TYPE_2D,
      kFormat,
      1, /* mipLevels */
      1, /* arrayLayers */
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
      0, /* createFlags */
      VK_SAMPLE_COUNT_1_BIT,
      "Lazily Allocated Fallback");

  // Should succeed even on Lavapipe due to fallback to DEVICE_LOCAL
  EXPECT_TRUE(image.valid());
  EXPECT_NE(image.getVkImage(), VK_NULL_HANDLE);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX
