/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <array>
#include <igl/Device.h>
#include <igl/Texture.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanTexture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class TextureVulkanExtendedTest : public ::testing::Test {
 public:
  TextureVulkanExtendedTest() = default;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
};

TEST_F(TextureVulkanExtendedTest, Create3D) {
  if (!iglDev_->hasFeature(DeviceFeatures::Texture3D)) {
    GTEST_SKIP() << "3D textures not supported.";
  }

  Result ret;
  const TextureDesc desc = TextureDesc::new3D(
      TextureFormat::RGBA_UNorm8, 4, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(texture->getType(), TextureType::ThreeD);
}

TEST_F(TextureVulkanExtendedTest, CreateCube) {
  Result ret;
  const TextureDesc desc = TextureDesc::newCube(
      TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(texture->getType(), TextureType::Cube);
}

TEST_F(TextureVulkanExtendedTest, CreateArray) {
  if (!iglDev_->hasFeature(DeviceFeatures::Texture2DArray)) {
    GTEST_SKIP() << "Texture arrays not supported.";
  }

  Result ret;
  const TextureDesc desc = TextureDesc::new2DArray(
      TextureFormat::RGBA_UNorm8, 4, 4, 3, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(texture->getType(), TextureType::TwoDArray);
  EXPECT_EQ(texture->getNumLayers(), 3u);
}

TEST_F(TextureVulkanExtendedTest, CreateMSAA) {
  Result ret;
  TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Attachment);
  desc.numSamples = 4;
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(texture->getSamples(), 4u);
}

TEST_F(TextureVulkanExtendedTest, DepthStencilFormats) {
  Result ret;

  const TextureDesc desc = TextureDesc::new2D(TextureFormat::Z_UNorm24,
                                              4,
                                              4,
                                              TextureDesc::TextureUsageBits::Attachment |
                                                  TextureDesc::TextureUsageBits::Sampled);
  auto depthTexture = iglDev_->createTexture(desc, &ret);
  if (ret.isOk()) {
    ASSERT_NE(depthTexture, nullptr);
  }

  const TextureDesc descDS = TextureDesc::new2D(TextureFormat::S8_UInt_Z24_UNorm,
                                                4,
                                                4,
                                                TextureDesc::TextureUsageBits::Attachment |
                                                    TextureDesc::TextureUsageBits::Sampled);
  auto dsTexture = iglDev_->createTexture(descDS, &ret);
  if (ret.isOk()) {
    ASSERT_NE(dsTexture, nullptr);
  }
}

TEST_F(TextureVulkanExtendedTest, TextureId) {
  Result ret;
  const TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 2, 2, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  auto* vulkanTexture = static_cast<igl::vulkan::Texture*>(texture.get());
  ASSERT_NE(vulkanTexture, nullptr);
  EXPECT_NE(vulkanTexture->getVulkanTexture().textureId_, 0u);
}

TEST_F(TextureVulkanExtendedTest, Upload2D) {
  Result ret;
  const TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 2, 2, TextureDesc::TextureUsageBits::Sampled);
  const auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  const std::array<uint32_t, 4> pixels = {0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF};
  ret = texture->upload(texture->getFullRange(0), pixels.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
}

TEST_F(TextureVulkanExtendedTest, StorageTexture) {
  Result ret;
  const TextureDesc desc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                              4,
                                              4,
                                              TextureDesc::TextureUsageBits::Storage |
                                                  TextureDesc::TextureUsageBits::Sampled);
  const auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(texture->getType(), TextureType::TwoD);
}

TEST_F(TextureVulkanExtendedTest, TextureDimensions) {
  Result ret;
  const TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 16, 32, TextureDesc::TextureUsageBits::Sampled);
  const auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  const auto dims = texture->getDimensions();
  EXPECT_EQ(dims.width, 16u);
  EXPECT_EQ(dims.height, 32u);
  EXPECT_EQ(dims.depth, 1u);
}

TEST_F(TextureVulkanExtendedTest, TextureFormatQuery) {
  Result ret;
  const TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  const auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  EXPECT_EQ(texture->getFormat(), TextureFormat::RGBA_UNorm8);
  EXPECT_EQ(texture->getType(), TextureType::TwoD);
  EXPECT_EQ(texture->getNumLayers(), 1u);
  EXPECT_EQ(texture->getSamples(), 1u);
  EXPECT_EQ(texture->getNumMipLevels(), 1u);
}

TEST_F(TextureVulkanExtendedTest, TextureUploadPropagatesStagingFailure) {
  if (!IGL_VULKAN_USE_VMA) {
    GTEST_SKIP() << "The allocation tests run on the VMA path";
  }
  util::device::vulkan::FaultingDevice fd(util::device::vulkan::getContextConfig());
  ASSERT_NE(fd.device(), nullptr) << fd.creationResult().message.c_str();

  Result ret;
  const auto texture = fd.device()->createTexture(
      TextureDesc::new2D(
          TextureFormat::RGBA_UNorm8, 64, 64, TextureDesc::TextureUsageBits::Sampled),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  fd.drainStaging();
  fd.arm(util::device::vulkan::FaultingDevice::Call::CreateBuffer,
         VK_ERROR_OUT_OF_DEVICE_MEMORY,
         /*skip=*/0,
         /*count=*/UINT32_MAX);
  const std::vector<uint32_t> pixels(64 * 64, 0xFFFFFFFFu);
  const Result upload = texture->upload(texture->getFullRange(0), pixels.data());
  EXPECT_FALSE(upload.isOk());
  EXPECT_EQ(upload.code, Result::Code::RuntimeError);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
