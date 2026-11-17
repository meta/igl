/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanStagingDevice.h>

#include "../util/TestDevice.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>
#include <igl/Buffer.h>
#include <igl/CommandBuffer.h>
#include <igl/Framebuffer.h>
#include <igl/Texture.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanHelpers.h>
#include <igl/vulkan/VulkanTexture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class VulkanStagingDeviceTest : public ::testing::Test {
 public:
  VulkanStagingDeviceTest() = default;

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
};

TEST_F(VulkanStagingDeviceTest, BufferSubDataSmallUpload) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);

  Result ret;
  const BufferDesc bufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = 256,
      .storage = ResourceStorage::Private,
  };
  auto buffer = iglDev_->createBuffer(bufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  std::vector<uint8_t> srcData(256);
  for (size_t i = 0; i < srcData.size(); ++i) {
    srcData[i] = static_cast<uint8_t>(i & 0xFF);
  }

  ret = buffer->upload(srcData.data(), BufferRange(256, 0));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const auto* downloadedData = static_cast<uint8_t*>(buffer->map(BufferRange(256, 0), &ret));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  for (size_t i = 0; i < 256; ++i) {
    EXPECT_EQ(downloadedData[i], srcData[i]);
  }

  buffer->unmap();
}

TEST_F(VulkanStagingDeviceTest, BufferSubDataLargerThanStaging) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);

  const VkDeviceSize maxStagingSize = ctx.stagingDevice_->getMaxStagingBufferSize();

  size_t maxBufferLength = 0;
  iglDev_->getFeatureLimits(DeviceFeatureLimits::MaxStorageBufferBytes, maxBufferLength);

  const VkDeviceSize desiredSize = std::min<VkDeviceSize>(maxStagingSize * 2, maxBufferLength);

  if (desiredSize < maxStagingSize) {
    GTEST_SKIP() << "Max buffer size is smaller than staging buffer, cannot test chunked upload.";
  }

  Result ret;
  const BufferDesc bufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = static_cast<size_t>(desiredSize),
      .storage = ResourceStorage::Private,
  };
  auto buffer = iglDev_->createBuffer(bufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  std::vector<uint16_t> srcData(desiredSize / 2);
  for (size_t i = 0; i < srcData.size(); ++i) {
    srcData[i] = static_cast<uint16_t>(i & 0xFFFF);
  }

  ret = buffer->upload(srcData.data(), BufferRange(desiredSize, 0));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const auto* downloadedData =
      static_cast<uint16_t*>(buffer->map(BufferRange(desiredSize, 0), &ret));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  for (size_t i = 0; i < srcData.size(); ++i) {
    ASSERT_EQ(downloadedData[i], srcData[i]) << "Mismatch at index " << i;
  }

  buffer->unmap();
}

TEST_F(VulkanStagingDeviceTest, ImageDataUpload) {
  Result ret;

  const TextureDesc texDesc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  const size_t dataSize = 4 * 4 * 4;
  std::vector<uint8_t> pixelData(dataSize);
  for (size_t i = 0; i < dataSize; ++i) {
    pixelData[i] = static_cast<uint8_t>(i % 256);
  }

  ret = texture->upload(texture->getFullRange(0), pixelData.data());
  EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
}

TEST_F(VulkanStagingDeviceTest, Private3DUploadRoundTripAfter2DUpload) {
  ASSERT_TRUE(iglDev_->hasFeature(DeviceFeatures::Texture3D)) << "3D textures are unavailable";
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);
  constexpr uint32_t kWidth = 3;
  constexpr uint32_t kHeight = 2;
  constexpr uint32_t kDepth = 3;
  constexpr size_t kBytesPerSlice = kWidth * kHeight * 4;
  constexpr size_t kDataSize = kBytesPerSlice * kDepth;
  VkImageFormatProperties formatProperties{};
  ASSERT_EQ(ctx.vf_.vkGetPhysicalDeviceImageFormatProperties(ctx.getVkPhysicalDevice(),
                                                             VK_FORMAT_R8G8B8A8_UNORM,
                                                             VK_IMAGE_TYPE_3D,
                                                             VK_IMAGE_TILING_OPTIMAL,
                                                             VK_IMAGE_USAGE_SAMPLED_BIT |
                                                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                             0,
                                                             &formatProperties),
            VK_SUCCESS)
      << "Private RGBA8 3D upload/readback is unsupported";
  ASSERT_GE(formatProperties.maxExtent.width, kWidth);
  ASSERT_GE(formatProperties.maxExtent.height, kHeight);
  ASSERT_GE(formatProperties.maxExtent.depth, kDepth);

  Result result;
  const TextureDesc primeDesc{
      .width = 1,
      .height = 1,
      .usage = TextureDesc::TextureUsageBits::Sampled,
      .type = TextureType::TwoD,
      .format = TextureFormat::RGBA_UNorm8,
      .storage = ResourceStorage::Private,
  };
  const auto prime = iglDev_->createTexture(primeDesc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(prime, nullptr);
  const std::array<uint8_t, 4> primePixels = {0xA1, 0xB2, 0xC3, 0xD4};
  result = prime->upload(prime->getFullRange(), primePixels.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  const VkDeviceSize freeAfterPrime = ctx.stagingDevice_->getFreeStagingBufferSize();
  ASSERT_GT(freeAfterPrime, kDataSize + 256);

  const TextureDesc desc{
      .width = kWidth,
      .height = kHeight,
      .depth = kDepth,
      .usage = TextureDesc::TextureUsageBits::Sampled,
      .type = TextureType::ThreeD,
      .format = TextureFormat::RGBA_UNorm8,
      .storage = ResourceStorage::Private,
  };
  const auto texture = iglDev_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  ASSERT_EQ(texture->getStorage(), ResourceStorage::Private);
  std::array<uint8_t, kDataSize> pixels{};
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<uint8_t>(1 + i % kBytesPerSlice + 64 * (i / kBytesPerSlice));
  }
  result = texture->upload(texture->getFullRange(), pixels.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  EXPECT_LT(ctx.stagingDevice_->getFreeStagingBufferSize(), freeAfterPrime);

  auto& image = static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image;
  ASSERT_EQ(image.type_, VK_IMAGE_TYPE_3D);
  ASSERT_FALSE(image.isMappedPtrAccessible());
  const auto readback = ctx.createBuffer(kDataSize,
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                         &result,
                                         "3D staging round-trip readback");
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(readback, nullptr);
  ASSERT_NE(readback->getMappedPtr(), nullptr);
  const auto& commands = ctx.stagingDevice_->immediate->acquire();
  const VkImageSubresourceRange subresources{
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
  };
  ivkImageMemoryBarrier(&ctx.vf_,
                        commands.cmdBuf,
                        image.getVkImage(),
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT,
                        image.imageLayout_,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        subresources);
  image.imageLayout_ = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  const VkBufferImageCopy copy{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           .mipLevel = 0,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
      .imageOffset = {.x = 0, .y = 0, .z = 0},
      .imageExtent = {.width = kWidth, .height = kHeight, .depth = kDepth},
  };
  ctx.vf_.vkCmdCopyImageToBuffer(commands.cmdBuf,
                                 image.getVkImage(),
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 readback->getVkBuffer(),
                                 1,
                                 &copy);
  const VkBufferMemoryBarrier hostBarrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = readback->getVkBuffer(),
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  };
  ctx.vf_.vkCmdPipelineBarrier(commands.cmdBuf,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_HOST_BIT,
                               0,
                               0,
                               nullptr,
                               1,
                               &hostBarrier,
                               0,
                               nullptr);
  ASSERT_EQ(ctx.stagingDevice_->immediate->wait(ctx.stagingDevice_->immediate->submit(commands),
                                                ctx.config_.fenceTimeoutNanoseconds),
            VK_SUCCESS);
  if (!readback->isCoherentMemory()) {
    readback->invalidateMappedMemory(0, kDataSize);
  }
  for (size_t i = 0; i < pixels.size(); ++i) {
    EXPECT_EQ(readback->getMappedPtr()[i], pixels[i])
        << "byte " << i << ", slice " << i / kBytesPerSlice;
  }
}

TEST_F(VulkanStagingDeviceTest, GetImageData2D) {
  Result ret;

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 2,
                                                 2,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Attachment);
  auto texture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  constexpr uint32_t kColor = 0xAABBCCDD;
  const std::array<uint32_t, 4> srcData = {kColor, kColor, kColor, kColor};

  ret = texture->upload(texture->getFullRange(0), srcData.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  cmdQueue->submit(*cmdBuf);

  std::array<uint32_t, 4> downloadedData = {};
  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = texture}}};
  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  fb->copyBytesColorAttachment(*cmdQueue, 0, downloadedData.data(), texture->getFullRange(0));

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(downloadedData[i], kColor);
  }
}

TEST_F(VulkanStagingDeviceTest, PartialBufferUpdate) {
  Result ret;

  const BufferDesc bufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = 256,
      .storage = ResourceStorage::Private,
  };
  auto buffer = iglDev_->createBuffer(bufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  std::vector<uint8_t> zeros(256, 0);
  ret = buffer->upload(zeros.data(), BufferRange(256, 0));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  std::vector<uint8_t> partialData(64, 0xAB);
  const size_t offset = 64;
  ret = buffer->upload(partialData.data(), BufferRange(64, offset));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const auto* downloadedData = static_cast<uint8_t*>(buffer->map(BufferRange(256, 0), &ret));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  for (size_t i = 0; i < offset; ++i) {
    EXPECT_EQ(downloadedData[i], 0) << "Non-zero before offset at index " << i;
  }
  for (size_t i = offset; i < offset + 64; ++i) {
    EXPECT_EQ(downloadedData[i], 0xAB) << "Mismatch in partial update region at index " << i;
  }

  buffer->unmap();
}

TEST_F(VulkanStagingDeviceTest, PrivateMultiplanarUploadRoundTripAfter2DUpload) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);
  constexpr uint32_t kWidth = 4;
  constexpr uint32_t kHeight = 4;
  constexpr size_t kDataSize = kWidth * kHeight * 3 / 2;
  Result result;
  const TextureDesc primeDesc{
      .width = 1,
      .height = 1,
      .usage = TextureDesc::TextureUsageBits::Sampled,
      .type = TextureType::TwoD,
      .format = TextureFormat::RGBA_UNorm8,
      .storage = ResourceStorage::Private,
  };
  const auto prime = iglDev_->createTexture(primeDesc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(prime, nullptr);
  const std::array<uint8_t, 4> primePixels = {0xA1, 0xB2, 0xC3, 0xD4};
  result = prime->upload(prime->getFullRange(), primePixels.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_GT(ctx.stagingDevice_->getFreeStagingBufferSize(), kDataSize + 256);

  std::array<uint8_t, kDataSize> pixels{};
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<uint8_t>(17 + 7 * i);
  }
  for (const TextureFormat format : {TextureFormat::YUV_NV12, TextureFormat::YUV_420p}) {
    const VkFormat vkFormat = vulkan::textureFormatToVkFormat(format);
    const auto properties = TextureFormatProperties::fromTextureFormat(format);
    VkImageFormatProperties formatProperties{};
    ASSERT_EQ(ctx.vf_.vkGetPhysicalDeviceImageFormatProperties(ctx.getVkPhysicalDevice(),
                                                               vkFormat,
                                                               VK_IMAGE_TYPE_2D,
                                                               VK_IMAGE_TILING_OPTIMAL,
                                                               VK_IMAGE_USAGE_SAMPLED_BIT |
                                                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                               0,
                                                               &formatProperties),
              VK_SUCCESS)
        << "Private multiplanar upload/readback is unsupported for format "
        << static_cast<uint32_t>(format);
    ASSERT_GE(formatProperties.maxExtent.width, kWidth);
    ASSERT_GE(formatProperties.maxExtent.height, kHeight);
    ASSERT_GE(formatProperties.maxExtent.depth, 1u);
    ASSERT_GE(formatProperties.maxMipLevels, 1u);
    ASSERT_GE(formatProperties.maxArrayLayers, 1u);
    ASSERT_NE(formatProperties.sampleCounts & VK_SAMPLE_COUNT_1_BIT, 0u);
    auto image = ctx.createImage(VK_IMAGE_TYPE_2D,
                                 VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                                 vkFormat,
                                 1,
                                 1,
                                 VK_IMAGE_TILING_OPTIMAL,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 0,
                                 VK_SAMPLE_COUNT_1_BIT,
                                 &result,
                                 "Multiplanar staging round-trip image");
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(image.getVkImage(), VK_NULL_HANDLE);
    ASSERT_FALSE(image.isMappedPtrAccessible());
    ASSERT_TRUE(ctx.stagingDevice_->imageData(image,
                                              TextureType::TwoD,
                                              TextureRangeDesc::new2D(0, 0, kWidth, kHeight),
                                              properties,
                                              0,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              pixels.data()));

    const auto readback = ctx.createBuffer(kDataSize,
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                           &result,
                                           "Multiplanar staging round-trip readback");
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(readback, nullptr);
    ASSERT_NE(readback->getMappedPtr(), nullptr);
    const auto& commands = ctx.stagingDevice_->immediate->acquire();
    ivkImageMemoryBarrier(&ctx.vf_,
                          commands.cmdBuf,
                          image.getVkImage(),
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT,
                          image.imageLayout_,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                  .levelCount = 1,
                                                  .layerCount = 1});
    const std::array<VkBufferImageCopy, 3> copies = {{
        {.bufferOffset = 0,
         .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT, .layerCount = 1},
         .imageExtent = {.width = kWidth, .height = kHeight, .depth = 1}},
        {.bufferOffset = kWidth * kHeight,
         .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT, .layerCount = 1},
         .imageExtent = {.width = kWidth / 2, .height = kHeight / 2, .depth = 1}},
        {.bufferOffset = kWidth * kHeight * 5 / 4,
         .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_PLANE_2_BIT, .layerCount = 1},
         .imageExtent = {.width = kWidth / 2, .height = kHeight / 2, .depth = 1}},
    }};
    ctx.vf_.vkCmdCopyImageToBuffer(commands.cmdBuf,
                                   image.getVkImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   readback->getVkBuffer(),
                                   properties.numPlanes,
                                   copies.data());
    const VkBufferMemoryBarrier hostBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = readback->getVkBuffer(),
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    ctx.vf_.vkCmdPipelineBarrier(commands.cmdBuf,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 1,
                                 &hostBarrier,
                                 0,
                                 nullptr);
    ASSERT_EQ(ctx.stagingDevice_->immediate->wait(ctx.stagingDevice_->immediate->submit(commands),
                                                  ctx.config_.fenceTimeoutNanoseconds),
              VK_SUCCESS);
    if (!readback->isCoherentMemory()) {
      readback->invalidateMappedMemory(0, kDataSize);
    }
    for (size_t i = 0; i < pixels.size(); ++i) {
      EXPECT_EQ(readback->getMappedPtr()[i], pixels[i])
          << "format " << static_cast<uint32_t>(format) << ", byte " << i;
    }
  }
}

TEST_F(VulkanStagingDeviceTest, ImageCopyAlignmentIncludesColorTexelBlockSize) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);
  constexpr VkDeviceSize kBaseAlignment = 16;
  const VkDeviceSize deviceAlignment =
      ctx.getVkPhysicalDeviceProperties().limits.optimalBufferCopyOffsetAlignment;
  const VkDeviceSize baseAlignment = std::max(kBaseAlignment, deviceAlignment);
  const auto rgbaProperties =
      TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
  EXPECT_EQ(ctx.stagingDevice_->getImageCopyAlignment(rgbaProperties), baseAlignment);
  const auto rgbProperties = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_F16);
  ASSERT_EQ(rgbProperties.bytesPerBlock, 6u);
  EXPECT_EQ(ctx.stagingDevice_->getImageCopyAlignment(rgbProperties), baseAlignment * 3);
}

TEST_F(VulkanStagingDeviceTest, ImageCopyAlignmentUsesFourByteDepthStencilUnits) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);
  constexpr VkDeviceSize kBaseAlignment = 16;
  const VkDeviceSize deviceAlignment =
      ctx.getVkPhysicalDeviceProperties().limits.optimalBufferCopyOffsetAlignment;
  const VkDeviceSize expectedAlignment = std::max(kBaseAlignment, deviceAlignment);
  const auto depthProperties = TextureFormatProperties::fromTextureFormat(TextureFormat::Z_UNorm24);
  ASSERT_EQ(depthProperties.bytesPerBlock, 3u);
  EXPECT_EQ(ctx.stagingDevice_->getImageCopyAlignment(depthProperties), expectedAlignment);
  const auto stencilProperties = TextureFormatProperties::fromTextureFormat(TextureFormat::S_UInt8);
  EXPECT_EQ(ctx.stagingDevice_->getImageCopyAlignment(stencilProperties), expectedAlignment);
}

TEST_F(VulkanStagingDeviceTest, ImageCopyAlignmentIsNonzeroForMultiplanarFormats) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);
  constexpr VkDeviceSize kBaseAlignment = 16;
  const VkDeviceSize deviceAlignment =
      ctx.getVkPhysicalDeviceProperties().limits.optimalBufferCopyOffsetAlignment;
  const VkDeviceSize expectedAlignment = std::max(kBaseAlignment, deviceAlignment);
  for (const TextureFormat format : {TextureFormat::YUV_NV12, TextureFormat::YUV_420p}) {
    const auto properties = TextureFormatProperties::fromTextureFormat(format);
    ASSERT_GT(properties.numPlanes, 1u);
    ASSERT_EQ(properties.bytesPerBlock, 0u);
    EXPECT_EQ(ctx.stagingDevice_->getImageCopyAlignment(properties), expectedAlignment);
  }
}

TEST_F(VulkanStagingDeviceTest, MaxStagingBufferSizePositive) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);

  EXPECT_GT(ctx.stagingDevice_->getMaxStagingBufferSize(), 0u);
}

TEST_F(VulkanStagingDeviceTest, FreeStagingBufferSizeInvariant) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);

  const VkDeviceSize freeSize = ctx.stagingDevice_->getFreeStagingBufferSize();
  const VkDeviceSize maxSize = ctx.stagingDevice_->getMaxStagingBufferSize();
  EXPECT_LE(freeSize, maxSize);
}

TEST_F(VulkanStagingDeviceTest, MergeRegionsRecoversFreeSpace) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);

  // Use a payload > 64KB to bypass the `vkCmdUpdateBuffer()` fast path in
  // `bufferSubData()` and force the upload through the staging device, so
  // `mergeRegionsAndFreeBuffers()` has actual regions to process.
  constexpr size_t kPayloadSize = 128 * 1024;

  Result ret;
  const BufferDesc bufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = kPayloadSize,
      .storage = ResourceStorage::Private,
  };
  auto buffer = iglDev_->createBuffer(bufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  std::vector<uint8_t> data(kPayloadSize, 0xAB);
  ret = buffer->upload(data.data(), BufferRange(kPayloadSize, 0));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Drain GPU work, then merge: must run cleanly and leave the device in a
  // self-consistent state where free space does not exceed the configured
  // max staging pool size.
  ctx.waitDeferredTasks();
  ctx.stagingDevice_->mergeRegionsAndFreeBuffers();

  EXPECT_LE(ctx.stagingDevice_->getFreeStagingBufferSize(),
            ctx.stagingDevice_->getMaxStagingBufferSize());

  // Device must remain usable for further uploads after merge.
  std::vector<uint8_t> data2(kPayloadSize, 0xCD);
  ret = buffer->upload(data2.data(), BufferRange(kPayloadSize, 0));
  EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
}

TEST_F(VulkanStagingDeviceTest, MultipleSequentialUploads) {
  Result ret;

  const BufferDesc bufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = 512,
      .storage = ResourceStorage::Private,
  };
  auto buffer = iglDev_->createBuffer(bufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  for (uint8_t round = 0; round < 3; ++round) {
    std::vector<uint8_t> srcData(512, round * 50);
    ret = buffer->upload(srcData.data(), BufferRange(512, 0));
    ASSERT_TRUE(ret.isOk()) << "Upload failed on round " << static_cast<int>(round);

    const auto* downloaded = static_cast<uint8_t*>(buffer->map(BufferRange(512, 0), &ret));
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    for (size_t i = 0; i < 512; ++i) {
      EXPECT_EQ(downloaded[i], round * 50)
          << "Mismatch on round " << static_cast<int>(round) << " at index " << i;
    }
    buffer->unmap();
  }
}

TEST_F(VulkanStagingDeviceTest, ImageDataUploadWithMipLevel) {
  Result ret;

  const uint32_t width = 8;
  const uint32_t height = 8;
  const uint32_t numMipLevels = 4;

  TextureDesc texDesc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, width, height, TextureDesc::TextureUsageBits::Sampled);
  texDesc.numMipLevels = numMipLevels;
  auto texture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  for (uint32_t mip = 0; mip < numMipLevels; ++mip) {
    const uint32_t mipWidth = std::max(1u, width >> mip);
    const uint32_t mipHeight = std::max(1u, height >> mip);
    const size_t dataSize = mipWidth * mipHeight * 4;
    const std::vector<uint8_t> pixelData(dataSize, static_cast<uint8_t>(mip * 60));

    ret = texture->upload(texture->getFullRange(mip), pixelData.data());
    EXPECT_TRUE(ret.isOk()) << "Upload failed for mip level " << mip << ": " << ret.message.c_str();
  }
}

TEST_F(VulkanStagingDeviceTest, BufferSubDataZeroOffset) {
  Result ret;

  const BufferDesc bufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = 128,
      .storage = ResourceStorage::Private,
  };
  auto buffer = iglDev_->createBuffer(bufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const std::vector<uint8_t> srcData(128, 0xFF);
  ret = buffer->upload(srcData.data(), BufferRange(128, 0));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const auto* downloaded = static_cast<uint8_t*>(buffer->map(BufferRange(128, 0), &ret));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  for (size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(downloaded[i], 0xFF);
  }
  buffer->unmap();
}

TEST_F(VulkanStagingDeviceTest, ABlockReusedAfterItsSubmissionRetiredCountsAsFreeOnce) {
  auto& ctx = getVulkanContext();
  ASSERT_NE(ctx.stagingDevice_, nullptr);
  auto& staging = *ctx.stagingDevice_;
  constexpr VkDeviceSize kMiB = 1024 * 1024;
  constexpr VkDeviceSize kHalf = kMiB / 2;
  // Device-local memory that is not mapped, so that copies go through the staging device.
  Result ret;
  const auto target =
      ctx.createBuffer(kMiB,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       &ret,
                       "Buffer: reused staging block target");
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(target, nullptr);
  ASSERT_FALSE(target->isMapped());
  // Waits for every staging copy and releases the staging buffers they no longer use.
  const auto drainStaging = [&staging] {
    staging.immediate->waitAll();
    staging.mergeRegionsAndFreeBuffers();
    staging.mergeRegionsAndFreeBuffers();
    EXPECT_EQ(staging.getFreeStagingBufferSize(), 0u);
  };
  drainStaging();

  // An upload takes a whole new staging buffer and returns its block with its submission. A 512 KiB
  // readback finds nothing free: it allocates a second staging buffer, which goes to the front of
  // the list, and takes half of it. A 1 MiB readback then copies two 512 KiB chunks: the first
  // takes the rest of the second buffer, the second splits the upload's block, whose submission has
  // finished, and that block counts as free once. When both readbacks are done, both staging
  // buffers are free, and the merges release them.
  const std::vector<uint8_t> data(kMiB, 0x5a);
  std::vector<uint8_t> readBack(kMiB);
  staging.bufferSubData(*target, 0, kMiB, data.data());
  EXPECT_EQ(staging.getFreeStagingBufferSize(), 0u);
  staging.getBufferSubData(*target, 0, kHalf, readBack.data());
  EXPECT_EQ(staging.getFreeStagingBufferSize(), kMiB);
  staging.getBufferSubData(*target, 0, kMiB, readBack.data());
  EXPECT_EQ(staging.getFreeStagingBufferSize(), 2 * kMiB);
  EXPECT_EQ(readBack, data);
  drainStaging();
}

// ---------------------------------------------------------------------------------------------
// Staging allocation failures. The device is built by `FaultingDevice`, which fails chosen
// allocation calls.
// ---------------------------------------------------------------------------------------------

namespace {

constexpr VkDeviceSize kMiB = 1024u * 1024u;

} // namespace

class VulkanStagingFaultTest : public ::testing::Test {
 protected:
  using Call = util::device::vulkan::FaultingDevice::Call;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    if (!IGL_VULKAN_USE_VMA) {
      GTEST_SKIP() << "The allocation tests run on the VMA path";
    }
    fd_ = std::make_unique<util::device::vulkan::FaultingDevice>(
        util::device::vulkan::getContextConfig());
    ASSERT_NE(fd_->device(), nullptr) << fd_->creationResult().message.c_str();
  }

  [[nodiscard]] igl::vulkan::VulkanContext& ctx() const {
    return fd_->context();
  }
  [[nodiscard]] igl::vulkan::VulkanStagingDevice& staging() const {
    return *ctx().stagingDevice_;
  }

  // Waits for the staging pool, returns every block and releases every fully recovered staging
  // buffer (the context's own dummy upload used one), leaving no free bytes and no region.
  void drainStaging() const {
    fd_->drainStaging();
    ASSERT_EQ(staging().getFreeStagingBufferSize(), 0u);
  }

  // A device-local buffer that staging uploads can target.
  [[nodiscard]] std::unique_ptr<igl::vulkan::VulkanBuffer> makeTarget(VkDeviceSize size) const {
    Result result;
    auto buffer =
        ctx().createBuffer(size,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           &result,
                           "Buffer: staging fault target");
    EXPECT_TRUE(result.isOk()) << result.message.c_str();
    return buffer;
  }

  std::unique_ptr<util::device::vulkan::FaultingDevice> fd_;
};

TEST_F(VulkanStagingFaultTest, StagingAllocationFailurePublishesNothingAndRecordsNoCopy) {
  drainStaging();
  const auto target = makeTarget(kMiB);
  ASSERT_NE(target, nullptr);
  const std::vector<uint8_t> data(128 * 1024, 0x33);

  fd_->resetCounters();
  fd_->arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/0, /*count=*/UINT32_MAX);
  EXPECT_FALSE(staging().bufferSubData(*target, 0, data.size(), data.data()));

  // No region was published (no free bytes).
  EXPECT_EQ(staging().getFreeStagingBufferSize(), 0u);
  EXPECT_GE(fd_->failed(Call::CreateBuffer), 2u);
}

TEST_F(VulkanStagingFaultTest, StagingRecoversWhenALaterAllocationSucceeds) {
  drainStaging();
  const auto target = makeTarget(kMiB);
  ASSERT_NE(target, nullptr);
  std::vector<uint8_t> data(128 * 1024);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i * 7u + 3u);
  }

  // The first allocation (and its retry) fails, the one after the staging device's wait works.
  fd_->resetCounters();
  fd_->arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/0, /*count=*/2);
  ASSERT_TRUE(staging().bufferSubData(*target, 0, data.size(), data.data()));
  EXPECT_EQ(fd_->failed(Call::CreateBuffer), 2u);

  std::vector<uint8_t> readBack(data.size(), 0);
  ASSERT_TRUE(staging().getBufferSubData(*target, 0, readBack.size(), readBack.data()));
  EXPECT_EQ(readBack, data);
}

TEST_F(VulkanStagingFaultTest, RepeatedStagingFailureNeverLoopsWithoutProgress) {
  drainStaging();
  const auto target = makeTarget(kMiB);
  ASSERT_NE(target, nullptr);
  const std::vector<uint8_t> data(512 * 1024, 0x44);

  fd_->resetCounters();
  fd_->arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/0, /*count=*/UINT32_MAX);
  constexpr uint32_t kCalls = 3;
  for (uint32_t i = 0; i < kCalls; ++i) {
    EXPECT_FALSE(staging().bufferSubData(*target, 0, data.size(), data.data()));
  }
  // Each call tries to allocate before and after its one wait, each try being a creation and its
  // drain retry: bounded work, no zero-progress iteration.
  EXPECT_LE(fd_->reached(Call::CreateBuffer), 4u * kCalls);
  EXPECT_GE(fd_->reached(Call::CreateBuffer), kCalls);
  EXPECT_EQ(staging().getFreeStagingBufferSize(), 0u);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
