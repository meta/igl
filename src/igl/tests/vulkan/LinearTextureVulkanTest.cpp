/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <igl/Device.h>
#include <igl/tests/util/TestDevice.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanHelpers.h>
#include <igl/vulkan/VulkanImmediateCommands.h>
#include <igl/vulkan/VulkanStagingDevice.h>
#include <igl/vulkan/VulkanTexture.h>

namespace igl::tests {

namespace {

// Copies one subresource of `image` into a tightly packed buffer with `vkCmdCopyImageToBuffer()`,
// so the bytes are read independently of how they were written.
void readBackSubresource(vulkan::VulkanContext& ctx,
                         vulkan::VulkanImage& image,
                         uint32_t mipLevel,
                         uint32_t layer,
                         const VkExtent3D& extent,
                         std::vector<uint8_t>& bytes) {
  Result result;
  const auto readback = ctx.createBuffer(bytes.size(),
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                         &result,
                                         "Texture readback");
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(readback, nullptr);
  ASSERT_NE(readback->getMappedPtr(), nullptr);

  const auto& commands = ctx.stagingDevice_->immediate->acquire();
  ivkImageMemoryBarrier(&ctx.vf_,
                        commands.cmdBuf,
                        image.getVkImage(),
                        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT,
                        image.imageLayout_,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VkImageSubresourceRange{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = VK_REMAINING_MIP_LEVELS,
                            .baseArrayLayer = 0,
                            .layerCount = VK_REMAINING_ARRAY_LAYERS,
                        });
  image.imageLayout_ = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  const VkBufferImageCopy copy{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .mipLevel = mipLevel,
              .baseArrayLayer = layer,
              .layerCount = 1,
          },
      .imageOffset =
          {
              .x = 0,
              .y = 0,
              .z = 0,
          },
      .imageExtent = extent,
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
    readback->invalidateMappedMemory(0, bytes.size());
  }
  std::copy(readback->getMappedPtr(), readback->getMappedPtr() + bytes.size(), bytes.begin());
}

// The message `ITexture::upload()` returns when a row does not fit in a host-mapped image's memory.
constexpr std::string_view kOutsideTheMemory =
    "The upload range does not fit in the mapped image memory";

// Fills `count` bytes in which no 256-byte run equals another within 64 KiB, so bytes read from
// the wrong offset do not compare equal.
std::vector<uint8_t> makeBytes(size_t count, uint8_t tag) {
  std::vector<uint8_t> bytes(count);
  for (size_t i = 0; i < count; ++i) {
    bytes[i] = static_cast<uint8_t>(tag + 7 * i + 13 * (i / 256));
  }
  return bytes;
}

std::vector<uint8_t> slice(const std::vector<uint8_t>& bytes, size_t offset, size_t count) {
  return {bytes.begin() + static_cast<std::ptrdiff_t>(offset),
          bytes.begin() + static_cast<std::ptrdiff_t>(offset + count)};
}

// Expects `actual` to equal `before`, except that the bytes at each of `rows` (an offset and the
// bytes an upload wrote there before it was refused) may hold either their `before` bytes or the
// written ones: a refused upload leaves the rows before the refused one unspecified.
void expectUnchangedExceptRowsBeforeTheRefusal(
    const std::vector<uint8_t>& actual,
    std::vector<uint8_t> before,
    const std::vector<std::pair<size_t, std::vector<uint8_t>>>& rows) {
  ASSERT_EQ(actual.size(), before.size());
  for (const auto& [offset, written] : rows) {
    const auto current = slice(actual, offset, written.size());
    EXPECT_TRUE(current == slice(before, offset, written.size()) || current == written)
        << "bytes at " << offset;
    std::copy(current.begin(), current.end(), before.begin() + static_cast<std::ptrdiff_t>(offset));
  }
  EXPECT_EQ(actual, before);
}

// Copies `rows` rows of `rowBytes` bytes from `src` into `dst`, `dstPitch` bytes apart, starting
// at `dstOffset`.
void copyRows(const uint8_t* src,
              size_t rows,
              size_t rowBytes,
              std::vector<uint8_t>& dst,
              size_t dstOffset,
              size_t dstPitch) {
  for (size_t row = 0; row < rows; ++row) {
    std::copy(src + row * rowBytes,
              src + (row + 1) * rowBytes,
              dst.begin() + static_cast<std::ptrdiff_t>(dstOffset + row * dstPitch));
  }
}

} // namespace

class LinearTextureVulkanTest : public ::testing::Test {
 public:
  // Set up common resources. This will create a device
  void SetUp() override {
    // Turn off debug break so unit tests can run
    setDebugBreakEnabled(false);

    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;

  vulkan::VulkanContext& context() {
    return static_cast<vulkan::Device&>(*iglDev_).getVulkanContext();
  }

  // Sets `unsupported` when the device has no linear image of `desc`'s type, size, number of levels
  // or number of layers in `expectedFormat`, which must be the format the backend picks for it.
  void checkLinearLimits(const TextureDesc& desc,
                         VkFormat expectedFormat,
                         std::string& unsupported) {
    ASSERT_EQ(vulkan::textureFormatToVkFormat(desc.format), expectedFormat);
    const bool isCube = desc.type == TextureType::Cube;
    VkImageFormatProperties limits{};
    if (context().vf_.vkGetPhysicalDeviceImageFormatProperties(
            context().getVkPhysicalDevice(),
            expectedFormat,
            desc.type == TextureType::ThreeD ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
            VK_IMAGE_TILING_LINEAR,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            isCube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
            &limits) != VK_SUCCESS) {
      unsupported = "The device has no linear image of this format and type";
      return;
    }
    if (limits.maxMipLevels < desc.numMipLevels) {
      unsupported =
          "Linear images have at most " + std::to_string(limits.maxMipLevels) + " mip levels";
      return;
    }
    if (limits.maxArrayLayers < (isCube ? 6u : desc.numLayers)) {
      unsupported =
          "Linear images have at most " + std::to_string(limits.maxArrayLayers) + " layers";
      return;
    }
    if (limits.maxExtent.width < desc.width || limits.maxExtent.height < desc.height ||
        limits.maxExtent.depth < desc.depth) {
      unsupported = "Linear images of this type are smaller than the texture";
    }
  }

  // Creates `desc` as a linearly tiled texture in host-visible memory, so that uploads to it take
  // the host-mapped path, or sets `unsupported` as `checkLinearLimits()` does.
  void createMappedLinearTexture(TextureDesc desc,
                                 VkFormat expectedFormat,
                                 std::shared_ptr<ITexture>& texture,
                                 std::string& unsupported) {
    desc.tiling = TextureDesc::TextureTiling::Linear;
    desc.storage = ResourceStorage::Shared;
    ASSERT_NO_FATAL_FAILURE(checkLinearLimits(desc, expectedFormat, unsupported));
    if (!unsupported.empty()) {
      return;
    }
    Result ret;
    texture = iglDev_->createTexture(desc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    ASSERT_NE(texture, nullptr);
    ASSERT_TRUE(
        static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image.isMappedPtrAccessible());
  }

  // The memory size the driver requires for `texture`'s image, read independently of the size the
  // backend records for its allocation.
  VkDeviceSize requiredMemorySize(const ITexture& texture) {
    VkMemoryRequirements requirements{};
    context().vf_.vkGetImageMemoryRequirements(
        context().getVkDevice(),
        static_cast<const vulkan::Texture&>(texture).getVulkanTexture().image.getVkImage(),
        &requirements);
    return requirements.size;
  }

  // Reads back every depth slice of `level` of one array layer or cube face, tightly packed.
  std::vector<uint8_t> readBack(const ITexture& texture, uint32_t level, uint32_t layer) {
    const auto range = texture.getFullRange(level).withNumLayers(1).withNumFaces(1);
    std::vector<uint8_t> bytes(texture.getProperties().getBytesPerRange(range));
    readBackSubresource(context(),
                        static_cast<const vulkan::Texture&>(texture).getVulkanTexture().image,
                        level,
                        layer,
                        VkExtent3D{
                            .width = range.width,
                            .height = range.height,
                            .depth = range.depth,
                        },
                        bytes);
    return bytes;
  }
};

// Without a storage request, a linear texture gets host-visible memory on a device whose only
// memory heap is host-visible and device-local, so its uploads also take the host-mapped path and
// do not use the staging device.
TEST_F(LinearTextureVulkanTest, WithoutStorageUploadsThroughItsMapping) {
  VkPhysicalDeviceMemoryProperties memory{};
  context().vf_.vkGetPhysicalDeviceMemoryProperties(context().getVkPhysicalDevice(), &memory);
  bool hasHostVisibleDeviceLocalType = false;
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
    constexpr VkMemoryPropertyFlags kFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    hasHostVisibleDeviceLocalType |= (memory.memoryTypes[i].propertyFlags & kFlags) == kFlags;
  }
  if (memory.memoryHeapCount != 1 || !hasHostVisibleDeviceLocalType) {
    GTEST_SKIP() << "Linear textures get device-local memory on a device with separate heaps";
  }
  if (!IGL_VULKAN_USE_VMA) {
    GTEST_SKIP() << "Staging allocations can be failed on the VMA path only";
  }
  TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 8, 8, TextureDesc::TextureUsageBits::Sampled);
  desc.tiling = TextureDesc::TextureTiling::Linear;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(checkLinearLimits(desc, VK_FORMAT_R8G8B8A8_UNORM, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }

  // A device whose staging allocations can be made to fail.
  using Call = util::device::vulkan::FaultingDevice::Call;
  util::device::vulkan::FaultingDevice faulting(util::device::vulkan::getContextConfig());
  ASSERT_NE(faulting.device(), nullptr) << faulting.creationResult().message;
  ASSERT_FALSE(faulting.context().useStagingForBuffers_);
  Result ret;
  const auto texture = faulting.device()->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(texture, nullptr);
  auto& image = static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image;
  ASSERT_TRUE(image.isMappedPtrAccessible());

  // With no staging buffer left and every new one failing, an upload through the staging device
  // would ask for a buffer and fail.
  faulting.drainStaging();
  faulting.resetCounters();
  faulting.arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/0, /*count=*/UINT32_MAX);
  const uint64_t lastSubmit =
      faulting.context().stagingDevice_->immediate->getLastSubmitHandle().handle();
  const auto texels = makeBytes(8 * 8 * 4, 5);
  ret = texture->upload(texture->getFullRange(), texels.data());
  EXPECT_EQ(faulting.reached(Call::CreateBuffer), 0u)
      << "The upload went through the staging device";
  EXPECT_EQ(faulting.context().stagingDevice_->immediate->getLastSubmitHandle().handle(),
            lastSubmit)
      << "The upload submitted a staging copy";
  faulting.disarm(Call::CreateBuffer);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  std::vector<uint8_t> actual(texels.size());
  ASSERT_NO_FATAL_FAILURE(readBackSubresource(
      faulting.context(), image, 0, 0, VkExtent3D{.width = 8, .height = 8, .depth = 1}, actual));
  EXPECT_EQ(actual, texels);
}

// A linearly tiled texture with host-visible memory is written through its mapping. Each level of a
// packed chain lands in its own subresource, at that level's row pitch.
TEST_F(LinearTextureVulkanTest, UploadWritesEveryMipLevel) {
  TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 32, 32, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 3;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto range = texture->getFullMipRange();
  const auto& properties = texture->getProperties();
  const auto chain = makeBytes(properties.getBytesPerRange(range), 1);
  const Result ret = texture->upload(range, chain.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  size_t offset = 0;
  for (uint32_t level = 0; level < 3; ++level) {
    const size_t levelBytes = properties.getBytesPerRange(range.atMipLevel(level));
    EXPECT_EQ(readBack(*texture, level, 0), slice(chain, offset, levelBytes)) << "level " << level;
    offset += levelBytes;
  }

  // Another linear texture takes its own layout: one of another size while the first is alive, and
  // one of the same size and format but two layers once the first is gone.
  TextureDesc smallerDesc = desc;
  smallerDesc.width = 8;
  smallerDesc.height = 8;
  smallerDesc.numMipLevels = 2;
  std::shared_ptr<ITexture> smaller;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(smallerDesc, VK_FORMAT_R8G8B8A8_UNORM, smaller, unsupported));
  ASSERT_TRUE(unsupported.empty()) << unsupported;
  const auto smallerChain = makeBytes(8 * 8 * 4 + 4 * 4 * 4, 61);
  ASSERT_TRUE(smaller->upload(smaller->getFullMipRange(), smallerChain.data()).isOk());
  EXPECT_EQ(readBack(*smaller, 0, 0), slice(smallerChain, 0, 8 * 8 * 4));
  EXPECT_EQ(readBack(*smaller, 1, 0), slice(smallerChain, 8 * 8 * 4, 4 * 4 * 4));

  texture.reset();
  TextureDesc layersDesc = desc;
  layersDesc.type = TextureType::TwoDArray;
  layersDesc.numLayers = 2;
  std::shared_ptr<ITexture> layers;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(layersDesc, VK_FORMAT_R8G8B8A8_UNORM, layers, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto layersRange = layers->getFullMipRange();
  const auto& layersProperties = layers->getProperties();
  const auto layersChain = makeBytes(layersProperties.getBytesPerRange(layersRange), 62);
  ASSERT_TRUE(layers->upload(layersRange, layersChain.data()).isOk());
  size_t layersOffset = 0;
  for (uint32_t level = 0; level < 3; ++level) {
    const size_t layerBytes =
        layersProperties.getBytesPerRange(layersRange.atMipLevel(level).atLayer(0));
    for (uint32_t layer = 0; layer < 2; ++layer) {
      EXPECT_EQ(readBack(*layers, level, layer), slice(layersChain, layersOffset, layerBytes))
          << "level " << level << ", layer " << layer;
      layersOffset += layerBytes;
    }
  }
}

// An upload that starts above level 0 writes the levels it names and leaves the others as they
// were.
TEST_F(LinearTextureVulkanTest, UploadStartingAboveLevelZeroWritesOnlyItsLevels) {
  TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 8, 8, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 4;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto& properties = texture->getProperties();
  const auto chain = makeBytes(properties.getBytesPerRange(texture->getFullMipRange()), 1);
  Result ret = texture->upload(texture->getFullMipRange(), chain.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto levels1And2 = makeBytes(4 * 4 * 4 + 2 * 2 * 4, 100);
  ret = texture->upload(texture->getFullRange(1, 2), levels1And2.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), slice(chain, 0, 256));
  EXPECT_EQ(readBack(*texture, 1, 0), slice(levels1And2, 0, 64));
  EXPECT_EQ(readBack(*texture, 2, 0), slice(levels1And2, 64, 16));
  EXPECT_EQ(readBack(*texture, 3, 0), slice(chain, 256 + 64 + 16, 4));
}

// A block-compressed chain is written in rows of blocks, and a region of whole blocks replaces
// exactly those blocks.
TEST_F(LinearTextureVulkanTest, UploadWritesBlockCompressedLevelsAndRegions) {
  constexpr size_t kBlockBytes = 16;
  TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_BC7_UNORM_4x4, 16, 16, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 2;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_BC7_UNORM_BLOCK, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto chain = makeBytes(16 * kBlockBytes + 4 * kBlockBytes, 1);
  Result ret = texture->upload(texture->getFullMipRange(), chain.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  EXPECT_EQ(readBack(*texture, 0, 0), slice(chain, 0, 16 * kBlockBytes));
  EXPECT_EQ(readBack(*texture, 1, 0), slice(chain, 16 * kBlockBytes, 4 * kBlockBytes));

  // Texels (4, 4) to (11, 11) of level 0 are blocks (1, 1) to (2, 2) of its 4 x 4 blocks.
  const auto region = makeBytes(4 * kBlockBytes, 150);
  ret = texture->upload(TextureRangeDesc::new2D(4, 4, 8, 8), region.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto expected = slice(chain, 0, 16 * kBlockBytes);
  copyRows(region.data(), 2, 2 * kBlockBytes, expected, (4 + 1) * kBlockBytes, 4 * kBlockBytes);
  EXPECT_EQ(readBack(*texture, 0, 0), expected);
  EXPECT_EQ(readBack(*texture, 1, 0), slice(chain, 16 * kBlockBytes, 4 * kBlockBytes));
}

// `bytesPerRow` describes the source rows; the destination rows are at the image's own row pitch.
TEST_F(LinearTextureVulkanTest, UploadReadsSourceRowsAtBytesPerRow) {
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kHeight = 3;
  constexpr size_t kRowBytes = kWidth * 4;
  constexpr size_t kSourceBytesPerRow = kRowBytes + 8;
  const TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, kWidth, kHeight, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto texels = makeBytes(kRowBytes * kHeight, 3);
  std::vector<uint8_t> source(kSourceBytesPerRow * kHeight, 0xEE);
  copyRows(texels.data(), kHeight, kRowBytes, source, 0, kSourceBytesPerRow);
  const Result ret = texture->upload(texture->getFullRange(), source.data(), kSourceBytesPerRow);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), texels);
}

// Every row of every depth slice is read `bytesPerRow` apart.
TEST_F(LinearTextureVulkanTest, UploadReadsEverySliceAtBytesPerRow) {
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kDepth = 3;
  constexpr size_t kRowBytes = kWidth * 4;
  constexpr size_t kSourceBytesPerRow = kRowBytes + 8;
  const TextureDesc desc = TextureDesc::new3D(
      TextureFormat::RGBA_UNorm8, kWidth, kHeight, kDepth, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto texels = makeBytes(kRowBytes * kHeight * kDepth, 3);
  std::vector<uint8_t> source(kSourceBytesPerRow * kHeight * kDepth, 0xEE);
  copyRows(texels.data(), kHeight * kDepth, kRowBytes, source, 0, kSourceBytesPerRow);
  const Result ret = texture->upload(texture->getFullRange(), source.data(), kSourceBytesPerRow);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), texels);
}

// An upload of a region of one level leaves the rest of that level and the other levels as they
// were.
TEST_F(LinearTextureVulkanTest, UploadWritesOnlyTheRange) {
  TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 8, 8, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 2;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto chain = makeBytes(8 * 8 * 4 + 4 * 4 * 4, 1);
  Result ret = texture->upload(texture->getFullMipRange(), chain.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  constexpr uint32_t kX = 2;
  constexpr uint32_t kY = 3;
  constexpr uint32_t kWidth = 3;
  constexpr uint32_t kHeight = 2;
  const auto region = makeBytes(kWidth * kHeight * 4, 200);
  ret = texture->upload(TextureRangeDesc::new2D(kX, kY, kWidth, kHeight), region.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  auto expectedLevel0 = slice(chain, 0, 8 * 8 * 4);
  copyRows(region.data(), kHeight, kWidth * 4, expectedLevel0, (kY * 8 + kX) * 4, 8 * 4);
  EXPECT_EQ(readBack(*texture, 0, 0), expectedLevel0);
  EXPECT_EQ(readBack(*texture, 1, 0), slice(chain, 8 * 8 * 4, 4 * 4 * 4));
}

// A region's source rows are read `bytesPerRow` apart, and only the region's own texels are
// written.
TEST_F(LinearTextureVulkanTest, UploadOfARegionReadsSourceRowsAtBytesPerRow) {
  const TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 8, 8, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto level = makeBytes(8 * 8 * 4, 1);
  Result ret = texture->upload(texture->getFullRange(), level.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  constexpr uint32_t kX = 2;
  constexpr uint32_t kY = 3;
  constexpr uint32_t kWidth = 3;
  constexpr uint32_t kHeight = 2;
  constexpr size_t kRowBytes = kWidth * 4;
  constexpr size_t kSourceBytesPerRow = kRowBytes + 8;
  // The padding between the rows holds bytes of its own, so writing it shows up in the readback.
  const auto source = makeBytes(kSourceBytesPerRow * kHeight, 200);
  ret = texture->upload(
      TextureRangeDesc::new2D(kX, kY, kWidth, kHeight), source.data(), kSourceBytesPerRow);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  auto expected = level;
  for (size_t row = 0; row < kHeight; ++row) {
    std::copy(source.begin() + static_cast<std::ptrdiff_t>(row * kSourceBytesPerRow),
              source.begin() + static_cast<std::ptrdiff_t>(row * kSourceBytesPerRow + kRowBytes),
              expected.begin() + static_cast<std::ptrdiff_t>(((kY + row) * 8 + kX) * 4));
  }
  EXPECT_EQ(readBack(*texture, 0, 0), expected);
}

// A row that would end past the mapped memory is refused. At level 1 of this chain the range starts
// at x = 16 of a 16-texel-wide level, so each of its rows lands one row further on and the last one
// runs past the end of the image's memory. Only the rows before it may have changed. One column to
// the left the same range fits.
TEST_F(LinearTextureVulkanTest, UploadPastTheMappedMemoryIsRefused) {
  TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 33, 8, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 2;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto& image = static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image;
  const VkImageSubresource level1Subresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 1,
      .arrayLayer = 0,
  };
  VkSubresourceLayout layout = {};
  context().vf_.vkGetImageSubresourceLayout(
      context().getVkDevice(), image.getVkImage(), &level1Subresource, &layout);
  if (layout.rowPitch != 16 * 4 ||
      layout.offset + 3 * layout.rowPitch + 16 * 4 + 4 <= requiredMemorySize(*texture)) {
    GTEST_SKIP() << "Level 1 is not packed, or the refused row does not run past its memory";
  }
  constexpr size_t kLevel0Bytes = 33 * 8 * 4;
  constexpr size_t kLevel1Bytes = 16 * 4 * 4;
  const auto before = makeBytes(kLevel0Bytes + kLevel1Bytes, 30);
  ASSERT_TRUE(texture->upload(texture->getFullMipRange(), before.data()).isOk());

  const auto column = makeBytes(8 * 4 + 4 * 4, 90);
  Result ret = texture->upload(TextureRangeDesc::new2D(32, 0, 1, 8, 0, 2), column.data());
  EXPECT_EQ(ret.code, Result::Code::ArgumentOutOfRange) << ret.message;
  EXPECT_EQ(ret.message, kOutsideTheMemory);
  // Level 0's column 32 and the first three rows of level 1, which land on texels (0, 1) to (0, 3),
  // come before the refused row.
  std::vector<std::pair<size_t, std::vector<uint8_t>>> level0Rows;
  level0Rows.reserve(8);
  for (size_t y = 0; y < 8; ++y) {
    level0Rows.emplace_back((y * 33 + 32) * 4, slice(column, y * 4, 4));
  }
  expectUnchangedExceptRowsBeforeTheRefusal(
      readBack(*texture, 0, 0), slice(before, 0, kLevel0Bytes), level0Rows);
  std::vector<std::pair<size_t, std::vector<uint8_t>>> level1Rows;
  level1Rows.reserve(3);
  for (size_t y = 0; y < 3; ++y) {
    level1Rows.emplace_back((y + 1) * 16 * 4, slice(column, 32 + y * 4, 4));
  }
  expectUnchangedExceptRowsBeforeTheRefusal(
      readBack(*texture, 1, 0), slice(before, kLevel0Bytes, kLevel1Bytes), level1Rows);

  ret = texture->upload(TextureRangeDesc::new2D(31, 0, 1, 8, 0, 2), column.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto written0 = readBack(*texture, 0, 0);
  for (size_t y = 0; y < 8; ++y) {
    EXPECT_EQ(slice(written0, (y * 33 + 31) * 4, 4), slice(column, y * 4, 4))
        << "level 0, row " << y;
  }
  const auto written1 = readBack(*texture, 1, 0);
  for (size_t y = 0; y < 4; ++y) {
    EXPECT_EQ(slice(written1, (y * 16 + 15) * 4, 4), slice(column, 32 + y * 4, 4))
        << "level 1, row " << y;
  }
}

// A row that starts past the end of the image's memory is refused too. At level 1 (2 x 1 x 2) of
// this chain the range is at y = 1 and z = 2, beyond the last slice. Only the row written at level
// 0 changes.
TEST_F(LinearTextureVulkanTest, UploadStartingPastTheMappedMemoryIsRefused) {
  TextureDesc desc = TextureDesc::new3D(
      TextureFormat::RGBA_UNorm8, 5, 3, 5, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 2;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto& image = static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image;
  const VkImageSubresource level1Subresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 1,
      .arrayLayer = 0,
  };
  VkSubresourceLayout layout = {};
  context().vf_.vkGetImageSubresourceLayout(
      context().getVkDevice(), image.getVkImage(), &level1Subresource, &layout);
  if (layout.offset + 2 * layout.depthPitch + layout.rowPitch <= requiredMemorySize(*texture)) {
    GTEST_SKIP() << "The refused row does not start past the image's memory";
  }
  constexpr size_t kLevel0Bytes = 5 * 3 * 5 * 4;
  constexpr size_t kLevel1Bytes = 2 * 1 * 2 * 4;
  const auto before = makeBytes(kLevel0Bytes + kLevel1Bytes, 40);
  ASSERT_TRUE(texture->upload(texture->getFullMipRange(), before.data()).isOk());

  const auto row = makeBytes(2 * 4 + 4, 95);
  const Result ret = texture->upload(TextureRangeDesc::new3D(0, 2, 4, 2, 1, 1, 0, 2), row.data());
  EXPECT_EQ(ret.code, Result::Code::ArgumentOutOfRange) << ret.message;
  EXPECT_EQ(ret.message, kOutsideTheMemory);
  // Level 0's texels (0, 2, 4) and (1, 2, 4) come before the refused row.
  constexpr size_t kRowStart = ((4 * 3 + 2) * 5) * 4;
  expectUnchangedExceptRowsBeforeTheRefusal(readBack(*texture, 0, 0),
                                            slice(before, 0, kLevel0Bytes),
                                            {{kRowStart, slice(row, 0, 2 * 4)}});
  EXPECT_EQ(readBack(*texture, 1, 0), slice(before, kLevel0Bytes, kLevel1Bytes));
}

// A row that starts inside the image's memory and ends past it is refused, and none of it is
// written. No valid range makes one on a packed layout, so the test shortens the memory the backend
// records for the image by 8 bytes: the last row then runs 8 bytes past it.
TEST_F(LinearTextureVulkanTest, UploadOfARowEndingPastTheMappedMemoryIsRefused) {
  const TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 16, 4, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  auto& image = static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image;
  const VkImageSubresource subresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .arrayLayer = 0,
  };
  VkSubresourceLayout layout = {};
  context().vf_.vkGetImageSubresourceLayout(
      context().getVkDevice(), image.getVkImage(), &subresource, &layout);
  const VkDeviceSize lastRowEnd = layout.offset + 3 * layout.rowPitch + 16 * 4;
  ASSERT_LE(lastRowEnd, image.allocatedSize);
  const auto before = makeBytes(16 * 4 * 4, 70);
  ASSERT_TRUE(texture->upload(texture->getFullRange(), before.data()).isOk());

  const VkDeviceSize allocatedSize = image.allocatedSize;
  image.allocatedSize = lastRowEnd - 8;
  const auto texels = makeBytes(16 * 4 * 4, 75);
  const Result ret = texture->upload(texture->getFullRange(), texels.data());
  image.allocatedSize = allocatedSize;
  EXPECT_EQ(ret.code, Result::Code::ArgumentOutOfRange) << ret.message;
  EXPECT_EQ(ret.message, kOutsideTheMemory);
  // Rows 0 to 2 come before the refused row; row 3 is not written.
  std::vector<std::pair<size_t, std::vector<uint8_t>>> rows;
  rows.reserve(3);
  for (size_t y = 0; y < 3; ++y) {
    rows.emplace_back(y * 16 * 4, slice(texels, y * 16 * 4, 16 * 4));
  }
  expectUnchangedExceptRowsBeforeTheRefusal(readBack(*texture, 0, 0), before, rows);
}

// A chain given the first level's packed `bytesPerRow`, which `ITexture::upload()` accepts, is read
// packed level after level, as `getBytesPerRange()` describes a chain.
TEST_F(LinearTextureVulkanTest, UploadReadsAChainGivenItsFirstLevelPitchPacked) {
  TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 32, 32, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 3;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  constexpr size_t kLevel0Bytes = 32 * 32 * 4;
  constexpr size_t kLevel1Bytes = 16 * 16 * 4;
  constexpr size_t kLevel2Bytes = 8 * 8 * 4;
  // Distinct bytes after the chain, which a reading at the first level's pitch would reach.
  const auto chain = makeBytes(kLevel0Bytes + kLevel1Bytes + kLevel2Bytes + 1024, 11);
  const Result ret = texture->upload(texture->getFullMipRange(), chain.data(), 32 * 4);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), slice(chain, 0, kLevel0Bytes));
  EXPECT_EQ(readBack(*texture, 1, 0), slice(chain, kLevel0Bytes, kLevel1Bytes));
  EXPECT_EQ(readBack(*texture, 2, 0), slice(chain, kLevel0Bytes + kLevel1Bytes, kLevel2Bytes));
}

// A single level between others is read `bytesPerRow` apart too, and the levels around it stay as
// they were.
TEST_F(LinearTextureVulkanTest, UploadOfALevelAboveZeroReadsRowsAtBytesPerRow) {
  TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 8, 8, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 3;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  constexpr size_t kLevel0Bytes = 8 * 8 * 4;
  constexpr size_t kLevel1Bytes = 4 * 4 * 4;
  constexpr size_t kLevel2Bytes = 2 * 2 * 4;
  const auto chain = makeBytes(kLevel0Bytes + kLevel1Bytes + kLevel2Bytes, 8);
  ASSERT_TRUE(texture->upload(texture->getFullMipRange(), chain.data()).isOk());

  constexpr size_t kRowBytes = 4 * 4;
  constexpr size_t kSourceBytesPerRow = kRowBytes + 8;
  const auto texels = makeBytes(kLevel1Bytes, 23);
  auto source = makeBytes(kSourceBytesPerRow * 4, 180);
  copyRows(texels.data(), 4, kRowBytes, source, 0, kSourceBytesPerRow);
  const Result ret = texture->upload(texture->getFullRange(1), source.data(), kSourceBytesPerRow);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), slice(chain, 0, kLevel0Bytes));
  EXPECT_EQ(readBack(*texture, 1, 0), texels);
  EXPECT_EQ(readBack(*texture, 2, 0), slice(chain, kLevel0Bytes + kLevel1Bytes, kLevel2Bytes));
}

// A block-compressed chain whose levels end in partial blocks: 20 x 12, 10 x 6 and 5 x 3 texels.
TEST_F(LinearTextureVulkanTest, UploadWritesABlockCompressedChainWithPartialBlocks) {
  TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_BC7_UNORM_4x4, 20, 12, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 3;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_BC7_UNORM_BLOCK, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto range = texture->getFullMipRange();
  const auto& properties = texture->getProperties();
  const auto chain = makeBytes(properties.getBytesPerRange(range), 51);
  const Result ret = texture->upload(range, chain.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  size_t offset = 0;
  for (uint32_t level = 0; level < 3; ++level) {
    const size_t levelBytes = properties.getBytesPerRange(range.atMipLevel(level));
    EXPECT_EQ(readBack(*texture, level, 0), slice(chain, offset, levelBytes)) << "level " << level;
    offset += levelBytes;
  }
}

// Every layer of a single level is read `bytesPerRow` apart.
TEST_F(LinearTextureVulkanTest, UploadReadsEveryLayerAtBytesPerRow) {
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kHeight = 4;
  constexpr uint32_t kLayers = 3;
  constexpr size_t kRowBytes = kWidth * 4;
  constexpr size_t kSourceBytesPerRow = kRowBytes + 8;
  const TextureDesc desc = TextureDesc::new2DArray(
      TextureFormat::RGBA_UNorm8, kWidth, kHeight, kLayers, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto texels = makeBytes(kRowBytes * kHeight * kLayers, 21);
  // The padding holds bytes of its own, so reading it as texels shows up in the readback.
  auto source = makeBytes(kSourceBytesPerRow * kHeight * kLayers, 170);
  copyRows(texels.data(), kHeight * kLayers, kRowBytes, source, 0, kSourceBytesPerRow);
  const Result ret = texture->upload(texture->getFullRange(), source.data(), kSourceBytesPerRow);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    EXPECT_EQ(readBack(*texture, 0, layer),
              slice(texels, layer * kRowBytes * kHeight, kRowBytes * kHeight))
        << "layer " << layer;
  }
}

// Every face of a single cube level is read `bytesPerRow` apart.
TEST_F(LinearTextureVulkanTest, UploadReadsEveryCubeFaceAtBytesPerRow) {
  constexpr uint32_t kSide = 5;
  constexpr size_t kRowBytes = kSide * 4;
  constexpr size_t kSourceBytesPerRow = kRowBytes + 8;
  constexpr size_t kFaceBytes = kRowBytes * kSide;
  const TextureDesc desc = TextureDesc::newCube(
      TextureFormat::RGBA_UNorm8, kSide, kSide, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto texels = makeBytes(kFaceBytes * 6, 31);
  auto source = makeBytes(kSourceBytesPerRow * kSide * 6, 190);
  copyRows(texels.data(), kSide * 6, kRowBytes, source, 0, kSourceBytesPerRow);
  const Result ret = texture->upload(texture->getFullRange(), source.data(), kSourceBytesPerRow);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  for (uint32_t face = 0; face < 6; ++face) {
    EXPECT_EQ(readBack(*texture, 0, face), slice(texels, face * kFaceBytes, kFaceBytes))
        << "face " << face;
  }
}

// A block-compressed level is read in rows of blocks `bytesPerRow` apart.
TEST_F(LinearTextureVulkanTest, UploadReadsBlockRowsAtBytesPerRow) {
  constexpr size_t kBlockBytes = 16;
  constexpr size_t kRowBytes = 4 * kBlockBytes;
  constexpr size_t kSourceBytesPerRow = kRowBytes + kBlockBytes;
  const TextureDesc desc = TextureDesc::new2D(
      TextureFormat::RGBA_BC7_UNORM_4x4, 16, 16, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_BC7_UNORM_BLOCK, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto blocks = makeBytes(4 * kRowBytes, 41);
  auto source = makeBytes(4 * kSourceBytesPerRow, 210);
  copyRows(blocks.data(), 4, kRowBytes, source, 0, kSourceBytesPerRow);
  const Result ret = texture->upload(texture->getFullRange(), source.data(), kSourceBytesPerRow);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), blocks);
}

// A region given for several levels is scaled down at each level.
TEST_F(LinearTextureVulkanTest, UploadWritesARegionOfEveryLevel) {
  TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 8, 8, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 2;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto chain = makeBytes(8 * 8 * 4 + 4 * 4 * 4, 1);
  Result ret = texture->upload(texture->getFullMipRange(), chain.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  // (2, 2) 4 x 4 at level 0 is (1, 1) 2 x 2 at level 1.
  const auto region = makeBytes(4 * 4 * 4 + 2 * 2 * 4, 150);
  ret = texture->upload(TextureRangeDesc::new2D(2, 2, 4, 4, 0, 2), region.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  auto expectedLevel0 = slice(chain, 0, 8 * 8 * 4);
  copyRows(region.data(), 4, 4 * 4, expectedLevel0, (2 * 8 + 2) * 4, 8 * 4);
  auto expectedLevel1 = slice(chain, 8 * 8 * 4, 4 * 4 * 4);
  copyRows(region.data() + 4 * 4 * 4, 2, 2 * 4, expectedLevel1, (1 * 4 + 1) * 4, 4 * 4);
  EXPECT_EQ(readBack(*texture, 0, 0), expectedLevel0);
  EXPECT_EQ(readBack(*texture, 1, 0), expectedLevel1);
}

// An upload to one layer of a 2D array writes that layer and leaves the others as they were.
TEST_F(LinearTextureVulkanTest, UploadWritesTheRequestedLayer) {
  constexpr uint32_t kSize = 4;
  constexpr size_t kLayerBytes = kSize * kSize * 4;
  const TextureDesc desc = TextureDesc::new2DArray(
      TextureFormat::RGBA_UNorm8, kSize, kSize, 3, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto layers = makeBytes(kLayerBytes * 3, 1);
  Result ret = texture->upload(texture->getFullRange(), layers.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto layer1 = makeBytes(kLayerBytes, 100);
  ret = texture->upload(TextureRangeDesc::new2DArray(0, 0, kSize, kSize, 1, 1), layer1.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), slice(layers, 0, kLayerBytes));
  EXPECT_EQ(readBack(*texture, 0, 1), layer1);
  EXPECT_EQ(readBack(*texture, 0, 2), slice(layers, 2 * kLayerBytes, kLayerBytes));
}

// A chain of a 2D array is packed level by level, each level holding every layer.
TEST_F(LinearTextureVulkanTest, UploadWritesEveryLayerOfEveryLevel) {
  TextureDesc desc = TextureDesc::new2DArray(
      TextureFormat::RGBA_UNorm8, 4, 4, 3, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 2;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  constexpr size_t kLevel0LayerBytes = 4 * 4 * 4;
  constexpr size_t kLevel1LayerBytes = 2 * 2 * 4;
  const auto chain = makeBytes(3 * kLevel0LayerBytes + 3 * kLevel1LayerBytes, 1);
  const Result ret = texture->upload(texture->getFullMipRange(), chain.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  for (uint32_t layer = 0; layer < 3; ++layer) {
    EXPECT_EQ(readBack(*texture, 0, layer),
              slice(chain, layer * kLevel0LayerBytes, kLevel0LayerBytes))
        << "level 0, layer " << layer;
    EXPECT_EQ(readBack(*texture, 1, layer),
              slice(chain, 3 * kLevel0LayerBytes + layer * kLevel1LayerBytes, kLevel1LayerBytes))
        << "level 1, layer " << layer;
  }
}

// An upload to one face of a cube writes that face and leaves the others as they were.
TEST_F(LinearTextureVulkanTest, UploadWritesTheRequestedCubeFace) {
  constexpr uint32_t kSize = 4;
  constexpr size_t kFaceBytes = kSize * kSize * 4;
  const TextureDesc desc = TextureDesc::newCube(
      TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto faces = makeBytes(kFaceBytes * 6, 1);
  Result ret = texture->upload(texture->getFullRange(), faces.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto face3 = makeBytes(kFaceBytes, 100);
  ret = texture->upload(TextureRangeDesc::newCubeFace(0, 0, kSize, kSize, 3), face3.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  for (uint32_t face = 0; face < 6; ++face) {
    EXPECT_EQ(readBack(*texture, 0, face),
              face == 3 ? face3 : slice(faces, face * kFaceBytes, kFaceBytes))
        << "face " << face;
  }
}

// Each depth slice of a 3D texture is written at the depth pitch the driver reports, and an upload
// of one slice leaves the others as they were.
TEST_F(LinearTextureVulkanTest, UploadWritesEachDepthSlice) {
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kDepth = 3;
  constexpr size_t kSliceBytes = kWidth * kHeight * 4;
  const TextureDesc desc = TextureDesc::new3D(
      TextureFormat::RGBA_UNorm8, kWidth, kHeight, kDepth, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  auto expected = makeBytes(kSliceBytes * kDepth, 1);
  Result ret = texture->upload(texture->getFullRange(), expected.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto slice2 = makeBytes(kSliceBytes, 100);
  ret = texture->upload(TextureRangeDesc::new3D(0, 0, 2, kWidth, kHeight, 1), slice2.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  std::copy(slice2.begin(),
            slice2.end(),
            expected.begin() + static_cast<std::ptrdiff_t>(2 * kSliceBytes));
  EXPECT_EQ(readBack(*texture, 0, 0), expected);
}

// Every level of a 3D chain holds all of its own depth slices.
TEST_F(LinearTextureVulkanTest, UploadWritesEveryLevelOfA3DTexture) {
  TextureDesc desc = TextureDesc::new3D(
      TextureFormat::RGBA_UNorm8, 4, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  desc.numMipLevels = 2;
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  constexpr size_t kLevel0Bytes = 4 * 4 * 4 * 4;
  constexpr size_t kLevel1Bytes = 2 * 2 * 2 * 4;
  const auto chain = makeBytes(kLevel0Bytes + kLevel1Bytes, 1);
  const Result ret = texture->upload(texture->getFullMipRange(), chain.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), slice(chain, 0, kLevel0Bytes));
  EXPECT_EQ(readBack(*texture, 1, 0), slice(chain, kLevel0Bytes, kLevel1Bytes));
}

// The last row is written even when it ends exactly where the image's memory ends. That needs a
// driver that lays the image out so, as Lavapipe does with a 64-byte row pitch.
TEST_F(LinearTextureVulkanTest, UploadWritesTheLastRow) {
  const TextureDesc desc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 16, 4, TextureDesc::TextureUsageBits::Sampled);
  std::shared_ptr<ITexture> texture;
  std::string unsupported;
  ASSERT_NO_FATAL_FAILURE(
      createMappedLinearTexture(desc, VK_FORMAT_R8G8B8A8_UNORM, texture, unsupported));
  if (!unsupported.empty()) {
    GTEST_SKIP() << unsupported;
  }
  const auto& image = static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image;
  const VkImageSubresource subresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .arrayLayer = 0,
  };
  VkSubresourceLayout layout = {};
  context().vf_.vkGetImageSubresourceLayout(
      context().getVkDevice(), image.getVkImage(), &subresource, &layout);
  if (layout.offset + 3 * layout.rowPitch + 16 * 4 != requiredMemorySize(*texture)) {
    GTEST_SKIP() << "The last row does not end where the image memory ends";
  }
  const auto texels = makeBytes(16 * 4 * 4, 7);
  const Result ret = texture->upload(texture->getFullRange(), texels.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(readBack(*texture, 0, 0), texels);
}

} // namespace igl::tests
