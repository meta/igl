/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <IGLU/texture_loader/ktx2/Header.h>
#include <IGLU/texture_loader/ktx2/TextureLoaderFactory.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>
#include <igl/vulkan/util/TextureFormat.h>

namespace igl::tests::ktx2 {

namespace {

template<typename T>
T align(T offset, T alignment) {
  return (offset + (alignment - 1)) & ~(alignment - 1);
}

std::vector<uint8_t> getBuffer(uint32_t capacity) {
  std::vector<uint8_t> buffer(static_cast<size_t>(capacity));
  return buffer;
}

template<typename T>
void put(std::vector<uint8_t>& buffer, uint32_t offset, T data) {
  if (buffer.size() < static_cast<size_t>(offset) + sizeof(T)) {
    throw std::runtime_error("Overflow when storing a word");
  }
  std::memcpy(buffer.data() + offset, &data, sizeof(T));
}

constexpr uint32_t kHeaderSize = 80u;
constexpr uint32_t kOffsetVkFormat = 12u;
constexpr uint32_t kOffsetTypeSize = 16u;
constexpr uint32_t kOffsetWidth = 20u;
constexpr uint32_t kOffsetHeight = 24u;
constexpr uint32_t kOffsetFaceCount = 36u;
constexpr uint32_t kOffsetLevelCount = 40u;
constexpr uint32_t kOffsetSupercompressionScheme = 44u;
constexpr uint32_t kOffsetDfdByteOffset = 48u;
constexpr uint32_t kOffsetDfdByteLength = 52u;
constexpr uint32_t kOffsetKvdByteOffset = 56u;
constexpr uint32_t kOffsetKvdByteLength = 60u;

constexpr uint32_t kMipmapMetadataSize = 24u;
constexpr uint32_t kDfdCompressedMetadataSize = 48u;
constexpr uint32_t kDfdUncompressedMetadataSize = 92u;

// NOLINTBEGIN(readability-identifier-naming)
constexpr uint32_t VK_FORMAT_R8G8B8A8_UNORM = 37u;
constexpr uint32_t VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG = 1000054000u;
constexpr uint32_t VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK = 147u;

constexpr uint8_t KHR_DF_FLAG_ALPHA_STRAIGHT = 0;

constexpr uint8_t KHR_DF_TRANSFER_SRGB = 1;
constexpr uint8_t KHR_DF_TRANSFER_LINEAR = 2;

constexpr uint8_t KHR_DF_PRIMARIES_BT709 = 1;

constexpr uint8_t KHR_DF_MODEL_RGBSDA = 1;
constexpr uint8_t KHR_DF_MODEL_ETC1 = 160;
constexpr uint8_t KHR_DF_MODEL_PVRTC = 164;

constexpr uint8_t KHR_DF_SAMPLE_DATATYPE_LINEAR = (1U << 4U);

constexpr uint8_t KHR_DF_CHANNEL_RGBSDA_RED = 0u;
constexpr uint8_t KHR_DF_CHANNEL_RGBSDA_GREEN = 1u;
constexpr uint8_t KHR_DF_CHANNEL_RGBSDA_BLUE = 2u;
constexpr uint8_t KHR_DF_CHANNEL_RGBSDA_ALPHA = 15u;
// NOLINTEND(readability-identifier-naming)

uint32_t getTotalHeaderSize(uint32_t vkFormat,
                            uint32_t numMipLevels,
                            uint32_t bytesOfKeyValueData) {
  return kHeaderSize + numMipLevels * kMipmapMetadataSize + bytesOfKeyValueData +
         (vkFormat == VK_FORMAT_R8G8B8A8_UNORM ? kDfdUncompressedMetadataSize
                                               : kDfdCompressedMetadataSize);
}

uint32_t getTotalDataSize(uint32_t vkFormat,
                          uint32_t width,
                          uint32_t height,
                          uint32_t numMipLevels) {
  const auto format = vulkan::util::vkTextureFormatToTextureFormat(static_cast<int32_t>(vkFormat));
  const auto properties = igl::TextureFormatProperties::fromTextureFormat(format);

  const auto range = TextureRangeDesc::new2D(0, 0, std::max(width, 1u), std::max(height, 1u));

  const uint32_t mipLevelAlignment = std::lcm(static_cast<uint32_t>(properties.bytesPerBlock), 4u);

  uint32_t dataSize = 0;
  for (uint32_t i = 0; i < numMipLevels; ++i) {
    const uint32_t mipLevel = numMipLevels - i - 1;
    const uint32_t rangeBytes =
        static_cast<uint32_t>(properties.getBytesPerRange(range.atMipLevel(mipLevel)));

    dataSize = align(dataSize + rangeBytes, mipLevelAlignment);
  }

  return dataSize;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void putDfd(std::vector<uint8_t>& buffer, uint32_t vkFormat, uint32_t numMipLevels) {
  const uint32_t dfdMetadataOffset = kHeaderSize + numMipLevels * kMipmapMetadataSize;

  ASSERT_TRUE(vkFormat == VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG ||
              vkFormat == VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK ||
              vkFormat == VK_FORMAT_R8G8B8A8_UNORM);

  const auto format = vulkan::util::vkTextureFormatToTextureFormat(static_cast<int32_t>(vkFormat));
  const auto properties = igl::TextureFormatProperties::fromTextureFormat(format);

  const uint16_t descriptorType = 0;
  const uint16_t vendorId = 0;
  const uint16_t descriptorBlockSize = vkFormat == VK_FORMAT_R8G8B8A8_UNORM ? 88 : 40;
  const uint16_t version = 2;
  const uint8_t flags = KHR_DF_FLAG_ALPHA_STRAIGHT;
  const uint8_t transferFunction = properties.isSRGB() ? KHR_DF_TRANSFER_SRGB
                                                       : KHR_DF_TRANSFER_LINEAR;
  const uint8_t colorPrimaries = KHR_DF_PRIMARIES_BT709;
  const uint8_t colorModel =
      vkFormat == VK_FORMAT_R8G8B8A8_UNORM
          ? KHR_DF_MODEL_RGBSDA
          : (vkFormat == VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG ? KHR_DF_MODEL_PVRTC
                                                               : KHR_DF_MODEL_ETC1);
  const uint8_t texelBlockDimension3 = 0;
  const uint8_t texelBlockDimension2 = properties.blockDepth - 1;
  const uint8_t texelBlockDimension1 = properties.blockHeight - 1;
  const uint8_t texelBlockDimension0 = properties.blockWidth - 1;
  const uint32_t bytesPlane3210 = vkFormat == VK_FORMAT_R8G8B8A8_UNORM ? 4u : 8u;
  const uint32_t bytesPlane7654 = 0;

  const uint32_t dfdMetadataSize = descriptorBlockSize + 4u;

  put(buffer, kOffsetDfdByteOffset, dfdMetadataOffset);
  put(buffer, kOffsetDfdByteLength, dfdMetadataSize);

  put(buffer, dfdMetadataOffset, dfdMetadataSize); // Total length

  put(buffer, dfdMetadataOffset + 4u, vendorId);
  put(buffer, dfdMetadataOffset + 6u, descriptorType);
  put(buffer, dfdMetadataOffset + 8u, version);
  put(buffer, dfdMetadataOffset + 10u, descriptorBlockSize);
  put(buffer, dfdMetadataOffset + 12u, colorModel);
  put(buffer, dfdMetadataOffset + 13u, colorPrimaries);
  put(buffer, dfdMetadataOffset + 14u, transferFunction);
  put(buffer, dfdMetadataOffset + 15u, flags);
  put(buffer, dfdMetadataOffset + 16u, texelBlockDimension0);
  put(buffer, dfdMetadataOffset + 17u, texelBlockDimension1);
  put(buffer, dfdMetadataOffset + 18u, texelBlockDimension2);
  put(buffer, dfdMetadataOffset + 19u, texelBlockDimension3);
  put(buffer, dfdMetadataOffset + 20u, bytesPlane3210);
  put(buffer, dfdMetadataOffset + 24u, bytesPlane7654);

  if (vkFormat == VK_FORMAT_R8G8B8A8_UNORM) {
    constexpr std::array<uint8_t, 4> kChannelFlags = {
        KHR_DF_CHANNEL_RGBSDA_RED,
        KHR_DF_CHANNEL_RGBSDA_GREEN,
        KHR_DF_CHANNEL_RGBSDA_BLUE,
        KHR_DF_CHANNEL_RGBSDA_ALPHA | KHR_DF_SAMPLE_DATATYPE_LINEAR,
    };
    uint32_t offset = dfdMetadataOffset + 28u;
    for (int i = 0; i < 4; ++i) {
      const uint8_t channelFlags = kChannelFlags[i];
      const uint8_t bitLength = 7;
      const uint16_t bitOffset = 8 * i;
      const uint8_t samplePosition3 = 0;
      const uint8_t samplePosition2 = 0;
      const uint8_t samplePosition1 = 0;
      const uint8_t samplePosition0 = 0;
      const uint32_t sampleLower = 0;
      const uint32_t sampleUpper = std::numeric_limits<uint8_t>::max();
      put(buffer, offset + 0u, bitOffset);
      put(buffer, offset + 2u, bitLength);
      put(buffer, offset + 3u, channelFlags);
      put(buffer, offset + 4u, samplePosition0);
      put(buffer, offset + 5u, samplePosition1);
      put(buffer, offset + 6u, samplePosition2);
      put(buffer, offset + 7u, samplePosition3);
      put(buffer, offset + 8u, sampleLower);
      put(buffer, offset + 12u, sampleUpper);
      offset += 16u;
    }
  } else {
    const uint8_t channelFlags = 0;
    const uint8_t bitLength = 63;
    const uint16_t bitOffset = 0;
    const uint8_t samplePosition3 = 0;
    const uint8_t samplePosition2 = 0;
    const uint8_t samplePosition1 = 0;
    const uint8_t samplePosition0 = 0;
    const uint32_t sampleLower = 0;
    const uint32_t sampleUpper = std::numeric_limits<uint32_t>::max();
    put(buffer, dfdMetadataOffset + 28u, bitOffset);
    put(buffer, dfdMetadataOffset + 30u, bitLength);
    put(buffer, dfdMetadataOffset + 31u, channelFlags);
    put(buffer, dfdMetadataOffset + 32u, samplePosition0);
    put(buffer, dfdMetadataOffset + 33u, samplePosition1);
    put(buffer, dfdMetadataOffset + 34u, samplePosition2);
    put(buffer, dfdMetadataOffset + 35u, samplePosition3);
    put(buffer, dfdMetadataOffset + 36u, sampleLower);
    put(buffer, dfdMetadataOffset + 40u, sampleUpper);
  }
}

void putMipLevel(std::vector<uint8_t>& buffer,
                 uint32_t vkFormat,
                 uint32_t mipLevel,
                 uint32_t imageSize) {
  const auto* header = reinterpret_cast<const iglu::textureloader::ktx2::Header*>(buffer.data());
  const auto format =
      vulkan::util::vkTextureFormatToTextureFormat(static_cast<int32_t>(header->vkFormat));
  const auto properties = igl::TextureFormatProperties::fromTextureFormat(format);

  const auto range = TextureRangeDesc::new2D(
      0, 0, std::max(header->pixelWidth, 1u), std::max(header->pixelHeight, 1u));

  const uint32_t maxMipLevels = TextureDesc::calcNumMipLevels(range.width, range.height);
  const uint32_t levelCount = std::min(maxMipLevels, header->levelCount);

  const uint32_t mipLevelAlignment = std::lcm(static_cast<uint32_t>(properties.bytesPerBlock), 4u);
  const uint32_t mipmapMetadataLength = levelCount * kMipmapMetadataSize;

  const uint32_t metadataLength =
      iglu::textureloader::ktx2::kHeaderLength + mipmapMetadataLength + header->kvdByteLength +
      (vkFormat == VK_FORMAT_R8G8B8A8_UNORM ? kDfdUncompressedMetadataSize
                                            : kDfdCompressedMetadataSize);

  std::vector<uint32_t> mipmapOffsets(levelCount);
  uint32_t mipmapOffset = align(metadataLength, mipLevelAlignment);

  for (size_t i = 0; i < levelCount; ++i) {
    const uint32_t workingLevel = levelCount - i - 1;
    mipmapOffsets[workingLevel] = mipmapOffset;
    if (workingLevel == mipLevel) {
      break;
    }

    const uint32_t rangeBytes =
        static_cast<uint32_t>(properties.getBytesPerRange(range.atMipLevel(workingLevel)));

    mipmapOffset = align(mipmapOffset + rangeBytes, mipLevelAlignment);
  }

  const uint32_t mipmapMetadataOffset = kHeaderSize + mipLevel * kMipmapMetadataSize;
  put(buffer, mipmapMetadataOffset, static_cast<uint64_t>(mipmapOffsets[mipLevel]));
  put(buffer, mipmapMetadataOffset + 8u, static_cast<uint64_t>(imageSize));
  put(buffer, mipmapMetadataOffset + 16u, static_cast<uint64_t>(imageSize));
}

void populateMinimalValidFile(std::vector<uint8_t>& buffer,
                              uint32_t vkFormat,
                              uint32_t width,
                              uint32_t height,
                              uint32_t numMipLevels,
                              uint32_t bytesOfKeyValueData,
                              uint32_t imageSize,
                              bool forceDfdAfterMipLevel1 = false) {
  // HEADER
  // Zero-out the whole buffer, since there might be garbage in it.
  std::memset(buffer.data(), 0x00, buffer.size());

  const uint32_t mipmapMetadataLength = numMipLevels * kMipmapMetadataSize;

  // Put the default values in
  const char fixedTag[] = {'\xAB', 'K', 'T', 'X', ' ', '2', '0', '\xBB', '\r', '\n', '\x1A', '\n'};
  std::memcpy(buffer.data(), &fixedTag, sizeof(fixedTag));
  put(buffer, kOffsetVkFormat, vkFormat);
  put(buffer, kOffsetTypeSize, 1);
  put(buffer, kOffsetFaceCount, 1);

  put(buffer, kOffsetWidth, width);
  put(buffer, kOffsetHeight, height);
  put(buffer, kOffsetLevelCount, numMipLevels);
  put(buffer,
      kOffsetKvdByteOffset,
      bytesOfKeyValueData == 0 ? 0u : kHeaderSize + mipmapMetadataLength);
  put(buffer, kOffsetKvdByteLength, bytesOfKeyValueData);

  putMipLevel(buffer, vkFormat, 0u, imageSize);

  putDfd(buffer, vkFormat, forceDfdAfterMipLevel1 ? 1u : numMipLevels);
}

constexpr uint32_t kSupercompressionSchemeZlib = 3u; // KTX_SS_ZLIB

// Builds a minimal KTX2 buffer with a single ZLIB-supercompressed level whose compressed data
// begins with the given two-byte zlib header (cmf, flg). Every check preceding the zlib-header
// conformance guard in validate() is satisfied, so tests can exercise that guard in isolation.
// A byteLength < 2 lets the "too short for a zlib header" branch be exercised.
std::vector<uint8_t> makeZlibHeaderTestFile(uint8_t cmf,
                                            uint8_t flg,
                                            uint32_t levelByteLength = 16u) {
  constexpr uint32_t width = 4u;
  constexpr uint32_t height = 4u;
  constexpr uint32_t levelDataOffset = kHeaderSize + kMipmapMetadataSize;
  std::vector<uint8_t> buffer(static_cast<size_t>(levelDataOffset) + levelByteLength, 0u);

  const char fixedTag[] = {'\xAB', 'K', 'T', 'X', ' ', '2', '0', '\xBB', '\r', '\n', '\x1A', '\n'};
  std::memcpy(buffer.data(), &fixedTag, sizeof(fixedTag));
  put(buffer, kOffsetVkFormat, VK_FORMAT_R8G8B8A8_UNORM);
  put(buffer, kOffsetTypeSize, 1u);
  put(buffer, kOffsetWidth, width);
  put(buffer, kOffsetHeight, height);
  put(buffer, kOffsetFaceCount, 1u);
  put(buffer, kOffsetLevelCount, 1u);
  put(buffer, kOffsetSupercompressionScheme, kSupercompressionSchemeZlib);

  // Single level index entry: byteOffset, (compressed) byteLength, uncompressedByteLength. A 1:1
  // ratio keeps the expansion-ratio guard satisfied so the zlib-header guard is what's exercised.
  put(buffer, kHeaderSize, static_cast<uint64_t>(levelDataOffset));
  put(buffer, kHeaderSize + 8u, static_cast<uint64_t>(levelByteLength));
  put(buffer, kHeaderSize + 16u, static_cast<uint64_t>(levelByteLength));

  // Compressed level data: the two-byte zlib header under test, remainder zero-filled.
  if (levelByteLength >= 1u) {
    put(buffer, levelDataOffset, cmf);
  }
  if (levelByteLength >= 2u) {
    put(buffer, levelDataOffset + 1u, flg);
  }

  return buffer;
}

} // namespace

class Ktx2TextureLoaderTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
  }

  void TearDown() override {}

 protected:
  iglu::textureloader::ktx2::TextureLoaderFactory factory_;
};

TEST_F(Ktx2TextureLoaderTest, EmptyBufferFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx2TextureLoaderTest, MinimumValidHeaderSucceeds) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 512u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(ret.isOk()) << ret.message;
}

TEST_F(Ktx2TextureLoaderTest, HeaderWithMipLevelsSucceeds) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 5u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t imageSize = 512u; // For first mip level
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  // Fill the other mip levels
  putMipLevel(buffer, vkFormat, 1u, 128u);
  putMipLevel(buffer, vkFormat, 2u, 32u);
  putMipLevel(buffer, vkFormat, 3u, 32u);
  putMipLevel(buffer, vkFormat, 4u, 32u);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(ret.isOk()) << ret.message;
}

TEST_F(Ktx2TextureLoaderTest, ValidHeaderWithExtraDataSucceeds) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 512u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize + 1u);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(ret.isOk()) << ret.message;
}

TEST_F(Ktx2TextureLoaderTest, InsufficientDataFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 512u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize - 1u);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx2TextureLoaderTest, InsufficientDataWithMipLevelsFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 6u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t imageSize = 512; // For first mip level
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize - 1u);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  // Fill the other mip levels
  putMipLevel(buffer, vkFormat, 1u, 128u);
  putMipLevel(buffer, vkFormat, 2u, 32u);
  putMipLevel(buffer, vkFormat, 3u, 32u);
  putMipLevel(buffer, vkFormat, 4u, 32u);
  putMipLevel(buffer, vkFormat, 5u, 32u);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx2TextureLoaderTest, ValidHeaderWithInvalidImageSizeFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 4096u;
  const uint32_t vkFormat = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx2TextureLoaderTest, InvalidHeaderWithExcessiveImageSizeFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 4294967290u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx2TextureLoaderTest, InvalidHeaderWithExcessiveMipLevelsFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 4294967290u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 512u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, 6u, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, 6u);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize);
  populateMinimalValidFile(buffer,
                           vkFormat,
                           width,
                           height,
                           numMipLevels,
                           bytesOfKeyValueData,
                           imageSize,
                           true) /* forceDfdAfterMipLevel1*/;

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx2TextureLoaderTest, InvalidHeaderWithExcessiveKeyValueDataFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 4294967290u;
  const uint32_t imageSize = 512u;
  const uint32_t vkFormat = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx2TextureLoaderTest, MinimumValidHeader1x1Rgba8Succeeds) {
  const uint32_t width = 1u;
  const uint32_t height = 1u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 4u;
  const uint32_t vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
  const uint32_t totalHeaderSize = getTotalHeaderSize(vkFormat, numMipLevels, bytesOfKeyValueData);
  const uint32_t totalDataSize = getTotalDataSize(vkFormat, width, height, numMipLevels);

  auto buffer = getBuffer(totalHeaderSize + totalDataSize);
  populateMinimalValidFile(
      buffer, vkFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(ret.isOk()) << ret.message;
}

TEST_F(Ktx2TextureLoaderTest, ZlibConformantHeaderNotRejectedByHeaderGuard) {
  // A conformant zlib header (CM=8, CINFO=7, no preset dictionary, valid check bits) must pass the
  // zlib-header conformance guard. The hand-built file is not otherwise complete, so tryCreate
  // still fails downstream -- but the failure must not come from the zlib-header guard.
  auto buffer = makeZlibHeaderTestFile(0x78u, 0x01u);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  // The file is intentionally incomplete, so tryCreate() must still fail downstream...
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
  // ...but the failure must not come from the zlib-header conformance guard.
  EXPECT_EQ(std::string(ret.message).find("zlib header"), std::string::npos) << ret.message;
}

TEST_F(Ktx2TextureLoaderTest, ZlibHeaderInvalidWindowSizeFails) {
  // CMF 0xF8 -> CINFO=15, a 32MB window RFC 1950 forbids (CINFO must be <= 7). Check bits are valid
  // and CM=8, so this isolates the window-size rejection. This is the crash input's header.
  auto buffer = makeZlibHeaderTestFile(0xF8u, 0x00u);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_EQ(std::string(ret.message),
            "ZLIB supercompressed level has a non-conformant zlib header.");
}

TEST_F(Ktx2TextureLoaderTest, ZlibHeaderInvalidCheckBitsFails) {
  // CMF/FLG 0x7800 % 31 != 0, so FCHECK fails while CM=8, CINFO=7, no preset dictionary.
  auto buffer = makeZlibHeaderTestFile(0x78u, 0x00u);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_EQ(std::string(ret.message),
            "ZLIB supercompressed level has a non-conformant zlib header.");
}

TEST_F(Ktx2TextureLoaderTest, ZlibHeaderPresetDictionaryFails) {
  // FLG 0x20 sets FDICT (preset dictionary) while keeping CM=8, CINFO=7 and valid check bits.
  auto buffer = makeZlibHeaderTestFile(0x78u, 0x20u);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_EQ(std::string(ret.message),
            "ZLIB supercompressed level has a non-conformant zlib header.");
}

TEST_F(Ktx2TextureLoaderTest, ZlibHeaderInvalidCompressionMethodFails) {
  // CMF 0x77 -> CM=7 (not DEFLATE); CINFO=7, no preset dictionary and valid check bits.
  auto buffer = makeZlibHeaderTestFile(0x77u, 0x09u);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_EQ(std::string(ret.message),
            "ZLIB supercompressed level has a non-conformant zlib header.");
}

TEST_F(Ktx2TextureLoaderTest, ZlibLevelTooShortForHeaderFails) {
  // A single-byte level cannot hold the two-byte zlib header.
  auto buffer = makeZlibHeaderTestFile(0x78u, 0x01u, /*levelByteLength=*/1u);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_EQ(std::string(ret.message), "ZLIB supercompressed level is too short for a zlib header.");
}

} // namespace igl::tests::ktx2
