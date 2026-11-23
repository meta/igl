/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <IGLU/texture_loader/ktx1/Header.h>
#include <IGLU/texture_loader/ktx1/TextureLoaderFactory.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include <igl/opengl/util/TextureFormat.h>

namespace igl::tests::ktx1 {

namespace {

std::vector<uint8_t> getBuffer(uint32_t capacity) {
  std::vector<uint8_t> buffer(static_cast<size_t>(capacity));
  return buffer;
}

void put(std::vector<uint8_t>& buffer, uint32_t offset, uint32_t data) {
  if (buffer.size() < static_cast<size_t>(offset) + sizeof(data)) {
    throw std::runtime_error("Overflow when storing a word");
  }
  std::memcpy(buffer.data() + offset, &data, sizeof(data));
}

constexpr uint32_t kHeaderSize = 64u;
constexpr uint32_t kOffsetEndianness = 12u;
constexpr uint32_t kOffsetTypeSize = 20u;
constexpr uint32_t kOffsetGlFormat = 28u;
constexpr uint32_t kOffsetWidth = 36u;
constexpr uint32_t kOffsetHeight = 40u;
constexpr uint32_t kOffsetNumberOfFaces = 52u;
constexpr uint32_t kOffsetNumberOfMipmapLevels = 56u;
constexpr uint32_t kOffsetBytesOfKeyValueData = 60u;
constexpr uint32_t kOffsetImages = 64u;

// NOLINTBEGIN(readability-identifier-naming)
constexpr uint32_t GL_RGBA8 = 0x8058;
constexpr uint32_t GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG = 0x8C03;
constexpr uint32_t GL_COMPRESSED_RGB8_ETC2 = 0x9274;
// NOLINTEND(readability-identifier-naming)

void putMipLevel(std::vector<uint8_t>& buffer, uint32_t mipLevel, uint32_t imageSize) {
  const auto* header = reinterpret_cast<const iglu::textureloader::ktx1::Header*>(buffer.data());

  const auto range = TextureRangeDesc::new2D(
      0, 0, std::max(header->pixelWidth, 1u), std::max(header->pixelHeight, 1u));

  const auto format = opengl::util::glTextureFormatToTextureFormat(
      header->glInternalFormat, header->glFormat, header->glType);
  const auto properties = igl::TextureFormatProperties::fromTextureFormat(format);

  uint32_t offset = kOffsetImages;
  for (uint32_t i = 0; i < mipLevel; ++i) {
    const uint32_t rangeBytes =
        static_cast<uint32_t>(properties.getBytesPerRange(range.atMipLevel(i)));
    offset += rangeBytes + 4u;
  }
  put(buffer, offset, imageSize);
}

void populateMinimalValidFile(std::vector<uint8_t>& buffer,
                              uint32_t glFormat,
                              uint32_t width,
                              uint32_t height,
                              uint32_t numMipLevels,
                              uint32_t bytesOfKeyValueData,
                              uint32_t imageSize) {
  // HEADER
  // Zero-out the whole buffer, since there might be garbage in it.
  std::memset(buffer.data(), 0x00, buffer.size());

  // Put the default values in
  const char fixedTag[] = {'\xAB', 'K', 'T', 'X', ' ', '1', '1', '\xBB', '\r', '\n', '\x1A', '\n'};
  std::memcpy(buffer.data(), &fixedTag, sizeof(fixedTag));
  put(buffer, kOffsetEndianness, 0x04030201);
  put(buffer, kOffsetTypeSize, 1);
  put(buffer, kOffsetNumberOfFaces, 1);

  put(buffer, kOffsetWidth, width);
  put(buffer, kOffsetHeight, height);
  put(buffer, kOffsetNumberOfMipmapLevels, numMipLevels);
  put(buffer, kOffsetGlFormat, glFormat);
  put(buffer, kOffsetBytesOfKeyValueData, bytesOfKeyValueData);

  // IMAGES
  putMipLevel(buffer, 0, imageSize);
}

std::optional<iglu::textureloader::DataReader> getReader(const std::vector<uint8_t>& buffer) {
  Result ret;
  auto maybeReader = iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), &ret);

  EXPECT_TRUE(ret.isOk()) << ret.message;

  return maybeReader;
}
} // namespace

class Ktx1TextureLoaderTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
  }

  void TearDown() override {}

 protected:
  iglu::textureloader::ktx1::TextureLoaderFactory factory_;
};

TEST_F(Ktx1TextureLoaderTest, EmptyBufferFails) {
  const uint32_t numMipLevels = 1u;
  const uint32_t imageSize = 512u;
  auto buffer = getBuffer(kHeaderSize + imageSize + 4u * numMipLevels /* for imageSize */);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx1TextureLoaderTest, MinimumValidHeaderSucceeds) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 512u;
  const uint32_t glFormat = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
  auto buffer = getBuffer(kHeaderSize + imageSize + 4u * numMipLevels /* for imageSize */);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(ret.isOk()) << ret.message;
}

TEST_F(Ktx1TextureLoaderTest, HeaderWithMipLevelsSucceeds) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 5u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t glFormat = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
  const uint32_t imageSize = 512u; // For first mip level
  // size:  512 + 128 + 32 + 32 + 32 + 32 - 64x32, 32x16, 16x8, 8x4, 4x2, 2x1, 1x1
  auto buffer = getBuffer(kHeaderSize + 512u + 128u + 32u + 32u + 32u + 32u + 4u * numMipLevels);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  // Fill the other mip levels
  putMipLevel(buffer, 1u, 128u);
  putMipLevel(buffer, 2u, 32u);
  putMipLevel(buffer, 3u, 32u);
  putMipLevel(buffer, 4u, 32u);

  Result ret;
  auto maybeReader = getReader(buffer);
  ASSERT_TRUE(maybeReader.has_value());
  auto loader = factory_.tryCreate(*maybeReader, &ret);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(ret.isOk()) << ret.message;
}

TEST_F(Ktx1TextureLoaderTest, ValidHeaderWithExtraDataSucceeds) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 512u;
  const uint32_t glFormat = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
  auto buffer = getBuffer(kHeaderSize + imageSize + 1u + 4u * numMipLevels);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto maybeReader = getReader(buffer);
  ASSERT_TRUE(maybeReader.has_value());
  auto loader = factory_.tryCreate(*maybeReader, &ret);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(ret.isOk()) << ret.message;
}

TEST_F(Ktx1TextureLoaderTest, InsufficientDataFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 512u;
  const uint32_t glFormat = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;

  auto buffer = getBuffer(kHeaderSize + imageSize + 4u * numMipLevels - 1u);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto maybeReader = getReader(buffer);
  ASSERT_TRUE(maybeReader.has_value());
  auto loader = factory_.tryCreate(*maybeReader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx1TextureLoaderTest, InsufficientDataWithMipLevelsFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 6u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t glFormat = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
  const uint32_t imageSize = 512; // For first mip level
  // size:  512 + 128 + 32 + 32 + 32 + 32 - 64x32, 32x16, 16x8, 8x4, 4x2, 2x1, 1x1
  auto buffer =
      getBuffer(kHeaderSize + 512u + 128u + 32u + 32u + 32u + 32u + 4u * numMipLevels - 1u);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  // Fill the other mip levels
  putMipLevel(buffer, 1u, 128u);
  putMipLevel(buffer, 2u, 32u);
  putMipLevel(buffer, 3u, 32u);
  putMipLevel(buffer, 4u, 32u);
  putMipLevel(buffer, 5u, 32u);

  Result ret;
  auto maybeReader = getReader(buffer);
  ASSERT_TRUE(maybeReader.has_value());
  auto loader = factory_.tryCreate(*maybeReader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx1TextureLoaderTest, ValidHeaderWithInvalidImageSizeFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 4096u;
  const uint32_t glFormat = GL_COMPRESSED_RGB8_ETC2;

  auto buffer = getBuffer(kHeaderSize + imageSize + 4u * numMipLevels);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto maybeReader = getReader(buffer);
  ASSERT_TRUE(maybeReader.has_value());
  auto loader = factory_.tryCreate(*maybeReader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx1TextureLoaderTest, InvalidHeaderWithExcessiveImageSizeFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 4294967290u;
  const uint32_t glFormat = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;

  auto buffer = getBuffer(kHeaderSize + 4u * numMipLevels);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto maybeReader = getReader(buffer);
  ASSERT_TRUE(maybeReader.has_value());
  auto loader = factory_.tryCreate(*maybeReader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx1TextureLoaderTest, InvalidHeaderWithExcessiveMipLevelsFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 4294967290u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 512u;
  const uint32_t glFormat = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;

  auto buffer = getBuffer(kHeaderSize + imageSize);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto maybeReader = getReader(buffer);
  ASSERT_TRUE(maybeReader.has_value());
  auto loader = factory_.tryCreate(*maybeReader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx1TextureLoaderTest, InvalidHeaderWithExcessiveKeyValueDataFails) {
  const uint32_t width = 64u;
  const uint32_t height = 32u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 4294967290u;
  const uint32_t imageSize = 512u;
  const uint32_t glFormat = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;

  auto buffer = getBuffer(kHeaderSize + imageSize);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto maybeReader = getReader(buffer);
  ASSERT_TRUE(maybeReader.has_value());
  auto loader = factory_.tryCreate(*maybeReader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(Ktx1TextureLoaderTest, MinimumValidHeader1x1Rgba8Succeeds) {
  const uint32_t width = 1u;
  const uint32_t height = 1u;
  const uint32_t numMipLevels = 1u;
  const uint32_t bytesOfKeyValueData = 0u;
  const uint32_t imageSize = 4u;
  const uint32_t glFormat = GL_RGBA8;
  auto buffer = getBuffer(kHeaderSize + imageSize + 4u * numMipLevels /* for imageSize */);
  populateMinimalValidFile(
      buffer, glFormat, width, height, numMipLevels, bytesOfKeyValueData, imageSize);

  Result ret;
  auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(ret.isOk()) << ret.message;
}
} // namespace igl::tests::ktx1
