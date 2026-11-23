/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if !defined(IGL_CMAKE_BUILD)

#include <gtest/gtest.h>

#include <IGLU/texture_loader/webp/Header.h>
#include <IGLU/texture_loader/webp/TextureLoaderFactory.h>
#include <cstring>
#include <vector>
#include <webp/encode.h>

namespace igl::tests::webp {

namespace {

// Create a valid WebP image with solid red pixels using libwebp encoder
std::vector<uint8_t> createWebPImage(size_t width, size_t height) {
  std::vector<uint8_t> rgba(width * height * 4);
  for (size_t i = 0; i < width * height; ++i) {
    rgba[i * 4 + 0] = 255; // R
    rgba[i * 4 + 1] = 0; // G
    rgba[i * 4 + 2] = 0; // B
    rgba[i * 4 + 3] = 255; // A
  }

  uint8_t* output = nullptr;
  const size_t outputSize = WebPEncodeLosslessRGBA(rgba.data(),
                                                   static_cast<int>(width),
                                                   static_cast<int>(height),
                                                   static_cast<int>(width) * 4,
                                                   &output);

  std::vector<uint8_t> result;
  if (outputSize > 0 && output != nullptr) {
    result.assign(output, output + outputSize);
    WebPFree(output);
  }
  return result;
}

constexpr std::array<uint8_t, 12> kPngHeader{
    {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D}};

} // namespace

class WebPTextureLoaderTest : public ::testing::Test {
 protected:
  iglu::textureloader::webp::TextureLoaderFactory factory_;
};

TEST_F(WebPTextureLoaderTest, EmptyBufferFails) {
  std::vector<uint8_t> buffer(iglu::textureloader::webp::kHeaderLength, 0);

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(WebPTextureLoaderTest, PngDataFails) {
  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      kPngHeader.data(), static_cast<uint32_t>(kPngHeader.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(WebPTextureLoaderTest, ValidWebP1x1Succeeds) {
  const auto buffer = createWebPImage(1, 1);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  ASSERT_NE(loader, nullptr);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(loader->descriptor().width, 1u);
  EXPECT_EQ(loader->descriptor().height, 1u);
  EXPECT_EQ(loader->descriptor().format, igl::TextureFormat::RGBA_UNorm8);
}

TEST_F(WebPTextureLoaderTest, ValidWebP2x2LoadProducesCorrectSize) {
  const auto buffer = createWebPImage(2, 2);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  ASSERT_NE(loader, nullptr);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  const auto data = loader->load(&ret);
  ASSERT_NE(data, nullptr);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  EXPECT_EQ(data->size(), 2u * 2u * 4u);
}

TEST_F(WebPTextureLoaderTest, ValidWebPPixelDataIsCorrect) {
  const auto buffer = createWebPImage(1, 1);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  ASSERT_NE(loader, nullptr);

  const auto data = loader->load(&ret);
  ASSERT_NE(data, nullptr);
  ASSERT_GE(data->size(), 4u);

  EXPECT_EQ(data->data()[0], 255); // R
  EXPECT_EQ(data->data()[1], 0); // G
  EXPECT_EQ(data->data()[2], 0); // B
  EXPECT_EQ(data->data()[3], 255); // A
}

TEST_F(WebPTextureLoaderTest, CorruptWebPPayloadFails) {
  const std::vector<uint8_t> corrupt = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P'};

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      corrupt.data(), static_cast<uint32_t>(corrupt.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  EXPECT_EQ(loader, nullptr);
  EXPECT_FALSE(ret.isOk());
}

TEST_F(WebPTextureLoaderTest, MinHeaderLength) {
  EXPECT_EQ(factory_.minHeaderLength(), iglu::textureloader::webp::kHeaderLength);
  EXPECT_EQ(factory_.minHeaderLength(), 12u);
}

TEST_F(WebPTextureLoaderTest, CanCreateWithValidWebP) {
  const auto buffer = createWebPImage(1, 1);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  EXPECT_TRUE(factory_.canCreate(buffer.data(), static_cast<uint32_t>(buffer.size()), &ret));
  EXPECT_TRUE(ret.isOk());
}

TEST_F(WebPTextureLoaderTest, CanCreateWithInvalidDataReturnsFalse) {
  Result ret;
  EXPECT_FALSE(
      factory_.canCreate(kPngHeader.data(), static_cast<uint32_t>(kPngHeader.size()), &ret));
  EXPECT_FALSE(ret.isOk());
}

TEST_F(WebPTextureLoaderTest, CanUploadSourceDataReturnsFalse) {
  const auto buffer = createWebPImage(1, 1);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  ASSERT_NE(loader, nullptr);

  EXPECT_FALSE(loader->canUploadSourceData());
}

TEST_F(WebPTextureLoaderTest, ShouldGenerateMipmaps) {
  const auto buffer = createWebPImage(4, 4);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  ASSERT_NE(loader, nullptr);

  // 4x4 image should have numMipLevels > 1
  EXPECT_TRUE(loader->shouldGenerateMipmaps());
}

TEST_F(WebPTextureLoaderTest, ShouldNotGenerateMipmapsFor1x1) {
  const auto buffer = createWebPImage(1, 1);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  ASSERT_NE(loader, nullptr);

  // 1x1 image has numMipLevels == 1
  EXPECT_FALSE(loader->shouldGenerateMipmaps());
}

TEST_F(WebPTextureLoaderTest, PreferredFormatIsRespected) {
  const auto buffer = createWebPImage(1, 1);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, TextureFormat::RGBA_SRGB, &ret);
  ASSERT_NE(loader, nullptr);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  EXPECT_EQ(loader->descriptor().format, igl::TextureFormat::RGBA_SRGB);
}

TEST_F(WebPTextureLoaderTest, DescriptorFieldsAreCorrect) {
  const auto buffer = createWebPImage(8, 4);
  ASSERT_FALSE(buffer.empty());

  Result ret;
  const auto reader = *iglu::textureloader::DataReader::tryCreate(
      buffer.data(), static_cast<uint32_t>(buffer.size()), nullptr);
  const auto loader = factory_.tryCreate(reader, &ret);
  ASSERT_NE(loader, nullptr);

  const auto& desc = loader->descriptor();
  EXPECT_EQ(desc.width, 8u);
  EXPECT_EQ(desc.height, 4u);
  EXPECT_EQ(desc.depth, 1u);
  EXPECT_EQ(desc.numLayers, 1u);
  EXPECT_EQ(desc.type, igl::TextureType::TwoD);
  EXPECT_EQ(desc.format, igl::TextureFormat::RGBA_UNorm8);
}

TEST_F(WebPTextureLoaderTest, HeaderTagValidation) {
  iglu::textureloader::webp::Header validHeader{};
  std::memcpy(validHeader.riff.data(), "RIFF", 4);
  validHeader.fileSize = 0;
  std::memcpy(validHeader.webp.data(), "WEBP", 4);
  EXPECT_TRUE(validHeader.tagIsValid());

  iglu::textureloader::webp::Header invalidRiff{};
  std::memcpy(invalidRiff.riff.data(), "XXXX", 4);
  invalidRiff.fileSize = 0;
  std::memcpy(invalidRiff.webp.data(), "WEBP", 4);
  EXPECT_FALSE(invalidRiff.tagIsValid());

  iglu::textureloader::webp::Header invalidWebp{};
  std::memcpy(invalidWebp.riff.data(), "RIFF", 4);
  invalidWebp.fileSize = 0;
  std::memcpy(invalidWebp.webp.data(), "XXXX", 4);
  EXPECT_FALSE(invalidWebp.tagIsValid());
}

} // namespace igl::tests::webp

#endif // !defined(IGL_CMAKE_BUILD)
