/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <IGLU/bitmap/BitmapWriter.h>

#include "../../util/TestDevice.h"

#include <cstring>
#include <sstream>

namespace {
// Dumped from a bmp file that was manually validated as being a checkerboard pattern
const uint8_t kExpectedData[] = {
    0x42, 0x4d, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x28,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x18, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
    0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00};
} // namespace

namespace igl::tests::bitmap_writer {

class BitmapWriterTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
    device_ = util::createTestDevice();
    ASSERT_TRUE(device_ != nullptr);

    Result result;
    texDesc_ = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                  kTexWidth,
                                  kTexWidth,
                                  TextureDesc::TextureUsageBits::Sampled |
                                      TextureDesc::TextureUsageBits::Attachment);
    texture_ = device_->createTexture(texDesc_, &result);
    ASSERT_TRUE(result.isOk());

    // Initialize a red checkerboard pattern
    std::vector<uint8_t> data;
    data.reserve(kTexWidth * kTexWidth * 4);
    for (int i = 0; i < kTexWidth; i++) {
      for (int j = 0; j < kTexWidth; j++) {
        if (i % 2 != j % 2) {
          data.push_back(255);
        } else {
          data.push_back(0);
        }
        data.push_back(0);
        data.push_back(0);
        data.push_back(255);
      }
    }

    // Initialize texture data
    const auto range = TextureRangeDesc::new2D(0, 0, kTexWidth, kTexWidth);
    texture_->upload(range, data.data());
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> device_;
  std::shared_ptr<ITexture> texture_;
  TextureDesc texDesc_;
  constexpr static size_t kTexWidth = 4;
};

TEST_F(BitmapWriterTest, WriteFile) {
  std::stringstream ss;
  igl::iglu::writeBitmap(ss, texture_, *device_);
  std::string s = ss.str();
  const uint8_t* fileData = reinterpret_cast<const uint8_t*>(s.c_str());
  ASSERT_NE(fileData, nullptr);
  ASSERT_EQ(s.size(), sizeof(kExpectedData));
  ASSERT_EQ(std::memcmp(fileData, kExpectedData, sizeof(kExpectedData)), 0);
}

// Only the formats that have a defined RGB byte layout in the writer are
// reported as supported; everything else (single-channel, float, compressed,
// or invalid) must be rejected.
TEST(BitmapWriterFormatTest, IsSupportedBitmapTextureFormat) {
  EXPECT_TRUE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::RGBA_UNorm8));
  EXPECT_TRUE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::RGBX_UNorm8));
  EXPECT_TRUE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::RGBA_SRGB));
  EXPECT_TRUE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::BGRA_UNorm8));
  EXPECT_TRUE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::BGRA_SRGB));

  EXPECT_FALSE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::Invalid));
  EXPECT_FALSE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::R_UNorm8));
  EXPECT_FALSE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::BGRA_UNorm8_Rev));
  EXPECT_FALSE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::RGB_F16));
  EXPECT_FALSE(igl::iglu::isSupportedBitmapTextureFormat(TextureFormat::RGBA_F32));
}

// The raw-byte overload prepends a 54-byte BMP header and copies the supplied
// pixel data verbatim (no swizzling) to the stream.
TEST(BitmapWriterRawTest, WriteBitmapRawBytes) {
  // Three BGR pixels forming a 3x1 image.
  const uint8_t pixels[] = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90};
  constexpr uint32_t kWidth = 3;
  constexpr uint32_t kHeight = 1;
  constexpr uint32_t kImageSize = kWidth * kHeight * 3;
  constexpr size_t kHeaderSize = 54; // sizeof(BMPHeader)

  std::stringstream ss;
  igl::iglu::writeBitmap(ss, pixels, kWidth, kHeight);
  const std::string s = ss.str();

  ASSERT_EQ(s.size(), kHeaderSize + kImageSize);

  const uint8_t* out = reinterpret_cast<const uint8_t*>(s.data());

  // "BM" signature.
  EXPECT_EQ(out[0], 'B');
  EXPECT_EQ(out[1], 'M');

  // Reads a little-endian uint32 at a byte offset into the BMP header.
  const auto readU32 = [out](size_t offset) {
    return static_cast<uint32_t>(out[offset]) | (static_cast<uint32_t>(out[offset + 1]) << 8) |
           (static_cast<uint32_t>(out[offset + 2]) << 16) |
           (static_cast<uint32_t>(out[offset + 3]) << 24);
  };

  EXPECT_EQ(readU32(2), kHeaderSize + kImageSize); // fileSize
  EXPECT_EQ(readU32(10), kHeaderSize); // dataOffset to pixel array
  EXPECT_EQ(readU32(14), 40u); // DIB header size
  EXPECT_EQ(readU32(18), kWidth); // imageWidth
  EXPECT_EQ(readU32(22), kHeight); // imageHeight
  EXPECT_EQ(out[28], 24); // bitsPerPixel
  EXPECT_EQ(readU32(34), kImageSize); // imageSizeBytes

  // Pixel payload is copied verbatim, unmodified.
  EXPECT_EQ(std::memcmp(out + kHeaderSize, pixels, kImageSize), 0);
}

} // namespace igl::tests::bitmap_writer
