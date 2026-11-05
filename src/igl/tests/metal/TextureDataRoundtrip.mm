/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"
#include "../util/TestDevice.h"

#include <cstring>
#include <vector>
#include <igl/IGL.h>
#include <igl/metal/CommandBuffer.h>
#include <igl/metal/Device.h>
#include <igl/metal/Texture.h>

namespace igl::tests {

#define ROUNDTRIP_TEX_WIDTH 4
#define ROUNDTRIP_TEX_HEIGHT 4

//
// MetalTextureDataRoundtripTest
//
// This test covers uploading texture data and reading it back on Metal.
// Tests RGBA upload/readback, mip level uploads, and 3D textures.
//
class MetalTextureDataRoundtripTest : public ::testing::Test {
 public:
  MetalTextureDataRoundtripTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(device_, cmdQueue_);
    ASSERT_NE(device_, nullptr);
  }

  void TearDown() override {}

 protected:
  void expectPrivateTexels(const std::shared_ptr<ITexture>& texture,
                           const TextureRangeDesc& range,
                           const std::vector<uint32_t>& expected) {
    @autoreleasepool {
      const id<MTLDevice> nativeDevice = static_cast<metal::Device&>(*device_).get();
      const id<MTLTexture> nativeTexture = static_cast<metal::Texture&>(*texture).get();
      ASSERT_NE(nativeTexture, nil);
      ASSERT_EQ(nativeTexture.storageMode, MTLStorageModePrivate);
      const auto& properties = texture->getProperties();
      const size_t length = properties.getBytesPerRange(range);
      ASSERT_EQ(length, expected.size() * sizeof(uint32_t));
      id<MTLBuffer> readback = [nativeDevice newBufferWithLength:length
                                                         options:MTLResourceStorageModeShared];
      readback.label = @"Private texture upload test readback";
      ASSERT_NE(readback, nil);
      const id<MTLCommandQueue> queue = [nativeDevice newCommandQueue];
      ASSERT_NE(queue, nil);
      const id<MTLCommandBuffer> command = [queue commandBuffer];
      ASSERT_NE(command, nil);
      const id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
      ASSERT_NE(encoder, nil);
      const size_t rowBytes = properties.getBytesPerRow(range);
      const size_t imageBytes = range.depth > 1
                                    ? properties.getBytesPerLayer(
                                          TextureRangeDesc::new2D(0, 0, range.width, range.height))
                                    : 0;
      [encoder copyFromTexture:nativeTexture
                       sourceSlice:metal::Texture::getMetalSlice(
                                       texture->getType(), range.face, range.layer)
                       sourceLevel:range.mipLevel
                      sourceOrigin:MTLOriginMake(range.x, range.y, range.z)
                        sourceSize:MTLSizeMake(range.width, range.height, range.depth)
                          toBuffer:readback
                 destinationOffset:0
            destinationBytesPerRow:rowBytes
          destinationBytesPerImage:imageBytes];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      ASSERT_EQ(command.status, MTLCommandBufferStatusCompleted);
      ASSERT_NE(readback.contents, nullptr);
      std::vector<uint32_t> actual(expected.size());
      std::memcpy(actual.data(), readback.contents, length);
      EXPECT_EQ(actual, expected);
    }
  }

  std::shared_ptr<IDevice> device_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

//
// UploadAndReadbackRGBA
//
// Upload RGBA data to a texture, read it back via framebuffer copyBytes, and verify match.
//
TEST_F(MetalTextureDataRoundtripTest, UploadAndReadbackRGBA) {
  Result res;

  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           ROUNDTRIP_TEX_WIDTH,
                                           ROUNDTRIP_TEX_HEIGHT,
                                           TextureDesc::TextureUsageBits::Sampled |
                                               TextureDesc::TextureUsageBits::Attachment);
  auto texture = device_->createTexture(texDesc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(texture, nullptr);

  // Prepare test data: solid green
  const size_t numPixels = ROUNDTRIP_TEX_WIDTH * ROUNDTRIP_TEX_HEIGHT;
  std::vector<uint32_t> uploadData(numPixels, 0xFF00FF00); // RGBA: green with full alpha

  auto range = TextureRangeDesc::new2D(0, 0, ROUNDTRIP_TEX_WIDTH, ROUNDTRIP_TEX_HEIGHT);
  res = texture->upload(range, uploadData.data());
  ASSERT_TRUE(res.isOk()) << res.message;

  // Create framebuffer to read back
  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = texture}}};
  auto framebuffer = device_->createFramebuffer(fbDesc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;

  std::vector<uint32_t> readbackData(numPixels, 0);
  framebuffer->copyBytesColorAttachment(
      *cmdQueue_, 0, readbackData.data(), range, ROUNDTRIP_TEX_WIDTH * 4);

  for (size_t i = 0; i < numPixels; ++i) {
    ASSERT_EQ(readbackData[i], uploadData[i]) << "Mismatch at pixel " << i;
  }
}

//
// UploadToMipLevel
//
// Upload data to a specific mip level and verify the upload succeeds.
//
TEST_F(MetalTextureDataRoundtripTest, UploadToMipLevel) {
  Result res;

  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           ROUNDTRIP_TEX_WIDTH,
                                           ROUNDTRIP_TEX_HEIGHT,
                                           TextureDesc::TextureUsageBits::Sampled |
                                               TextureDesc::TextureUsageBits::Attachment);
  texDesc.numMipLevels = 2; // base (4x4) + mip 1 (2x2)

  auto texture = device_->createTexture(texDesc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(texture, nullptr);

  // Upload to mip level 0
  const size_t numPixels0 = ROUNDTRIP_TEX_WIDTH * ROUNDTRIP_TEX_HEIGHT;
  std::vector<uint32_t> data0(numPixels0, 0xFFFF0000);
  res = texture->upload(texture->getFullRange(0), data0.data());
  ASSERT_TRUE(res.isOk()) << res.message;

  // Upload to mip level 1 (2x2)
  const size_t numPixels1 = (ROUNDTRIP_TEX_WIDTH / 2) * (ROUNDTRIP_TEX_HEIGHT / 2);
  std::vector<uint32_t> data1(numPixels1, 0xFF0000FF);
  res = texture->upload(texture->getFullRange(1), data1.data());
  ASSERT_TRUE(res.isOk()) << res.message;
}

//
// Upload3DTexture
//
// Upload data to a 3D texture and verify the upload succeeds.
//
TEST_F(MetalTextureDataRoundtripTest, Upload3DTexture) {
  Result res;

  constexpr uint32_t kWidth = 4;
  constexpr uint32_t kHeight = 4;
  constexpr uint32_t kDepth = 4;

  const TextureDesc texDesc{.type = TextureType::ThreeD,
                            .format = TextureFormat::RGBA_UNorm8,
                            .width = kWidth,
                            .height = kHeight,
                            .depth = kDepth,
                            .numMipLevels = 1,
                            .usage = TextureDesc::TextureUsageBits::Sampled};

  auto texture = device_->createTexture(texDesc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(texture, nullptr);

  const size_t numPixels = kWidth * kHeight * kDepth;
  std::vector<uint32_t> data(numPixels, 0xFF808080);

  auto range = TextureRangeDesc::new3D(0, 0, 0, kWidth, kHeight, kDepth);
  res = texture->upload(range, data.data());
  ASSERT_TRUE(res.isOk()) << res.message;
}

TEST_F(MetalTextureDataRoundtripTest, PrivateUploadPaddedRows) {
  const TextureDesc desc{.width = 4,
                         .height = 4,
                         .usage = TextureDesc::TextureUsageBits::Sampled,
                         .type = TextureType::TwoD,
                         .format = TextureFormat::RGBA_UNorm8,
                         .storage = ResourceStorage::Private};
  Result result;
  auto texture = device_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  std::vector<uint32_t> source(32, 0xDEADBEEF);
  std::vector<uint32_t> expected(16);
  for (size_t row = 0; row < 4; ++row) {
    for (size_t column = 0; column < 4; ++column) {
      expected[row * 4 + column] = 0x12340000u + static_cast<uint32_t>(row * 4 + column);
      source[row * 8 + column] = expected[row * 4 + column];
    }
  }
  result = texture->upload(texture->getFullRange(), source.data(), 8 * sizeof(uint32_t));
  ASSERT_TRUE(result.isOk()) << result.message;
  expectPrivateTexels(texture, texture->getFullRange(), expected);
}

TEST_F(MetalTextureDataRoundtripTest, PrivateUploadAllMipLevels) {
  const TextureDesc desc{.width = 4,
                         .height = 4,
                         .usage = TextureDesc::TextureUsageBits::Sampled,
                         .numMipLevels = 3,
                         .type = TextureType::TwoD,
                         .format = TextureFormat::RGBA_UNorm8,
                         .storage = ResourceStorage::Private};
  Result result;
  auto texture = device_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  std::vector<uint32_t> data(21);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = 0x12345600u + static_cast<uint32_t>(i);
  }
  result = texture->upload(texture->getFullMipRange(), data.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  size_t offset = 0;
  for (uint32_t level = 0; level < 3; ++level) {
    const TextureRangeDesc range = texture->getFullRange(level);
    const size_t count = range.width * range.height;
    const std::vector<uint32_t> expected(data.begin() + offset, data.begin() + offset + count);
    expectPrivateTexels(texture, range, expected);
    offset += count;
  }
}

TEST_F(MetalTextureDataRoundtripTest, PrivateUploadArrayLayersAndPartialRegion) {
  const TextureDesc desc{.width = 4,
                         .height = 2,
                         .numLayers = 3,
                         .usage = TextureDesc::TextureUsageBits::Sampled,
                         .type = TextureType::TwoDArray,
                         .format = TextureFormat::RGBA_UNorm8,
                         .storage = ResourceStorage::Private};
  Result result;
  auto texture = device_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  std::vector<uint32_t> data(24);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = 0x56781200u + static_cast<uint32_t>(i);
  }
  result = texture->upload(texture->getFullRange(), data.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  for (uint32_t layer = 0; layer < 3; ++layer) {
    const std::vector<uint32_t> expected(data.begin() + layer * 8, data.begin() + (layer + 1) * 8);
    expectPrivateTexels(texture, texture->getLayerRange(layer), expected);
  }
  const TextureRangeDesc subRange = TextureRangeDesc::new2DArray(1, 1, 2, 1, 1, 2);
  const std::vector<uint32_t> replacement{0x01020304, 0x11121314, 0x21222324, 0x31323334};
  result = texture->upload(subRange, replacement.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  data[13] = replacement[0];
  data[14] = replacement[1];
  data[21] = replacement[2];
  data[22] = replacement[3];
  for (uint32_t layer = 0; layer < 3; ++layer) {
    const std::vector<uint32_t> expected(data.begin() + layer * 8, data.begin() + (layer + 1) * 8);
    expectPrivateTexels(texture, texture->getLayerRange(layer), expected);
  }
}

TEST_F(MetalTextureDataRoundtripTest, PrivateUploadCubeFaces) {
  const TextureDesc desc{.width = 4,
                         .height = 4,
                         .usage = TextureDesc::TextureUsageBits::Sampled,
                         .type = TextureType::Cube,
                         .format = TextureFormat::RGBA_UNorm8,
                         .storage = ResourceStorage::Private};
  Result result;
  auto texture = device_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  std::vector<uint32_t> data(96);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = 0x34567800u + static_cast<uint32_t>(i);
  }
  result = texture->upload(texture->getFullRange(), data.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  for (uint32_t face = 0; face < 6; ++face) {
    const std::vector<uint32_t> expected(data.begin() + face * 16, data.begin() + (face + 1) * 16);
    expectPrivateTexels(texture, texture->getCubeFaceRange(face), expected);
  }
}

TEST_F(MetalTextureDataRoundtripTest, PrivateUpload3DAndPartialDepth) {
  const TextureDesc desc{.width = 4,
                         .height = 4,
                         .depth = 3,
                         .usage = TextureDesc::TextureUsageBits::Sampled,
                         .type = TextureType::ThreeD,
                         .format = TextureFormat::RGBA_UNorm8,
                         .storage = ResourceStorage::Private};
  Result result;
  auto texture = device_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  std::vector<uint32_t> data(48);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = 0x789ABC00u + static_cast<uint32_t>(i);
  }
  result = texture->upload(texture->getFullRange(), data.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  expectPrivateTexels(texture, texture->getFullRange(), data);
  const TextureRangeDesc subRange = TextureRangeDesc::new3D(1, 1, 1, 2, 1, 1);
  const std::vector<uint32_t> replacement{0x12345678, 0x87654321};
  result = texture->upload(subRange, replacement.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  data[21] = replacement[0];
  data[22] = replacement[1];
  expectPrivateTexels(texture, texture->getFullRange(), data);
}

TEST_F(MetalTextureDataRoundtripTest, PrivateUpload3DPaddedRowsAndPartialRegion) {
  const TextureDesc desc{.width = 5,
                         .height = 4,
                         .depth = 4,
                         .usage = TextureDesc::TextureUsageBits::Sampled,
                         .type = TextureType::ThreeD,
                         .format = TextureFormat::RGBA_UNorm8,
                         .storage = ResourceStorage::Private};
  Result result;
  auto texture = device_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  std::vector<uint32_t> expected(5 * 4 * 4, 0xA1B2C3D4);
  result = texture->upload(texture->getFullRange(), expected.data());
  ASSERT_TRUE(result.isOk()) << result.message;

  const TextureRangeDesc range = TextureRangeDesc::new3D(1, 1, 1, 3, 2, 3);
  constexpr size_t kSourceRowWords = 6;
  std::vector<uint32_t> source((range.height * range.depth - 1) * kSourceRowWords + range.width,
                               0xDEADBEEF);
  for (size_t z = 0; z < range.depth; ++z) {
    for (size_t y = 0; y < range.height; ++y) {
      for (size_t x = 0; x < range.width; ++x) {
        const uint32_t value = 0x12340000u + static_cast<uint32_t>(z * 0x100 + y * 0x10 + x);
        source[(z * range.height + y) * kSourceRowWords + x] = value;
        expected[((z + range.z) * desc.height + y + range.y) * desc.width + x + range.x] = value;
      }
    }
  }
  result = texture->upload(range, source.data(), kSourceRowWords * sizeof(uint32_t));
  ASSERT_TRUE(result.isOk()) << result.message;
  expectPrivateTexels(texture, texture->getFullRange(), expected);
}

TEST_F(MetalTextureDataRoundtripTest, WrappedNativePrivateUpload) {
  const id<MTLDevice> nativeDevice = static_cast<metal::Device&>(*device_).get();
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:8
                                                        height:8
                                                     mipmapped:NO];
  desc.storageMode = MTLStorageModePrivate;
  desc.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> nativeTexture = [nativeDevice newTextureWithDescriptor:desc];
  nativeTexture.label = @"Wrapped private upload test texture";
  ASSERT_NE(nativeTexture, nil);
  auto texture = std::make_shared<metal::Texture>(nativeTexture, *device_);
  const std::vector<uint32_t> data(64, 0x12345678);
  const Result result = texture->upload(texture->getFullRange(), data.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  expectPrivateTexels(texture, texture->getFullRange(), data);
}

TEST_F(MetalTextureDataRoundtripTest, PrivateUploadGeneratesMipmapsBeforeReturning) {
  const TextureDesc desc{.width = 4,
                         .height = 4,
                         .usage = TextureDesc::TextureUsageBits::Sampled,
                         .numMipLevels = 3,
                         .type = TextureType::TwoD,
                         .format = TextureFormat::RGBA_UNorm8,
                         .storage = ResourceStorage::Private,
                         .mipmapGeneration =
                             TextureDesc::TextureMipmapGeneration::AutoGenerateOnUpload};
  Result result;
  auto texture = device_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  const std::vector<uint32_t> data(16, 0x12345678);
  result = texture->upload(texture->getFullRange(), data.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  EXPECT_FALSE(texture->isRequiredGenerateMipmap());
  for (uint32_t level = 0; level < 3; ++level) {
    const TextureRangeDesc range = texture->getFullRange(level);
    expectPrivateTexels(texture, range, std::vector<uint32_t>(range.width * range.height, data[0]));
  }
  const std::vector<uint32_t> replacement(4, 0xDEADBEEF);
  result = texture->upload(texture->getFullRange(1), replacement.data());
  EXPECT_EQ(result.code, Result::Code::InvalidOperation);
  expectPrivateTexels(texture, texture->getFullRange(1), std::vector<uint32_t>(4, data[0]));
}

#if IGL_PLATFORM_MACOSX
TEST_F(MetalTextureDataRoundtripTest, PrivateUploadCompressedBlocks) {
  const TextureDesc desc{.width = 8,
                         .height = 8,
                         .usage = TextureDesc::TextureUsageBits::Sampled,
                         .type = TextureType::TwoD,
                         .format = TextureFormat::RGBA_BC7_UNORM_4x4,
                         .storage = ResourceStorage::Private};
  Result result;
  auto texture = device_->createTexture(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(texture, nullptr);
  std::vector<uint32_t> data(16);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = 0x12345678u + static_cast<uint32_t>(i);
  }
  result = texture->upload(texture->getFullRange(), data.data());
  ASSERT_TRUE(result.isOk()) << result.message;
  expectPrivateTexels(texture, texture->getFullRange(), data);
}
#endif

} // namespace igl::tests
