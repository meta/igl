/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/TextureBuffer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <igl/opengl/CommandQueue.h>
#include <igl/opengl/Device.h>
#include <igl/tests/util/TestDevice.h>
#include <igl/tests/util/TextureValidationHelpers.h>

namespace igl::tests {

// Picking this just to match the texture we will use. If you use a different
// size texture, then you will have to either create a new offscreenTexture_
// and the framebuffer object in your test, so know exactly what the end result
// would be after sampling
#define OFFSCREEN_TEX_WIDTH 2
#define OFFSCREEN_TEX_HEIGHT 2

// Picking this to check mipmap validity, max mipmap level = log(16) - 1 = 3
#define MIPMAP_TEX_WIDTH 16
#define MIPMAP_TEX_HEIGHT 16

struct TextureFormatToGL {
  TextureFormat texFormatInput = TextureFormat::Invalid;

  // Expected conversion to GL
  GLuint glInternalFormat = 0x0000;
  GLuint glFormat = 0x0000;
  GLuint glType = 0x0000;
};

//
// TextureBufferOGLTest
//
// Unit tests for igl::opengl::TextureBuffer.
// Covers code paths that may not be hit by top level texture calls from device.
//
class TextureBufferOGLTest : public ::testing::Test {
 private:
 public:
  TextureBufferOGLTest() = default;

  void SetUp() override {
    // Turn off debug breaks, only use in debug mode
    setDebugBreakEnabled(false);

    device_ = util::createTestDevice();
    ASSERT_TRUE(device_ != nullptr);
    context_ = &static_cast<opengl::Device&>(*device_).getContext();
    ASSERT_TRUE(context_ != nullptr);
  }

  void TearDown() override {}

  // Member variables
 protected:
  opengl::IContext* context_{};
  std::shared_ptr<IDevice> device_;
};

//
// Texture Creation Paths Test
//
// This tests all failure and success paths for TextureBuffer::create.
// Also covers private function createTexture which is called within create.
// See tests for Texture.cpp for cases covering cases specific to base class.
//
TEST_F(TextureBufferOGLTest, TextureCreation) {
  std::unique_ptr<opengl::TextureBuffer> textureBuffer;
  Result ret;
  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::ABGR_UNorm4,
                                           OFFSCREEN_TEX_WIDTH,
                                           OFFSCREEN_TEX_HEIGHT,
                                           TextureDesc::TextureUsageBits::Sampled);

  // Correct usage of TextureBuffer::create
  textureBuffer = std::make_unique<opengl::TextureBuffer>(*context_, texDesc.format);
  ret = textureBuffer->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  // kRenderTarget not supported by TextureBuffer
  texDesc.usage = TextureDesc::TextureUsageBits::Attachment;
  textureBuffer = std::make_unique<opengl::TextureBuffer>(*context_, texDesc.format);
  ret = textureBuffer->create(texDesc, false);
  ASSERT_FALSE(ret.isOk());

  texDesc.usage = TextureDesc::TextureUsageBits::Sampled;

  // Incorrect texture format
  texDesc.format = TextureFormat::Invalid;
  textureBuffer = std::make_unique<opengl::TextureBuffer>(*context_, texDesc.format);
  ret = textureBuffer->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

//
// TextureBuffer Mipmap generation Test
//
// Tests expected behavior for Texture mipmap for supported GL formats.
// Test paths are tested through TextureBuffer::create
//
TEST_F(TextureBufferOGLTest, TextureMipmapGen) {
  std::unique_ptr<opengl::TextureBuffer> textureBuffer;
  Result ret;
  // Generate mipmap and correct query of initial count
  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           MIPMAP_TEX_WIDTH,
                                           MIPMAP_TEX_HEIGHT,
                                           TextureDesc::TextureUsageBits::Sampled);

  const size_t maxDim = std::max<size_t>(texDesc.width, texDesc.height);
  const int targetlevel = std::floor(std::log2(maxDim)) + 1;

  texDesc.numMipLevels = targetlevel; // log(16) + 1
  textureBuffer = std::make_unique<opengl::TextureBuffer>(*context_, texDesc.format);
  ret = textureBuffer->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  opengl::CommandQueue queue;
  textureBuffer->generateMipmap(queue);
  ASSERT_EQ(textureBuffer->getNumMipLevels(), targetlevel);
}

//
// AutoGenerateOnUpload Test
//
// This test verifies that the AutoGenerateOnUpload flag correctly triggers
// mipmap generation when texture data is uploaded.
//
TEST_F(TextureBufferOGLTest, AutoGenerateMipmapOnUpload) {
  Result ret;

  constexpr uint32_t kNumMipLevels = 2u;
  constexpr uint32_t kTexWidth = 2u;
  constexpr uint32_t kTexHeight = 2u;

  constexpr uint32_t kColor = 0xdeadbeef;
  constexpr std::array<uint32_t, 4> kBaseMipData = {kColor, kColor, kColor, kColor};
  constexpr std::array<uint32_t, 1> kExpectedMip1Data = {kColor}; // Should be same color after
                                                                  // generation

  // Create texture with AutoGenerateOnUpload flag
  TextureDesc textureDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                               kTexWidth,
                                               kTexHeight,
                                               TextureDesc::TextureUsageBits::Sampled |
                                                   TextureDesc::TextureUsageBits::Attachment);
  textureDesc.numMipLevels = kNumMipLevels;
  textureDesc.mipmapGeneration = TextureDesc::TextureMipmapGeneration::AutoGenerateOnUpload;

  auto texture = device_->createTexture(textureDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  auto* oglTexture = static_cast<opengl::Texture*>(texture.get());
  ASSERT_NE(oglTexture, nullptr);

  ASSERT_EQ(oglTexture->getMipmapGeneration(),
            TextureDesc::TextureMipmapGeneration::AutoGenerateOnUpload);

  ASSERT_EQ(oglTexture->getNumMipLevels(), kNumMipLevels);

  CommandQueueDesc cmdQueueDesc{};
  auto cmdQueue = device_->createCommandQueue(cmdQueueDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cmdQueue, nullptr);

  // Upload data to mip level 0 - this should trigger automatic mipmap generation
  ret = texture->upload(texture->getFullRange(0), kBaseMipData.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Validate that mip level 0 contains the uploaded data
  util::validateUploadedTextureRange(*device_,
                                     *cmdQueue,
                                     texture,
                                     texture->getFullRange(0),
                                     kBaseMipData.data(),
                                     "AutoGen: Base level (0)");

  // Validate that mip level 1 was auto-generated with expected content
  // The auto-generated mip should contain the same solid color (averaged from base level)
  util::validateUploadedTextureRange(*device_,
                                     *cmdQueue,
                                     texture,
                                     texture->getFullRange(1),
                                     kExpectedMip1Data.data(),
                                     "AutoGen: Generated level (1)");
}

} // namespace igl::tests
