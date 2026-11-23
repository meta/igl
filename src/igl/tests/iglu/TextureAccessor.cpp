/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/TextureData.h"
#include "../util/Common.h"

#include <IGLU/texture_accessor/ITextureAccessor.h>
#include <IGLU/texture_accessor/TextureAccessorFactory.h>
#include <igl/Common.h>

#define OFFSCREEN_TEX_HEIGHT 2
#define OFFSCREEN_TEX_WIDTH 2

namespace igl::tests {

//
// TextureAccessorTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class TextureAccessorTest : public ::testing::Test {
 private:
 public:
  TextureAccessorTest() = default;

  //
  // SetUp()
  // Create device, commandQueue
  //
  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);

    Result result;
    texDesc_ = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                  OFFSCREEN_TEX_WIDTH,
                                  OFFSCREEN_TEX_HEIGHT,
                                  TextureDesc::TextureUsageBits::Sampled |
                                      TextureDesc::TextureUsageBits::Attachment);
    texture_ = iglDev_->createTexture(texDesc_, &result);
    ASSERT_TRUE(result.isOk());

    // Initialize texture data
    textureSizeInBytes_ = texture_->getProperties().getBytesPerRange(texture_->getFullRange());
    const auto range = TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT);
    texture_->upload(range, data::texture::kTexRgba2x2.data());
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ITexture> texture_;
  TextureDesc texDesc_;
  std::shared_ptr<iglu::textureaccessor::ITextureAccessor> textureAccessor_;
  int textureSizeInBytes_{};
};

//
// testRequestAndGetBytesSync Test
//
// Tests synchronous texture readback
//
TEST_F(TextureAccessorTest, testRequestAndGetBytesSync) {
  const auto dimensions = texture_->getDimensions();
  ASSERT_EQ(dimensions.width, static_cast<uint32_t>(OFFSCREEN_TEX_WIDTH));
  ASSERT_EQ(dimensions.height, static_cast<uint32_t>(OFFSCREEN_TEX_HEIGHT));

  ASSERT_NO_THROW(textureAccessor_ =
                      iglu::textureaccessor::TextureAccessorFactory::createTextureAccessor(
                          iglDev_->getBackendType(), texture_, *iglDev_));
  ASSERT_TRUE(textureAccessor_ != nullptr);

  // Verify requestStatus before
  ASSERT_EQ(textureAccessor_->getRequestStatus(),
            iglu::textureaccessor::RequestStatus::NotInitialized);

  // Update texture data
  const auto rangeDesc = TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT);
  texture_->upload(rangeDesc, data::texture::kTexRgba2x2.data());

  auto bytes = textureAccessor_->requestAndGetBytesSync(*cmdQueue_);
  // Verify requestStatus after
  ASSERT_EQ(textureAccessor_->getRequestStatus(), iglu::textureaccessor::RequestStatus::Ready);

  // 2x2 texture * 4 bytes per pixel
  ASSERT_EQ(bytes.size(), 16);
  // Verify data
  auto* pixels = reinterpret_cast<uint32_t*>(bytes.data());
  for (int i = 0; (i < textureSizeInBytes_ / 4); i++) {
    ASSERT_EQ(pixels[i], data::texture::kTexRgba2x2[i]);
  }
}

TEST_F(TextureAccessorTest, reuseTextureAccessor) {
  const auto dimensions = texture_->getDimensions();
  ASSERT_EQ(dimensions.width, static_cast<uint32_t>(OFFSCREEN_TEX_WIDTH));
  ASSERT_EQ(dimensions.height, static_cast<uint32_t>(OFFSCREEN_TEX_HEIGHT));

  ASSERT_NO_THROW(textureAccessor_ =
                      iglu::textureaccessor::TextureAccessorFactory::createTextureAccessor(
                          iglDev_->getBackendType(), texture_, *iglDev_));
  ASSERT_TRUE(textureAccessor_ != nullptr);

  // Verify requestStatus before
  ASSERT_EQ(textureAccessor_->getRequestStatus(),
            iglu::textureaccessor::RequestStatus::NotInitialized);

  // First Upload
  {
    // Update texture data
    const auto rangeDesc = TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT);
    texture_->upload(rangeDesc, data::texture::kTexRgba2x2.data());

    auto bytes = textureAccessor_->requestAndGetBytesSync(*cmdQueue_);
    // Verify requestStatus after
    ASSERT_EQ(textureAccessor_->getRequestStatus(), iglu::textureaccessor::RequestStatus::Ready);

    // 2x2 texture * 4 bytes per pixel
    ASSERT_EQ(bytes.size(), 16);
    // Verify data
    auto* pixels = reinterpret_cast<uint32_t*>(bytes.data());
    for (int i = 0; (i < textureSizeInBytes_ / 4); i++) {
      ASSERT_EQ(pixels[i], data::texture::kTexRgba2x2[i]);
    }
  }

  // Second Upload
  {
    // Update texture data
    const auto rangeDesc = TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT);
    texture_->upload(rangeDesc, data::texture::kTexRgbaGray2x2.data());

    auto bytes = textureAccessor_->requestAndGetBytesSync(*cmdQueue_);
    // Verify requestStatus after
    ASSERT_EQ(textureAccessor_->getRequestStatus(), iglu::textureaccessor::RequestStatus::Ready);

    // 2x2 texture * 4 bytes per pixel
    ASSERT_EQ(bytes.size(), 16);
    // Verify data
    auto* pixels = reinterpret_cast<uint32_t*>(bytes.data());
    for (int i = 0; (i < textureSizeInBytes_ / 4); i++) {
      ASSERT_EQ(pixels[i], data::texture::kTexRgbaGray2x2[i]);
    }
  }
}
} // namespace igl::tests
