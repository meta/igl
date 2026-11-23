/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// Texture3DOGLTest
//
// Tests for 3D texture creation and operations in OpenGL.
//
class Texture3DOGLTest : public ::testing::Test {
 public:
  Texture3DOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;
};

//
// Create3DTexture
//
// Create a 3D texture and verify it is valid.
//
TEST_F(Texture3DOGLTest, Create3DTexture) {
  if (!iglDev_->hasFeature(DeviceFeatures::Texture3D)) {
    GTEST_SKIP() << "3D textures not supported";
  }

  Result ret;
  TextureDesc desc = TextureDesc::new3D(
      TextureFormat::RGBA_UNorm8, 4, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
}

//
// Upload3DData
//
// Create a 3D texture and upload data to it.
//
TEST_F(Texture3DOGLTest, Upload3DData) {
  if (!iglDev_->hasFeature(DeviceFeatures::Texture3D)) {
    GTEST_SKIP() << "3D textures not supported";
  }

  Result ret;
  const int width = 2, height = 2, depth = 2;
  TextureDesc desc = TextureDesc::new3D(
      TextureFormat::RGBA_UNorm8, width, height, depth, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  // Upload data
  std::vector<uint32_t> pixels(width * height * depth, 0xFF00FF00);
  auto range = TextureRangeDesc::new3D(0, 0, 0, width, height, depth);
  auto uploadResult = texture->upload(range, pixels.data());
  ASSERT_TRUE(uploadResult.isOk()) << uploadResult.message.c_str();
}

//
// VerifyDimensions
//
// Verify the dimensions of a created 3D texture.
//
TEST_F(Texture3DOGLTest, VerifyDimensions) {
  if (!iglDev_->hasFeature(DeviceFeatures::Texture3D)) {
    GTEST_SKIP() << "3D textures not supported";
  }

  Result ret;
  const size_t width = 8, height = 4, depth = 2;
  TextureDesc desc = TextureDesc::new3D(
      TextureFormat::RGBA_UNorm8, width, height, depth, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  auto dimensions = texture->getDimensions();
  ASSERT_EQ(dimensions.width, width);
  ASSERT_EQ(dimensions.height, height);
  ASSERT_EQ(dimensions.depth, depth);
}

} // namespace igl::tests
