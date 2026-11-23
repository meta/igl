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
// CompressedTextureOGLTest
//
// Tests for compressed texture creation in OpenGL.
//
class CompressedTextureOGLTest : public ::testing::Test {
 public:
  CompressedTextureOGLTest() = default;

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
// CreateCompressed
//
// Create a compressed texture if the format is supported, otherwise skip.
//
TEST_F(CompressedTextureOGLTest, CreateCompressed) {
  // Try ETC2 format first (common on GLES 3.0+)
  TextureFormat format = TextureFormat::RGBA8_EAC_ETC2;
  auto caps = iglDev_->getTextureFormatCapabilities(format);

  if ((caps & ICapabilities::TextureFormatCapabilityBits::Sampled) == 0) {
    // Try ASTC
    format = TextureFormat::RGBA_ASTC_4x4;
    caps = iglDev_->getTextureFormatCapabilities(format);
  }

  if ((caps & ICapabilities::TextureFormatCapabilityBits::Sampled) == 0) {
    GTEST_SKIP() << "No supported compressed texture format found";
  }

  Result ret;
  TextureDesc desc = TextureDesc::new2D(format, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
}

} // namespace igl::tests
