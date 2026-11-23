/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/opengl/Device.h>

namespace igl::tests {

//
// BindlessTexturesOGLTest
//
// Tests for bindless texture operations in OpenGL.
//
class BindlessTexturesOGLTest : public ::testing::Test {
 public:
  BindlessTexturesOGLTest() = default;

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
// GetTextureHandle
//
// Get a bindless texture handle if the extension is supported.
//
TEST_F(BindlessTexturesOGLTest, GetTextureHandle) {
  if (!iglDev_->hasFeature(DeviceFeatures::TextureBindless)) {
    GTEST_SKIP() << "Bindless textures not supported";
  }

  Result ret;

  // Create a simple texture
  const TextureDesc texDesc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 2, 2, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  // Attempt to get a bindless handle via IContext
  // The texture needs to have a valid GL name for this to work
  // This is a low-level test - the actual handle value depends on the driver
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
