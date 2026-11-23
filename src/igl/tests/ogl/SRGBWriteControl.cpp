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
// SRGBWriteControlOGLTest
//
// Tests for sRGB write control in OpenGL.
//
class SRGBWriteControlOGLTest : public ::testing::Test {
 public:
  SRGBWriteControlOGLTest() = default;

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
// SRGBWriteEnableDisable
//
// Test enabling and disabling sRGB framebuffer write.
//
TEST_F(SRGBWriteControlOGLTest, SRGBWriteEnableDisable) {
  if (!iglDev_->hasFeature(DeviceFeatures::SRGBWriteControl)) {
    GTEST_SKIP() << "sRGB write control not supported";
  }

  // Enable GL_FRAMEBUFFER_SRGB
  context_->enable(GL_FRAMEBUFFER_SRGB);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  GLboolean isEnabled = context_->isEnabled(GL_FRAMEBUFFER_SRGB);
  ASSERT_EQ(isEnabled, GL_TRUE);

  // Disable GL_FRAMEBUFFER_SRGB
  context_->disable(GL_FRAMEBUFFER_SRGB);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  isEnabled = context_->isEnabled(GL_FRAMEBUFFER_SRGB);
  ASSERT_EQ(isEnabled, GL_FALSE);
}

} // namespace igl::tests
