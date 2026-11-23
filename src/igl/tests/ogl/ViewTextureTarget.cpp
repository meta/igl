/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/ViewTextureTarget.h>

#include "../util/TestDevice.h"

#include <igl/opengl/Device.h>

namespace igl::tests {

//
// ViewTextureTargetOGLTest
//
// Unit tests for igl::opengl::ViewTextureTarget.
// Covers code paths that may not be hit by top level texture calls from device.
//
class ViewTextureTargetOGLTest : public ::testing::Test {
 public:
  ViewTextureTargetOGLTest() = default;

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
// Specifications Test
//
// This test verifies that the default ViewTextureTarget specs are defined
// correctly. Note that "correct" in this case only means how the code is
// currently written. Should the ViewTextureTarget code change, then this
// test will need to be updated as well.
//
TEST_F(ViewTextureTargetOGLTest, Specifications) {
  std::unique_ptr<opengl::ViewTextureTarget> viewTextureTarget =
      std::make_unique<opengl::ViewTextureTarget>(*context_, TextureFormat::RGBA_UNorm8);
  ASSERT_EQ(viewTextureTarget->getType(), TextureType::TwoD);
  ASSERT_EQ(viewTextureTarget->getUsage(), TextureDesc::TextureUsageBits::Attachment);
  ASSERT_TRUE(viewTextureTarget->isImplicitStorage());
}

//
// NoOpFunctions Test
//
// This test calls no-op functions just to satisfy code coverage.
//
TEST_F(ViewTextureTargetOGLTest, NoOpFunctions) {
  std::unique_ptr<opengl::ViewTextureTarget> viewTextureTarget =
      std::make_unique<opengl::ViewTextureTarget>(*context_, TextureFormat::RGBA_UNorm8);
  viewTextureTarget->bind();
  viewTextureTarget->unbind();
  viewTextureTarget->attachAsColor(0, opengl::Texture::AttachmentParams{});
  viewTextureTarget->detachAsColor(0, false);
  viewTextureTarget->attachAsDepth(opengl::Texture::AttachmentParams{});
  viewTextureTarget->attachAsStencil(opengl::Texture::AttachmentParams{});
}

} // namespace igl::tests
