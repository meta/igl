/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/Framebuffer.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// MultiviewFramebufferOGLTest
//
// Tests for multiview framebuffer creation in OpenGL.
//
class MultiviewFramebufferOGLTest : public ::testing::Test {
 public:
  MultiviewFramebufferOGLTest() = default;

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
// CreateMultiviewFramebuffer
//
// Create a framebuffer with multiview texture array attachments.
//
TEST_F(MultiviewFramebufferOGLTest, CreateMultiviewFramebuffer) {
  if (!iglDev_->hasFeature(DeviceFeatures::Multiview)) {
    GTEST_SKIP() << "Multiview not supported";
  }

  if (!iglDev_->hasFeature(DeviceFeatures::Texture2DArray)) {
    GTEST_SKIP() << "Texture2DArray not supported";
  }

  Result ret;

  // Create a 2D texture array with 2 layers for stereo rendering
  TextureDesc texDesc = TextureDesc::new2DArray(TextureFormat::RGBA_UNorm8,
                                                4,
                                                4,
                                                2, // 2 layers for stereo
                                                TextureDesc::TextureUsageBits::Sampled |
                                                    TextureDesc::TextureUsageBits::Attachment);
  auto texture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  // Create framebuffer with multiview mode
  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = texture}},
                               .mode = FramebufferMode::Stereo};

  auto framebuffer = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(framebuffer, nullptr);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
