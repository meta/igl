/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/CommandBuffer.h>
#include <igl/RenderPass.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// CopyTexSubImageOGLTest
//
// Tests for glCopyTexSubImage2D operations via IGL.
//
class CopyTexSubImageOGLTest : public ::testing::Test {
 public:
  CopyTexSubImageOGLTest() = default;

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
// CopyTexSubImage2D
//
// Create a framebuffer, render to it, then copy a sub-region to a texture.
//
TEST_F(CopyTexSubImageOGLTest, CopyTexSubImage2D) {
  Result ret;

  // Create source framebuffer and clear to red
  const TextureDesc srcTexDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                    4,
                                                    4,
                                                    TextureDesc::TextureUsageBits::Sampled |
                                                        TextureDesc::TextureUsageBits::Attachment);
  auto srcTexture = iglDev_->createTexture(srcTexDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const FramebufferDesc srcFbDesc{.colorAttachments = {{.texture = srcTexture}}};
  auto srcFramebuffer = iglDev_->createFramebuffer(srcFbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Clear to red
  RenderPassDesc renderPass;
  renderPass = {
      .colorAttachments = {{.loadAction = LoadAction::Clear,
                            .storeAction = StoreAction::Store,
                            .clearColor = {1.0, 0.0, 0.0, 1.0}}},
  };

  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass, srcFramebuffer);
  ASSERT_NE(cmdEncoder, nullptr);
  cmdEncoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);

  // Create destination texture
  const TextureDesc dstTexDesc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  auto dstTexture = iglDev_->createTexture(dstTexDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Use copyTextureColorAttachment if available
  auto dstFbDesc = FramebufferDesc{};
  dstFbDesc.colorAttachments[0].texture = dstTexture;

  // Copy using framebuffer's copy method
  srcFramebuffer->copyTextureColorAttachment(
      *cmdQueue_, 0, dstTexture, TextureRangeDesc::new2D(0, 0, 4, 4));

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
