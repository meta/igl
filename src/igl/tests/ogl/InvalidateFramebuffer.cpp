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
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/ViewTextureTarget.h>

namespace igl::tests {

//
// InvalidateFramebufferOGLTest
//
// Tests for framebuffer invalidation in OpenGL.
//
class InvalidateFramebufferOGLTest : public ::testing::Test {
 public:
  InvalidateFramebufferOGLTest() = default;

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
// InvalidateNoError
//
// Invalidate a framebuffer attachment and verify no GL errors occur.
//
TEST_F(InvalidateFramebufferOGLTest, InvalidateNoError) {
  if (!context_->deviceFeatures().hasInternalFeature(
          opengl::InternalFeatures::InvalidateFramebuffer)) {
    GTEST_SKIP() << "InvalidateFramebuffer not supported";
  }

  Result ret;

  // Create a framebuffer
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Attachment);
  auto texture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = texture}}};
  auto framebuffer = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  // Clear the framebuffer first
  const RenderPassDesc renderPass{
      .colorAttachments = {{.loadAction = LoadAction::Clear,
                            // DontCare indicates invalidation
                            .storeAction = StoreAction::DontCare,
                            .clearColor = {0.0, 0.0, 0.0, 1.0}}},
  };

  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass, framebuffer);
  ASSERT_NE(cmdEncoder, nullptr);
  cmdEncoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// InvalidateDefaultFramebufferNoError
//
// A framebuffer backed by a view's implicit storage is the default framebuffer,
// whose buffers must be invalidated as GL_COLOR / GL_DEPTH rather than the
// GL_*_ATTACHMENT names that only framebuffer objects accept.
//
TEST_F(InvalidateFramebufferOGLTest, InvalidateDefaultFramebufferNoError) {
  if (!context_->deviceFeatures().hasInternalFeature(
          opengl::InternalFeatures::InvalidateFramebuffer)) {
    GTEST_SKIP() << "InvalidateFramebuffer not supported";
  }

  Result ret;
  auto color = std::make_shared<opengl::ViewTextureTarget>(*context_, TextureFormat::RGBA_UNorm8);
  auto depth = std::make_shared<opengl::ViewTextureTarget>(*context_, TextureFormat::Z_UNorm24);
  const FramebufferDesc fbDesc{
      .colorAttachments = {{.texture = color}},
      .depthAttachment = {.texture = depth},
  };
  auto framebuffer = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  const RenderPassDesc renderPass{
      .colorAttachments = {{.loadAction = LoadAction::Clear,
                            .storeAction = StoreAction::DontCare,
                            .clearColor = {0.0, 0.0, 0.0, 1.0}}},
      .depthAttachment = {.loadAction = LoadAction::Clear, .storeAction = StoreAction::DontCare},
  };
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass, framebuffer);
  ASSERT_NE(cmdEncoder, nullptr);
  cmdEncoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
