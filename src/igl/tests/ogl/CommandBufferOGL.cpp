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
#include <igl/opengl/Framebuffer.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/PlatformDevice.h>
#if IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX_USE_EGL
#include <igl/opengl/egl/PlatformDevice.h>
#endif

namespace igl::tests {

#define OFFSCREEN_TEX_WIDTH 2
#define OFFSCREEN_TEX_HEIGHT 2

//
// CommandBufferOGLTest
//
// Tests for the OpenGL CommandBuffer.
//
class CommandBufferOGLTest : public ::testing::Test {
 public:
  CommandBufferOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();

    Result ret;

    // Create offscreen texture
    const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                   OFFSCREEN_TEX_WIDTH,
                                                   OFFSCREEN_TEX_HEIGHT,
                                                   TextureDesc::TextureUsageBits::Sampled |
                                                       TextureDesc::TextureUsageBits::Attachment);
    offscreenTexture_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);

    // Create framebuffer
    const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = offscreenTexture_}}};
    framebuffer_ = iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);

    // Initialize render pass
    renderPass_ = {
        .colorAttachments = {{.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::Store,
                              .clearColor = {0.0, 0.0, 0.0, 1.0}}},
    };
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;

  RenderPassDesc renderPass_;
  std::shared_ptr<ITexture> offscreenTexture_;
  std::shared_ptr<IFramebuffer> framebuffer_;
};

//
// CreateFromQueue
//
// Create a command buffer from the command queue.
//
TEST_F(CommandBufferOGLTest, CreateFromQueue) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_NE(cmdBuf, nullptr);
}

//
// CreateRenderEncoder
//
// Create a render command encoder from a command buffer.
//
TEST_F(CommandBufferOGLTest, CreateRenderEncoder) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_NE(cmdBuf, nullptr);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass_, framebuffer_);
  ASSERT_NE(cmdEncoder, nullptr);

  cmdEncoder->endEncoding();
}

//
// SubmitToQueue
//
// Submit a command buffer to the queue.
//
TEST_F(CommandBufferOGLTest, SubmitToQueue) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_NE(cmdBuf, nullptr);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass_, framebuffer_);
  ASSERT_NE(cmdEncoder, nullptr);
  cmdEncoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);

  // Verify no GL errors after submission
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// SubpassesRejectCurrentFramebuffer
//
// Render passes with subpasses need a framebuffer whose color attachments are textures, and the
// color attachment of CurrentFramebuffer is a placeholder.
//
TEST_F(CommandBufferOGLTest, SubpassesRejectCurrentFramebuffer) {
  if (!iglDev_->hasFeature(DeviceFeatures::Subpasses)) {
    GTEST_SKIP() << "Subpasses are not supported";
  }

  auto* platformDevice = iglDev_->getPlatformDevice<opengl::PlatformDevice>();
  ASSERT_NE(platformDevice, nullptr);
  const std::shared_ptr<IFramebuffer> currentFramebuffer =
      platformDevice->createCurrentFramebuffer();
  ASSERT_NE(currentFramebuffer, nullptr);

  const RenderPassDesc renderPass{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}},
      .subpasses = {{.colorAttachmentMask = 0b1}, {.colorAttachmentMask = 0b1}},
  };
  Result ret;
  const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(cmdBuf, nullptr);

  const auto encoder = cmdBuf->createRenderCommandEncoder(renderPass, currentFramebuffer, &ret);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);
  EXPECT_EQ(encoder, nullptr);
  EXPECT_NE(ret.message.find("textures"), std::string::npos) << ret.message;
}

//
// SubpassesRejectWindowFramebuffer
//
// The framebuffer of the window system ignores every color attachment after the native drawable,
// so render passes with subpasses reject a framebuffer whose color attachment 0 is the drawable.
//
#if IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX_USE_EGL
TEST_F(CommandBufferOGLTest, SubpassesRejectWindowFramebuffer) {
  if (!iglDev_->hasFeature(DeviceFeatures::Subpasses)) {
    GTEST_SKIP() << "Subpasses are not supported";
  }

  auto* platformDevice = iglDev_->getPlatformDevice<opengl::egl::PlatformDevice>();
  if (!platformDevice) {
    GTEST_SKIP() << "EGL PlatformDevice not available";
  }

  Result drawableRet;
  const auto drawable =
      platformDevice->createTextureFromNativeDrawable(TextureFormat::RGBA_UNorm8, &drawableRet);
  if (!drawable || !drawableRet.isOk()) {
    GTEST_SKIP() << "Could not create texture from native drawable";
  }

  // Create a regular texture to use as the second attachment
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 drawable->getDimensions().width,
                                                 drawable->getDimensions().height,
                                                 TextureDesc::TextureUsageBits::Attachment);
  Result texRet;
  const auto regularTexture = iglDev_->createTexture(texDesc, &texRet);
  ASSERT_TRUE(texRet.isOk()) << texRet.message;
  ASSERT_NE(regularTexture, nullptr);

  // Create a framebuffer with the drawable and a regular texture
  const FramebufferDesc fbDesc{
      .colorAttachments = {{.texture = drawable}, {.texture = regularTexture}},
  };
  Result fbRet;
  const auto windowFramebuffer = iglDev_->createFramebuffer(fbDesc, &fbRet);
  ASSERT_TRUE(fbRet.isOk()) << fbRet.message;
  ASSERT_NE(windowFramebuffer, nullptr);

  const RenderPassDesc renderPass{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store},
                           {.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}},
      .subpasses = {{.colorAttachmentMask = 0b11}, {.colorAttachmentMask = 0b01}},
  };

  Result ret;
  const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(cmdBuf, nullptr);

  const auto encoder = cmdBuf->createRenderCommandEncoder(renderPass, windowFramebuffer, &ret);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);
  EXPECT_EQ(encoder, nullptr);
  EXPECT_NE(ret.message.find("textures"), std::string::npos) << ret.message;
}
#endif

} // namespace igl::tests
