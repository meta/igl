/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <cstring>
#include <igl/CommandBuffer.h>
#include <igl/Framebuffer.h>
#include <igl/RenderPass.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/GLIncludes.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/PlatformDevice.h>

namespace igl::tests {

#define OFFSCREEN_TEX_WIDTH 4
#define OFFSCREEN_TEX_HEIGHT 4

//
// FramebufferBlitOGLTest
//
// Tests for OpenGL framebuffer blit operations.
//
class FramebufferBlitOGLTest : public ::testing::Test {
 public:
  FramebufferBlitOGLTest() = default;

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
// ColorBlit
//
// Create source and destination FBOs, clear source to red, blit, verify destination.
//
TEST_F(FramebufferBlitOGLTest, ColorBlit) {
  if (!context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::FramebufferBlit)) {
    GTEST_SKIP() << "FramebufferBlit not supported";
  }

  Result ret;

  // Create source texture and framebuffer
  const TextureDesc srcTexDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                    OFFSCREEN_TEX_WIDTH,
                                                    OFFSCREEN_TEX_HEIGHT,
                                                    TextureDesc::TextureUsageBits::Sampled |
                                                        TextureDesc::TextureUsageBits::Attachment);
  auto srcTexture = iglDev_->createTexture(srcTexDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_NE(srcTexture, nullptr);

  const FramebufferDesc srcFbDesc{.colorAttachments = {{.texture = srcTexture}}};
  auto srcFramebuffer = iglDev_->createFramebuffer(srcFbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_NE(srcFramebuffer, nullptr);

  // Create destination texture and framebuffer
  const TextureDesc dstTexDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                    OFFSCREEN_TEX_WIDTH,
                                                    OFFSCREEN_TEX_HEIGHT,
                                                    TextureDesc::TextureUsageBits::Sampled |
                                                        TextureDesc::TextureUsageBits::Attachment);
  auto dstTexture = iglDev_->createTexture(dstTexDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_NE(dstTexture, nullptr);

  const FramebufferDesc dstFbDesc{.colorAttachments = {{.texture = dstTexture}}};
  auto dstFramebuffer = iglDev_->createFramebuffer(dstFbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_NE(dstFramebuffer, nullptr);

  // Clear source framebuffer to red
  RenderPassDesc renderPass;
  renderPass = {
      .colorAttachments = {{.loadAction = LoadAction::Clear,
                            .storeAction = StoreAction::Store,
                            .clearColor = {1.0, 0.0, 0.0, 1.0}}}, // Red
  };

  CommandBufferDesc cbDesc;
  auto cmdBuf = cmdQueue_->createCommandBuffer(cbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass, srcFramebuffer);
  ASSERT_NE(cmdEncoder, nullptr);
  cmdEncoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);

  // Clear destination framebuffer to black
  RenderPassDesc dstRenderPass;
  dstRenderPass = {
      .colorAttachments = {{.loadAction = LoadAction::Clear,
                            .storeAction = StoreAction::Store,
                            .clearColor = {0.0, 0.0, 0.0, 1.0}}}, // Black
  };

  auto cmdBuf2 = cmdQueue_->createCommandBuffer(cbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder2 = cmdBuf2->createRenderCommandEncoder(dstRenderPass, dstFramebuffer);
  ASSERT_NE(cmdEncoder2, nullptr);
  cmdEncoder2->endEncoding();
  cmdQueue_->submit(*cmdBuf2);

  // Blit from source to destination
  auto& device = static_cast<opengl::Device&>(*iglDev_);
  const auto& platformDevice =
      static_cast<const opengl::PlatformDevice&>(device.getPlatformDevice());

  platformDevice.blitFramebuffer(srcFramebuffer,
                                 0,
                                 0,
                                 OFFSCREEN_TEX_WIDTH,
                                 OFFSCREEN_TEX_HEIGHT,
                                 dstFramebuffer,
                                 0,
                                 0,
                                 OFFSCREEN_TEX_WIDTH,
                                 OFFSCREEN_TEX_HEIGHT,
                                 GL_COLOR_BUFFER_BIT,
                                 &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Read back destination pixels
  std::array<uint32_t, OFFSCREEN_TEX_WIDTH * OFFSCREEN_TEX_HEIGHT> pixels{};
  dstFramebuffer->copyBytesColorAttachment(
      *cmdQueue_,
      0,
      pixels.data(),
      TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT));

  // Verify destination has red pixels (RGBA = 0xFF0000FF in little-endian ABGR)
  for (auto px : pixels) {
    // Check red channel is non-zero
    uint8_t r = px & 0xFF;
    ASSERT_GT(r, 0u) << "Expected red channel to be non-zero after blit";
  }

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
