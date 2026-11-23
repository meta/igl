/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <array>
#include <igl/CommandBuffer.h>
#include <igl/Device.h>
#include <igl/Framebuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/Texture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class RenderCommandEncoderVulkanTest : public ::testing::Test {
 public:
  RenderCommandEncoderVulkanTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";

    Result ret;
    cmdQueue_ = iglDev_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdQueue_, nullptr);

    TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                             4,
                                             4,
                                             TextureDesc::TextureUsageBits::Attachment |
                                                 TextureDesc::TextureUsageBits::Sampled);
    colorTex_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(colorTex_, nullptr);

    const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex_}}};
    fb_ = iglDev_->createFramebuffer(fbDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(fb_, nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ITexture> colorTex_;
  std::shared_ptr<IFramebuffer> fb_;

  std::unique_ptr<IRenderCommandEncoder> createEncoder(std::shared_ptr<ICommandBuffer>& cmdBuf) {
    Result ret;
    cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
    if (!ret.isOk() || !cmdBuf) {
      return nullptr;
    }

    RenderPassDesc rpDesc;
    rpDesc = {
        .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}},
    };

    auto encoder = cmdBuf->createRenderCommandEncoder(rpDesc, fb_, {}, &ret);
    if (!ret.isOk()) {
      return nullptr;
    }
    return encoder;
  }
};

TEST_F(RenderCommandEncoderVulkanTest, BindViewport) {
  std::shared_ptr<ICommandBuffer> cmdBuf;
  auto encoder = createEncoder(cmdBuf);
  ASSERT_NE(encoder, nullptr);

  Viewport vp = {
      .x = 0.0f, .y = 0.0f, .width = 4.0f, .height = 4.0f, .minDepth = 0.0f, .maxDepth = 1.0f};
  encoder->bindViewport(vp);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(RenderCommandEncoderVulkanTest, BindScissorRect) {
  std::shared_ptr<ICommandBuffer> cmdBuf;
  auto encoder = createEncoder(cmdBuf);
  ASSERT_NE(encoder, nullptr);

  ScissorRect rect = {.x = 0, .y = 0, .width = 4, .height = 4};
  encoder->bindScissorRect(rect);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(RenderCommandEncoderVulkanTest, SetStencilReferenceValue) {
  std::shared_ptr<ICommandBuffer> cmdBuf;
  auto encoder = createEncoder(cmdBuf);
  ASSERT_NE(encoder, nullptr);

  encoder->setStencilReferenceValue(128);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(RenderCommandEncoderVulkanTest, SetBlendColor) {
  std::shared_ptr<ICommandBuffer> cmdBuf;
  auto encoder = createEncoder(cmdBuf);
  ASSERT_NE(encoder, nullptr);

  encoder->setBlendColor(Color(1.0f, 0.5f, 0.25f, 1.0f));

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(RenderCommandEncoderVulkanTest, SetDepthBias) {
  std::shared_ptr<ICommandBuffer> cmdBuf;
  auto encoder = createEncoder(cmdBuf);
  ASSERT_NE(encoder, nullptr);

  encoder->setDepthBias(1.0f, 1.0f, 0.0f);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(RenderCommandEncoderVulkanTest, SetFrontFacingWinding) {
  std::shared_ptr<ICommandBuffer> cmdBuf;
  auto encoder = createEncoder(cmdBuf);
  ASSERT_NE(encoder, nullptr);

  encoder->setFrontFacingWinding(WindingMode::Clockwise);
  encoder->setFrontFacingWinding(WindingMode::CounterClockwise);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(RenderCommandEncoderVulkanTest, DebugGroupLabels) {
  std::shared_ptr<ICommandBuffer> cmdBuf;
  auto encoder = createEncoder(cmdBuf);
  ASSERT_NE(encoder, nullptr);

  encoder->pushDebugGroupLabel("RenderGroup", Color(1.0f, 0.0f, 0.0f, 1.0f));
  encoder->insertDebugEventLabel("DrawCall", Color(0.0f, 1.0f, 0.0f, 1.0f));
  encoder->popDebugGroupLabel();

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

TEST_F(RenderCommandEncoderVulkanTest, MultipleEncodersSequentially) {
  for (int i = 0; i < 3; ++i) {
    std::shared_ptr<ICommandBuffer> cmdBuf;
    auto encoder = createEncoder(cmdBuf);
    ASSERT_NE(encoder, nullptr) << "Failed on iteration " << i;

    encoder->endEncoding();
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
  }
}

TEST_F(RenderCommandEncoderVulkanTest, ClearColorRenderPass) {
  Result ret;
  const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  const RenderPassDesc rpDesc{.colorAttachments = {{.loadAction = LoadAction::Clear,
                                                    .storeAction = StoreAction::Store,
                                                    .clearColor = {1.0f, 0.0f, 0.0f, 1.0f}}}};

  auto encoder = cmdBuf->createRenderCommandEncoder(rpDesc, fb_, {}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  std::array<uint32_t, 16> pixels = {};
  fb_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), colorTex_->getFullRange(0));

  for (const uint32_t pixel : pixels) {
    EXPECT_EQ(pixel, 0xFF0000FF);
  }
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
