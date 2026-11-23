/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <array>
#include <vector>
#include <igl/CommandBuffer.h>
#include <igl/Device.h>
#include <igl/Framebuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/ShaderCreator.h>
#include <igl/Texture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

#include <igl/vulkan/VulkanPipelineBuilder.h>

namespace igl::tests {

class SubpassPipelineCacheVulkanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
#if IGL_PLATFORM_LINUX
    // Lavapipe supports subpasses, so the subpass tests must not skip on Linux
    ASSERT_TRUE(iglDev_->hasFeature(DeviceFeatures::Subpasses));
#else
    if (!iglDev_->hasFeature(DeviceFeatures::Subpasses)) {
      GTEST_SKIP() << "Subpasses are not supported";
    }
#endif // IGL_PLATFORM_LINUX

    Result ret;
    cmdQueue_ = iglDev_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdQueue_, nullptr);
  }

  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;

  [[nodiscard]] std::shared_ptr<ITexture> createColorTexture(
      TextureDesc::TextureUsage usage) const {
    const TextureDesc desc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 8, 8, usage);
    Result ret;
    const auto tex = iglDev_->createTexture(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    EXPECT_NE(tex, nullptr);
    return tex;
  }

  [[nodiscard]] std::shared_ptr<IFramebuffer> createFramebuffer(
      const std::vector<std::shared_ptr<ITexture>>& colorAttachments) const {
    FramebufferDesc desc;
    for (size_t i = 0; i < colorAttachments.size() && i < IGL_COLOR_ATTACHMENTS_MAX; ++i) {
      desc.colorAttachments[i].texture = colorAttachments[i];
    }
    Result ret;
    const auto fb = iglDev_->createFramebuffer(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    EXPECT_NE(fb, nullptr);
    return fb;
  }

  [[nodiscard]] std::shared_ptr<IRenderPipelineState> createPipeline(const char* fragmentShader,
                                                                     size_t numColorFormats,
                                                                     const char* name) const {
    Result ret;
    const char* vertexShader = R"(#version 460
out gl_PerVertex { vec4 gl_Position; };
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

    auto stages = ShaderStagesCreator::fromModuleStringInput(
        *iglDev_, vertexShader, "main", "VS", fragmentShader, "main", name, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;

    const RenderPipelineDesc desc{
        .shaderStages = std::move(stages),
        .targetDesc = {.colorAttachments =
                           std::vector<RenderPipelineDesc::TargetDesc::ColorAttachment>(
                               numColorFormats, {.textureFormat = TextureFormat::RGBA_UNorm8})},
        .debugName = genNameHandle(name),
    };

    auto pipeline = iglDev_->createRenderPipeline(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    EXPECT_NE(pipeline, nullptr);
    return pipeline;
  }
};

TEST_F(SubpassPipelineCacheVulkanTest, ClearThenLoadDoesNotCreateNewPipelines) {
  const auto att0 = createColorTexture(TextureDesc::TextureUsageBits::Attachment |
                                       TextureDesc::TextureUsageBits::Sampled);
  const auto att1 = createColorTexture(TextureDesc::TextureUsageBits::Attachment |
                                       TextureDesc::TextureUsageBits::InputAttachment);
  const auto att2 = createColorTexture(TextureDesc::TextureUsageBits::Attachment |
                                       TextureDesc::TextureUsageBits::InputAttachment);
  const auto fb = createFramebuffer({att0, att1, att2});
  ASSERT_NE(fb, nullptr);

  const char* fs0 = R"(#version 460
layout(location = 1) out vec4 outA;
layout(location = 2) out vec4 outB;
void main() {
  outA = vec4(0.5, 0.0, 0.0, 1.0);
  outB = vec4(0.0, 0.5, 0.0, 1.0);
})";

  const char* fs1 = R"(#version 460
layout(input_attachment_index = 1, set = 2, binding = 17) uniform subpassInput inA;
layout(input_attachment_index = 2, set = 2, binding = 18) uniform subpassInput inB;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = subpassLoad(inA) + subpassLoad(inB);
})";

  auto p0 = createPipeline(fs0, 3, "cache_sp0");
  auto p1 = createPipeline(fs1, 3, "cache_sp1");

  // The frames differ only in the load action of attachment 0
  const auto makeRenderPass = [](LoadAction loadAction) {
    return RenderPassDesc{
        .colorAttachments = {{.loadAction = loadAction, .storeAction = StoreAction::Store},
                             {.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::DontCare},
                             {.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::DontCare}},
        .subpasses = {{.colorAttachmentMask = 0b110, .inputAttachmentMask = 0},
                      {.colorAttachmentMask = 0b001, .inputAttachmentMask = 0b110}},
    };
  };

  // Frame 1: Clear
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  auto encoder = cmdBuf->createRenderCommandEncoder(makeRenderPass(LoadAction::Clear), fb, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);

  encoder->bindRenderPipelineState(p0);
  encoder->draw(3);
  encoder->nextSubpass();
  encoder->bindRenderPipelineState(p1);
  encoder->draw(3);
  encoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  const uint32_t numPipelinesAfterFrame1 = vulkan::VulkanPipelineBuilder::getNumPipelinesCreated();

  // Frame 2: Load (should not create new pipelines)
  cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  encoder = cmdBuf->createRenderCommandEncoder(makeRenderPass(LoadAction::Load), fb, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);

  encoder->bindRenderPipelineState(p0);
  encoder->draw(3);
  encoder->nextSubpass();
  encoder->bindRenderPipelineState(p1);
  encoder->draw(3);
  encoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  const uint32_t numPipelinesAfterFrame2 = vulkan::VulkanPipelineBuilder::getNumPipelinesCreated();

  // No new pipelines should be created
  EXPECT_EQ(numPipelinesAfterFrame1, numPipelinesAfterFrame2);
}

TEST_F(SubpassPipelineCacheVulkanTest, PipelinePersistsAcrossNextSubpassCreatesOneExtraVariant) {
  const auto att0 = createColorTexture(TextureDesc::TextureUsageBits::Attachment |
                                       TextureDesc::TextureUsageBits::Sampled);
  const auto att1 = createColorTexture(TextureDesc::TextureUsageBits::Attachment |
                                       TextureDesc::TextureUsageBits::InputAttachment);
  const auto att2 = createColorTexture(TextureDesc::TextureUsageBits::Attachment |
                                       TextureDesc::TextureUsageBits::InputAttachment);
  const auto fb = createFramebuffer({att0, att1, att2});
  ASSERT_NE(fb, nullptr);

  // Pipeline that writes all 3 attachments (works in both subpasses)
  const char* fs = R"(#version 460
layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;
layout(location = 2) out vec4 out2;
void main() {
  out0 = vec4(1.0, 0.0, 0.0, 1.0);
  out1 = vec4(0.0, 1.0, 0.0, 1.0);
  out2 = vec4(0.0, 0.0, 1.0, 1.0);
})";

  auto pAll = createPipeline(fs, 3, "persist");

  const RenderPassDesc renderPass{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store},
                           {.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store},
                           {.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}},
      .subpasses = {{.colorAttachmentMask = 0b111, .inputAttachmentMask = 0},
                    {.colorAttachmentMask = 0b111, .inputAttachmentMask = 0}},
  };

  Result ret;
  const uint32_t numPipelinesBefore = vulkan::VulkanPipelineBuilder::getNumPipelinesCreated();

  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  auto encoder = cmdBuf->createRenderCommandEncoder(renderPass, fb, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);

  // Bind once and use across nextSubpass()
  encoder->bindRenderPipelineState(pAll);
  encoder->draw(3);
  encoder->nextSubpass();
  encoder->draw(3); // Same pipeline, different subpass
  encoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  const uint32_t numPipelinesAfter = vulkan::VulkanPipelineBuilder::getNumPipelinesCreated();

  // Should create exactly 2 VkPipelines: one for subpass 0, one for subpass 1
  EXPECT_EQ(numPipelinesAfter - numPipelinesBefore, 2u);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
