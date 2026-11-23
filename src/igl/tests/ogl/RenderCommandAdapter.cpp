/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/RenderCommandAdapter.h>

#include "../data/ShaderData.h"
#include "../data/VertexIndexData.h"
#include "../util/Common.h"

#include <cstring>
#include <igl/CommandBuffer.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/VertexInputState.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

#define OFFSCREEN_TEX_WIDTH 2
#define OFFSCREEN_TEX_HEIGHT 2

//
// RenderCommandAdapterOGLTest
//
// Tests for the OpenGL RenderCommandAdapter.
//
class RenderCommandAdapterOGLTest : public ::testing::Test {
 public:
  RenderCommandAdapterOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();

    Result ret;

    // Create an offscreen texture to render to
    const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                   OFFSCREEN_TEX_WIDTH,
                                                   OFFSCREEN_TEX_HEIGHT,
                                                   TextureDesc::TextureUsageBits::Sampled |
                                                       TextureDesc::TextureUsageBits::Attachment);
    offscreenTexture_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_NE(offscreenTexture_, nullptr);

    // Create framebuffer
    const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = offscreenTexture_}}};
    framebuffer_ = iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_NE(framebuffer_, nullptr);

    // Initialize render pass descriptor
    renderPass_ = {
        .colorAttachments = {{.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::Store,
                              .clearColor = {0.0, 0.0, 0.0, 1.0}}},
    };

    // Initialize shader stages
    std::unique_ptr<IShaderStages> stages;
    igl::tests::util::createSimpleShaderStages(iglDev_, stages);
    shaderStages_ = std::move(stages);

    // Initialize vertex input state
    // numAttributes has to equal numInputBindings when using more than one buffer
    const VertexInputStateDesc inputDesc{
        .numAttributes = 2,
        .attributes = {{.bufferIndex = data::shader::kSimplePosIndex,
                        .format = VertexAttributeFormat::Float4,
                        .offset = 0,
                        .name = std::string(data::shader::kSimplePos),
                        .location = 0},
                       {.bufferIndex = data::shader::kSimpleUvIndex,
                        .format = VertexAttributeFormat::Float2,
                        .offset = 0,
                        .name = std::string(data::shader::kSimpleUv),
                        .location = 1}},
        .numInputBindings = 2,
        .inputBindings = {{.stride = sizeof(float) * 4}, {.stride = sizeof(float) * 2}},
    };

    vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(vertexInputState_, nullptr);

    // Create vertex buffer
    const BufferDesc vbDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                            .data = data::vertex_index::kQuadVert.data(),
                            .length = sizeof(data::vertex_index::kQuadVert)};
    vb_ = iglDev_->createBuffer(vbDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    // Create UV buffer
    const BufferDesc uvDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                            .data = data::vertex_index::kQuadUv.data(),
                            .length = sizeof(data::vertex_index::kQuadUv)};
    uvb_ = iglDev_->createBuffer(uvDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    // Create index buffer
    const BufferDesc ibDesc{.type = BufferDesc::BufferTypeBits::Index,
                            .data = data::vertex_index::kQuadInd.data(),
                            .length = sizeof(data::vertex_index::kQuadInd)};
    ib_ = iglDev_->createBuffer(ibDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    // Initialize Render Pipeline Descriptor
    renderPipelineDesc_ = {
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = offscreenTexture_->getFormat()}}},
        .cullMode = CullMode::Disabled,
    };

    pipelineState_ = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(pipelineState_, nullptr);

    // Create a simple 2x2 input texture (all white)
    const TextureDesc inputTexDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                        OFFSCREEN_TEX_WIDTH,
                                                        OFFSCREEN_TEX_HEIGHT,
                                                        TextureDesc::TextureUsageBits::Sampled);
    inputTexture_ = iglDev_->createTexture(inputTexDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_NE(inputTexture_, nullptr);

    // Upload white pixel data
    const uint32_t whitePixels[] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    inputTexture_->upload(TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT),
                          whitePixels);

    // Create sampler
    const SamplerStateDesc samplerDesc{.minFilter = SamplerMinMagFilter::Nearest,
                                       .magFilter = SamplerMinMagFilter::Nearest};
    sampler_ = iglDev_->createSamplerState(samplerDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;

  RenderPassDesc renderPass_;
  std::shared_ptr<ITexture> offscreenTexture_;
  std::shared_ptr<ITexture> inputTexture_;
  std::shared_ptr<IFramebuffer> framebuffer_;
  std::shared_ptr<IShaderStages> shaderStages_;
  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::shared_ptr<IRenderPipelineState> pipelineState_;
  std::shared_ptr<ISamplerState> sampler_;
  std::unique_ptr<IBuffer> vb_;
  std::unique_ptr<IBuffer> uvb_;
  std::unique_ptr<IBuffer> ib_;

  RenderPipelineDesc renderPipelineDesc_;
};

//
// DrawArrays
//
// Render a full-screen quad using drawArrays, read pixels, verify non-zero output.
//
TEST_F(RenderCommandAdapterOGLTest, DrawArrays) {
  Result ret;
  CommandBufferDesc cbDesc;
  auto cmdBuf = cmdQueue_->createCommandBuffer(cbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass_, framebuffer_);
  ASSERT_NE(cmdEncoder, nullptr);

  cmdEncoder->bindRenderPipelineState(pipelineState_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimpleUvIndex, *uvb_);
  cmdEncoder->bindTexture(0, BindTarget::kFragment, inputTexture_.get());
  cmdEncoder->bindSamplerState(0, BindTarget::kFragment, sampler_.get());

  cmdEncoder->draw(4);
  cmdEncoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);

  // Read back pixels
  std::array<uint32_t, OFFSCREEN_TEX_WIDTH * OFFSCREEN_TEX_HEIGHT> pixels{};
  framebuffer_->copyBytesColorAttachment(
      *cmdQueue_,
      0,
      pixels.data(),
      TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT));

  // At least one pixel should be non-zero (we rendered white on black)
  bool anyNonZero = false;
  for (auto px : pixels) {
    if (px != 0) {
      anyNonZero = true;
      break;
    }
  }
  ASSERT_TRUE(anyNonZero);
}

//
// DrawElements
//
// Render using indexed geometry.
//
TEST_F(RenderCommandAdapterOGLTest, DrawElements) {
  Result ret;
  CommandBufferDesc cbDesc;
  auto cmdBuf = cmdQueue_->createCommandBuffer(cbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass_, framebuffer_);
  ASSERT_NE(cmdEncoder, nullptr);

  cmdEncoder->bindRenderPipelineState(pipelineState_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimpleUvIndex, *uvb_);
  cmdEncoder->bindTexture(0, BindTarget::kFragment, inputTexture_.get());
  cmdEncoder->bindSamplerState(0, BindTarget::kFragment, sampler_.get());
  cmdEncoder->bindIndexBuffer(*ib_, IndexFormat::UInt16);

  cmdEncoder->drawIndexed(6);
  cmdEncoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);

  // Read back pixels
  std::array<uint32_t, OFFSCREEN_TEX_WIDTH * OFFSCREEN_TEX_HEIGHT> pixels{};
  framebuffer_->copyBytesColorAttachment(
      *cmdQueue_,
      0,
      pixels.data(),
      TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT));

  // All pixels should be non-zero (white), since we rendered a full-screen quad
  for (auto px : pixels) {
    ASSERT_NE(px, 0u);
  }
}

//
// DrawArraysInstanced
//
// Render multiple instances using drawArrays.
//
TEST_F(RenderCommandAdapterOGLTest, DrawArraysInstanced) {
  if (!iglDev_->hasFeature(DeviceFeatures::DrawInstanced)) {
    GTEST_SKIP() << "DrawInstanced not supported";
  }

  Result ret;
  CommandBufferDesc cbDesc;
  auto cmdBuf = cmdQueue_->createCommandBuffer(cbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass_, framebuffer_);
  ASSERT_NE(cmdEncoder, nullptr);

  cmdEncoder->bindRenderPipelineState(pipelineState_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimpleUvIndex, *uvb_);
  cmdEncoder->bindTexture(0, BindTarget::kFragment, inputTexture_.get());
  cmdEncoder->bindSamplerState(0, BindTarget::kFragment, sampler_.get());

  // Draw 2 instances of the same triangle strip
  cmdEncoder->draw(4, 2);
  cmdEncoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);

  // Verify no GL errors
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// DrawElementsInstanced
//
// Render multiple instances using indexed geometry.
//
TEST_F(RenderCommandAdapterOGLTest, DrawElementsInstanced) {
  if (!iglDev_->hasFeature(DeviceFeatures::DrawInstanced)) {
    GTEST_SKIP() << "DrawInstanced not supported";
  }

  Result ret;
  CommandBufferDesc cbDesc;
  auto cmdBuf = cmdQueue_->createCommandBuffer(cbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass_, framebuffer_);
  ASSERT_NE(cmdEncoder, nullptr);

  cmdEncoder->bindRenderPipelineState(pipelineState_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimpleUvIndex, *uvb_);
  cmdEncoder->bindTexture(0, BindTarget::kFragment, inputTexture_.get());
  cmdEncoder->bindSamplerState(0, BindTarget::kFragment, sampler_.get());
  cmdEncoder->bindIndexBuffer(*ib_, IndexFormat::UInt16);

  cmdEncoder->drawIndexed(6, 2);
  cmdEncoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);

  // Verify no GL errors
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// DrawCountIncrement
//
// Verify that getCurrentDrawCount() increments after draws.
//
TEST_F(RenderCommandAdapterOGLTest, DrawCountIncrement) {
  const size_t drawCountBefore = iglDev_->getCurrentDrawCount();

  Result ret;
  CommandBufferDesc cbDesc;
  auto cmdBuf = cmdQueue_->createCommandBuffer(cbDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  auto cmdEncoder = cmdBuf->createRenderCommandEncoder(renderPass_, framebuffer_);
  ASSERT_NE(cmdEncoder, nullptr);

  cmdEncoder->bindRenderPipelineState(pipelineState_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmdEncoder->bindVertexBuffer(data::shader::kSimpleUvIndex, *uvb_);
  cmdEncoder->bindTexture(0, BindTarget::kFragment, inputTexture_.get());
  cmdEncoder->bindSamplerState(0, BindTarget::kFragment, sampler_.get());
  cmdEncoder->bindIndexBuffer(*ib_, IndexFormat::UInt16);

  cmdEncoder->drawIndexed(6);
  cmdEncoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);

  const size_t drawCountAfter = iglDev_->getCurrentDrawCount();
  ASSERT_GT(drawCountAfter, drawCountBefore);
}

} // namespace igl::tests
