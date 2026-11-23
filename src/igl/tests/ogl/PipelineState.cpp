/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"

#include <utility>
#include <igl/CommandBuffer.h>
#include <igl/RenderPass.h>
#include <igl/VertexInputState.h>
#include <igl/opengl/RenderPipelineState.h>

namespace igl::tests {

// Use a 4x4 texture for this test
#define OFFSCREEN_TEX_WIDTH 4
#define OFFSCREEN_TEX_HEIGHT 4

//
// PipelineStateOGLTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class PipelineStateOGLTest : public ::testing::Test {
 private:
 public:
  PipelineStateOGLTest() = default;

  //
  // SetUp()
  //
  // This function sets up a render pass and a render pipeline descriptor
  // so it is ready to render a simple quad with an input texture to an
  // offscreen texture.
  //
  // The actual creation of the render pipeline state object is left
  // to each test so that tests can replace the default settings with
  // something more appropriate.
  //
  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);

    // Create an offscreen texture to render to
    const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                   OFFSCREEN_TEX_WIDTH,
                                                   OFFSCREEN_TEX_HEIGHT,
                                                   TextureDesc::TextureUsageBits::Sampled |
                                                       TextureDesc::TextureUsageBits::Attachment);

    Result ret;
    offscreenTexture_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(offscreenTexture_ != nullptr);

    // Create framebuffer using the offscreen texture
    const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = offscreenTexture_}}};
    framebuffer_ = iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(framebuffer_ != nullptr);

    // Initialize render pass descriptor
    renderPass_ = {
        .colorAttachments = {{.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::Store,
                              .clearColor = {0.0, 0.0, 0.0, 1.0}}},
    };

    // Initialize input to vertex shader
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
    ASSERT_TRUE(vertexInputState_ != nullptr);

    // Initialize Render Pipeline Descriptor, but leave the creation
    // to the individual tests in case further customization is required
    renderPipelineDesc_ = {
        .vertexInputState = vertexInputState_,
        .targetDesc = {.colorAttachments = {{.textureFormat = offscreenTexture_->getFormat(),
                                             .blendEnabled = true}}},
        .cullMode = CullMode::Disabled,
    };
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ICommandBuffer> cmdBuf_;
  CommandBufferDesc cbDesc_ = {};

  RenderPassDesc renderPass_;
  std::shared_ptr<ITexture> offscreenTexture_;
  std::shared_ptr<IFramebuffer> framebuffer_;

  std::shared_ptr<IShaderStages> shaderStages_;

  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::shared_ptr<IBuffer> fragmentParamBuffer_;

  RenderPipelineDesc renderPipelineDesc_;
};

//
// GetIndexByName
//
// This test exercises ShaderStages.getIndexByName() for attribute array
// buffer by going through the pipelineState object.
//
TEST_F(PipelineStateOGLTest, GetIndexByName) {
  Result ret;
  std::shared_ptr<IRenderPipelineState> pipelineState;

  //----------------
  // Create Shaders
  //----------------
  // Initialize shader stages
  std::unique_ptr<IShaderStages> stages;
  igl::tests::util::createSimpleShaderStages(iglDev_, stages);
  shaderStages_ = std::move(stages);

  renderPipelineDesc_.shaderStages = shaderStages_;

  //----------------
  // Create Pipeline
  //----------------
  pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(pipelineState != nullptr);

  // These should have a location because they are attributes in the simple shader
  int idx =
      pipelineState->getIndexByName(IGL_NAMEHANDLE(data::shader::kSimpleUv), ShaderStage::Fragment);
  ASSERT_NE(idx, -1);

  idx = pipelineState->getIndexByName(IGL_NAMEHANDLE(data::shader::kSimplePos),
                                      ShaderStage::Fragment);
  ASSERT_NE(idx, -1);
}

// Test static conversions from IGL ops to OGL ops
TEST_F(PipelineStateOGLTest, ConvertOps) {
  //----------------
  // BlendOp
  //----------------
  const std::vector<std::pair<BlendOp, GLenum>> inputAndExpectedBlendOp = {
      std::make_pair(BlendOp::Add, GL_FUNC_ADD),
      std::make_pair(BlendOp::Subtract, GL_FUNC_SUBTRACT),
      std::make_pair(BlendOp::ReverseSubtract, GL_FUNC_REVERSE_SUBTRACT),
      std::make_pair(BlendOp::Min, GL_MIN),
      std::make_pair(BlendOp::Max, GL_MAX),
  };

  for (const auto& ioPair : inputAndExpectedBlendOp) {
    ASSERT_EQ(igl::opengl::RenderPipelineState::convertBlendOp(ioPair.first), ioPair.second);
  }

  //----------------
  // BlendFactor
  //----------------
  const std::vector<std::pair<BlendFactor, GLenum>> inputAndExpectedBlendFactor = {
      std::make_pair(BlendFactor::Zero, GL_ZERO),
      std::make_pair(BlendFactor::One, GL_ONE),
      std::make_pair(BlendFactor::SrcColor, GL_SRC_COLOR),
      std::make_pair(BlendFactor::OneMinusSrcColor, GL_ONE_MINUS_SRC_COLOR),
      std::make_pair(BlendFactor::DstColor, GL_DST_COLOR),
      std::make_pair(BlendFactor::OneMinusDstColor, GL_ONE_MINUS_DST_COLOR),
      std::make_pair(BlendFactor::SrcAlpha, GL_SRC_ALPHA),
      std::make_pair(BlendFactor::OneMinusSrcAlpha, GL_ONE_MINUS_SRC_ALPHA),
      std::make_pair(BlendFactor::DstAlpha, GL_DST_ALPHA),
      std::make_pair(BlendFactor::OneMinusDstAlpha, GL_ONE_MINUS_DST_ALPHA),
      std::make_pair(BlendFactor::BlendColor, GL_CONSTANT_COLOR),
      std::make_pair(BlendFactor::OneMinusBlendColor, GL_ONE_MINUS_CONSTANT_COLOR),
      std::make_pair(BlendFactor::BlendAlpha, GL_CONSTANT_ALPHA),
      std::make_pair(BlendFactor::OneMinusBlendAlpha, GL_ONE_MINUS_CONSTANT_ALPHA),
      std::make_pair(BlendFactor::SrcAlphaSaturated, GL_SRC_ALPHA_SATURATE),
      // Unsupported values default to GL_ONE
      std::make_pair(BlendFactor::Src1Color, GL_ONE),
      std::make_pair(BlendFactor::OneMinusSrc1Color, GL_ONE),
      std::make_pair(BlendFactor::Src1Alpha, GL_ONE),
      std::make_pair(BlendFactor::OneMinusSrc1Alpha, GL_ONE),
  };

  for (const auto& ioPair : inputAndExpectedBlendFactor) {
    ASSERT_EQ(igl::opengl::RenderPipelineState::convertBlendFactor(ioPair.first), ioPair.second);
  }
}

} // namespace igl::tests
