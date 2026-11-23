/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "data/ShaderData.h"
#include "data/TextureData.h"
#include "data/VertexIndexData.h"
#include "util/Color.h"
#include "util/Common.h"

#include <cstdlib>
#include <string>
#include <igl/CommandBuffer.h>
#include <igl/NameHandle.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/VertexInputState.h>

namespace igl::tests {

//
// TexturesRGBBaseTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class TexturesRGBBaseTest : public ::testing::Test {
 private:
 public:
  TexturesRGBBaseTest() = default;

  //
  // SetUp()
  //
  // This function sets up a render pass and a graphics pipeline descriptor
  // so it is ready to render a simple quad with an input texture to an
  // offscreen texture.
  //
  // The actual creation of the graphics pipeline state object is left
  // to each test so that tests can replace the default settings with
  // something more appropriate.
  //
  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);

    // Create an offscreen texture to render to
    const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_SRGB,
                                                   offscreenTexWidth_,
                                                   offscreenTexHeight_,
                                                   TextureDesc::TextureUsageBits::Sampled |
                                                       TextureDesc::TextureUsageBits::Attachment);

    Result ret;
    offscreenTexture_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(offscreenTexture_ != nullptr);

    // Create framebuffer using the offscreen texture
    const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = offscreenTexture_}}};
    framebuffer_ = iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(framebuffer_ != nullptr);

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
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(vertexInputState_ != nullptr);

    // Initialize index buffer
    BufferDesc bufDesc;

    bufDesc.type = BufferDesc::BufferTypeBits::Index;
    bufDesc.data = data::vertex_index::kQuadInd.data();
    bufDesc.length = sizeof(data::vertex_index::kQuadInd);

    ib_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(ib_ != nullptr);

    // Initialize vertex and sampler buffers
    bufDesc.type = BufferDesc::BufferTypeBits::Vertex;
    bufDesc.data = data::vertex_index::kQuadVert.data();
    bufDesc.length = sizeof(data::vertex_index::kQuadVert);

    vb_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(vb_ != nullptr);

    bufDesc.type = BufferDesc::BufferTypeBits::Vertex;
    bufDesc.data = data::vertex_index::kQuadUv.data();
    bufDesc.length = sizeof(data::vertex_index::kQuadUv);

    uv_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(uv_ != nullptr);

    // Initialize sampler state
    const SamplerStateDesc samplerDesc;
    samp_ = iglDev_->createSamplerState(samplerDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(samp_ != nullptr);

    // Initialize Graphics Pipeline Descriptor, but leave the creation
    // to the individual tests in case further customization is required
    renderPipelineDesc_ = {
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = offscreenTexture_->getFormat()}}},
        .cullMode = CullMode::Disabled,
        .fragmentUnitSamplerMap = {{textureUnit_, IGL_NAMEHANDLE(data::shader::kSimpleSampler)}},
    };

// The sRGB hardware extension should decode and re-encode to exactly the same color values
// which is what this test is trying to test.
// It turns out that not everyone gets it 100% right.
// the platforms below, on OpenGL result with some colors
// have a + or - 1 to the original value which is most probably due to the
// the level of precision used to perform the arithmetic.
// The kTolerance is used to not only make this test not fail but also to document
// which platform gets it right and which platform gets it slightly wrong.
// Panther is external codename for Quest3s and Eureka is external codename for Quest3
#if IGL_PLATFORM_LINUX_SWIFTSHADER || defined(PANTHER_PLATFORM) || defined(EUREKA_PLATFORM)
    if (iglDev_->getBackendType() == BackendType::OpenGL) {
      tolerance_ = 1; // Swiftshader and quest 3(s) opengl is not accurate enough.
    }
#endif
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ICommandBuffer> cmdBuf_;
  CommandBufferDesc cbDesc_ = {};
  const std::string backend_ = IGL_BACKEND_TYPE;

  RenderPassDesc renderPass_;
  std::shared_ptr<ITexture> offscreenTexture_;
  std::shared_ptr<IFramebuffer> framebuffer_;

  // Currently it is left to individual tests to initialize this
  std::shared_ptr<ITexture> inputTexture_;

  std::shared_ptr<IShaderStages> shaderStages_;

  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::shared_ptr<IBuffer> vb_, uv_, ib_;

  std::shared_ptr<ISamplerState> samp_;

  RenderPipelineDesc renderPipelineDesc_;
  size_t textureUnit_ = 0;

  // Picking this just to match the texture we will use. If you use a different
  // size texture, then you will have to either create a new offscreenTexture_
  // and the framebuffer object in your test, so know exactly what the end result
  // would be after sampling
  size_t offscreenTexWidth_ = 2;
  size_t offscreenTexHeight_ = 2;

  uint8_t tolerance_ = 0; // some platforms aren't perfect and need some tolerance
};

class TexturesRGBSmallTest : public TexturesRGBBaseTest {
  void SetUp() override {
    TexturesRGBBaseTest::SetUp();
  }
};

class TexturesRGBBigTest : public TexturesRGBBaseTest {
  void SetUp() override {
    offscreenTexWidth_ = 4096;
    offscreenTexHeight_ = 4096;
    TexturesRGBBaseTest::SetUp();
  }
};
//
// isSRGB test
//
// This test checks whether the texture format can be detected as sRGB or not
//
TEST_F(TexturesRGBSmallTest, TextureisSRGB) {
  Result ret;
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_SRGB,
                                                 offscreenTexWidth_,
                                                 offscreenTexHeight_,
                                                 TextureDesc::TextureUsageBits::Sampled);
  inputTexture_ = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(inputTexture_->getProperties().isSRGB());

  const TextureDesc texDesc2 = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                  offscreenTexWidth_,
                                                  offscreenTexHeight_,
                                                  TextureDesc::TextureUsageBits::Sampled);
  inputTexture_ = iglDev_->createTexture(texDesc, &ret);
  ASSERT_FALSE(!inputTexture_->getProperties().isSRGB());
}

//
// TexturesRGB Passthrough Test
//
// This test uses a simple shader to copy the input texture to a same
// sized output texture (offscreenTexture_) and make sure colors are being preserved
//
TEST_F(TexturesRGBSmallTest, Passthrough) {
  Result ret;
  std::shared_ptr<IRenderPipelineState> pipelineState;

  //-------------------------------------
  // Create input texture and upload data
  //-------------------------------------
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_SRGB,
                                                 offscreenTexWidth_,
                                                 offscreenTexHeight_,
                                                 TextureDesc::TextureUsageBits::Sampled);
  inputTexture_ = iglDev_->createTexture(texDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(inputTexture_ != nullptr);

  const auto rangeDesc = TextureRangeDesc::new2D(0, 0, offscreenTexWidth_, offscreenTexHeight_);

  inputTexture_->upload(rangeDesc, data::texture::kTexRgba2x2.data());

  //----------------
  // Create Pipeline
  //----------------
  pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(pipelineState != nullptr);

  //-------
  // Render
  //-------
  cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(cmdBuf_ != nullptr);

  auto cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);
  cmds->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmds->bindVertexBuffer(data::shader::kSimpleUvIndex, *uv_);

  cmds->bindRenderPipelineState(pipelineState);

  cmds->bindTexture(textureUnit_, BindTarget::kFragment, inputTexture_.get());
  cmds->bindSamplerState(textureUnit_, BindTarget::kFragment, samp_.get());

  cmds->bindIndexBuffer(*ib_, IndexFormat::UInt16);
  cmds->drawIndexed(6);

  cmds->endEncoding();

  cmdQueue_->submit(*cmdBuf_);

  cmdBuf_->waitUntilCompleted();

  //----------------------
  // Read back framebuffer
  //----------------------
  auto pixels = std::vector<uint32_t>(offscreenTexWidth_ * offscreenTexHeight_);

  framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);

  //--------------------------------
  // Verify against original texture
  //--------------------------------
  for (size_t i = 0; i < offscreenTexWidth_ * offscreenTexHeight_; i++) {
    const util::SRgbColor currentColor(pixels[i]);
    const util::SRgbColor testColor(data::texture::kTexRgba2x2[i]);
    ASSERT_LE(std::abs(currentColor.r - testColor.r), tolerance_);
    ASSERT_LE(std::abs(currentColor.g - testColor.g), tolerance_);
    ASSERT_LE(std::abs(currentColor.b - testColor.b), tolerance_);
    ASSERT_LE(std::abs(currentColor.a - testColor.a), tolerance_);
  }
}

TEST_F(TexturesRGBBigTest, Passthrough) {
  Result ret;
  std::shared_ptr<IRenderPipelineState> pipelineState;

  //-------------------------------------
  // Create input texture and upload data
  //-------------------------------------
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_SRGB,
                                                 offscreenTexWidth_,
                                                 offscreenTexHeight_,
                                                 TextureDesc::TextureUsageBits::Sampled);
  inputTexture_ = iglDev_->createTexture(texDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(inputTexture_ != nullptr);

  const auto rangeDesc = TextureRangeDesc::new2D(0, 0, offscreenTexWidth_, offscreenTexHeight_);

  const size_t kAllColorsCount = 256ul * 256ul * 256ul;
  ASSERT_TRUE(kAllColorsCount == offscreenTexWidth_ * offscreenTexHeight_);
  std::vector<uint32_t> allColorsBuffer(offscreenTexWidth_ * offscreenTexHeight_);
  size_t index = 0;
  for (size_t r = 0; r < 256ul; ++r) {
    for (size_t g = 0; g < 256ul; ++g) {
      for (size_t b = 0; b < 256ul; ++b) {
        ASSERT_TRUE(index < kAllColorsCount);
        allColorsBuffer[index] = (r << 24) | (g << 16) | (b << 8) | 0xFF;
        ++index;
      }
    }
  }
  inputTexture_->upload(rangeDesc, allColorsBuffer.data());

  //----------------
  // Create Pipeline
  //----------------
  pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(pipelineState != nullptr);

  //-------
  // Render
  //-------
  cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(cmdBuf_ != nullptr);

  auto cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);
  cmds->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmds->bindVertexBuffer(data::shader::kSimpleUvIndex, *uv_);

  cmds->bindRenderPipelineState(pipelineState);

  cmds->bindTexture(textureUnit_, BindTarget::kFragment, inputTexture_.get());
  cmds->bindSamplerState(textureUnit_, BindTarget::kFragment, samp_.get());

  cmds->bindIndexBuffer(*ib_, IndexFormat::UInt16);
  cmds->drawIndexed(6);

  cmds->endEncoding();

  cmdQueue_->submit(*cmdBuf_);

  cmdBuf_->waitUntilCompleted();

  //----------------------
  // Read back framebuffer
  //----------------------
  auto pixels = std::vector<uint32_t>(offscreenTexWidth_ * offscreenTexHeight_);

  framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);

  //--------------------------------
  // Verify against original texture
  //--------------------------------
  for (size_t i = 0; i < offscreenTexWidth_ * offscreenTexHeight_; i++) {
    const util::SRgbColor currentColor(pixels[i]);
    const util::SRgbColor testColor(allColorsBuffer[i]);
    ASSERT_LE(std::abs(currentColor.r - testColor.r), tolerance_);
    ASSERT_LE(std::abs(currentColor.g - testColor.g), tolerance_);
    ASSERT_LE(std::abs(currentColor.b - testColor.b), tolerance_);
    ASSERT_LE(std::abs(currentColor.a - testColor.a), tolerance_);
  }
}

} // namespace igl::tests
