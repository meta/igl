/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "data/ShaderData.h"
#include "data/VertexIndexData.h"
#include "util/Common.h"
#if IGL_BACKEND_OPENGL
#include <igl/opengl/IContext.h>
#include <igl/opengl/PlatformDevice.h>
#endif // #IGL_BACKEND_OPENGL
#include <string>
#include <igl/CommandBuffer.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/VertexInputState.h>

namespace igl::tests {

// Use a 1x1 Framebuffer for this test
constexpr size_t kOffscreenRtWidth = 1;
constexpr size_t kOffscreenRtHeight = 1;

//
// FramebufferTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class FramebufferTest : public ::testing::Test {
 private:
 public:
  FramebufferTest() = default;

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
                                                   kOffscreenRtWidth,
                                                   kOffscreenRtHeight,
                                                   TextureDesc::TextureUsageBits::Sampled |
                                                       TextureDesc::TextureUsageBits::Attachment);

    auto depthFormat = TextureFormat::S8_UInt_Z32_UNorm;

    if (backend_ == util::kBackendVul) {
      depthFormat = TextureFormat::S8_UInt_Z24_UNorm;
    }

    TextureDesc depthTexDesc = TextureDesc::new2D(depthFormat,
                                                  kOffscreenRtWidth,
                                                  kOffscreenRtHeight,
                                                  TextureDesc::TextureUsageBits::Sampled |
                                                      TextureDesc::TextureUsageBits::Attachment);
    depthTexDesc.storage = ResourceStorage::Private;

    Result ret;
    offscreenTexture_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(offscreenTexture_ != nullptr);

    depthStencilTexture_ = iglDev_->createTexture(depthTexDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(depthStencilTexture_ != nullptr);

    // Create framebuffer using the offscreen texture
    const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = offscreenTexture_}},
                                          .depthAttachment = {.texture = depthStencilTexture_},
                                          .stencilAttachment = {.texture = depthStencilTexture_},
                                          .debugName = "test"};

    framebuffer_ = iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(framebuffer_ != nullptr);

    // Initialize render pass descriptor
    renderPass_.colorAttachments.resize(1);
    renderPass_.colorAttachments[0].loadAction = LoadAction::Clear;
    renderPass_.colorAttachments[0].storeAction = StoreAction::Store;
    renderPass_.colorAttachments[0].clearColor = {0.0, 0.0, 0.0, 1.0};

    renderPass_.depthAttachment.loadAction = LoadAction::Clear;
    renderPass_.depthAttachment.storeAction = StoreAction::Store;
    renderPass_.depthAttachment.clearDepth = 0.0;

    renderPass_.stencilAttachment.loadAction = LoadAction::Clear;
    renderPass_.stencilAttachment.storeAction = StoreAction::Store;
    renderPass_.stencilAttachment.clearStencil = 0;

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
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(vertexInputState_ != nullptr);

    // Initialize index buffer
    BufferDesc bufDesc;

    bufDesc.type = BufferDesc::BufferTypeBits::Index;
    bufDesc.data = data::vertex_index::kQuadInd.data();
    bufDesc.length = sizeof(data::vertex_index::kQuadInd);

    ib_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(ib_ != nullptr);

    // Initialize vertex and sampler buffers
    bufDesc.type = BufferDesc::BufferTypeBits::Vertex;
    bufDesc.data = data::vertex_index::kQuadVert.data();
    bufDesc.length = sizeof(data::vertex_index::kQuadVert);

    vb_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(vb_ != nullptr);

    bufDesc.type = BufferDesc::BufferTypeBits::Vertex;
    bufDesc.data = data::vertex_index::kQuadUv.data();
    bufDesc.length = sizeof(data::vertex_index::kQuadUv);

    uv_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(uv_ != nullptr);

    // Initialize sampler state
    const SamplerStateDesc samplerDesc;
    samp_ = iglDev_->createSamplerState(samplerDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(samp_ != nullptr);

    // Initialize Render Pipeline Descriptor, but leave the creation
    // to the individual tests in case further customization is required
    renderPipelineDesc_ = {
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = offscreenTexture_->getFormat()}},
                       .depthAttachmentFormat = depthStencilTexture_->getFormat(),
                       .stencilAttachmentFormat = depthStencilTexture_->getFormat()},
        .cullMode = CullMode::Disabled,
    };

    //
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
  std::shared_ptr<ITexture> depthStencilTexture_;

  std::shared_ptr<IFramebuffer> framebuffer_;

  std::shared_ptr<IShaderStages> shaderStages_;

  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::shared_ptr<IBuffer> vb_, uv_, ib_;

  std::shared_ptr<ISamplerState> samp_;

  RenderPipelineDesc renderPipelineDesc_;
};

//
// Framebuffer Clear Test
//
// This test exercises the FB clearing behavior. The expectation is color
// buffer will be cleared to the color specified, and there will be no
// leaked settings from previous render passes
//
TEST_F(FramebufferTest, Clear) {
  Result ret;
  std::shared_ptr<IRenderPipelineState> pipelineState;

  std::shared_ptr<IDepthStencilState> depthStencilState;
  const DepthStencilStateDesc desc{.isDepthWriteEnabled = true};

  const auto rangeDesc = TextureRangeDesc::new2D(0, 0, kOffscreenRtWidth, kOffscreenRtHeight);

  //----------------
  // Create Pipeline
  //----------------
  pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(pipelineState != nullptr);

  depthStencilState = iglDev_->createDepthStencilState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(depthStencilState != nullptr);

  //---------------------------------
  // Clear FB to {0.5, 0.5, 0.5, 0.5}
  //---------------------------------
  cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(cmdBuf_ != nullptr);

  renderPass_.colorAttachments[0].clearColor = {0.501f, 0.501f, 0.501f, 0.501f};

  renderPass_.depthAttachment.clearDepth = 0.501f;
  renderPass_.stencilAttachment.clearStencil = 128;

  auto cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);
  cmds->endEncoding();

  cmdQueue_->submit(*cmdBuf_);

  cmdBuf_->waitUntilCompleted();

  //----------------------
  // Read back framebuffer
  //----------------------
  const auto dimensions = framebuffer_->getColorAttachment(0)->getDimensions();
  ASSERT_EQ(dimensions.width, static_cast<uint32_t>(kOffscreenRtWidth));
  ASSERT_EQ(dimensions.height, static_cast<uint32_t>(kOffscreenRtHeight));

  auto pixels = std::vector<uint32_t>(kOffscreenRtWidth * kOffscreenRtWidth);
  auto pixelsDepth = std::vector<float>(kOffscreenRtWidth * kOffscreenRtWidth);
  auto pixelsStencil = std::vector<uint8_t>(kOffscreenRtWidth * kOffscreenRtWidth);

  framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);
  ASSERT_EQ(pixels[0], 0x80808080);

#if !IGL_PLATFORM_IOS_SIMULATOR
  // IOS Simulator has specific restrictions on depth/stencil formats. It has to
  // be created with ".private" storage mode. This means we can't read it back
  // and verify the contents.
  // https://developer.apple.com/documentation/metal/developing_metal_apps_that_run_in_simulator
  framebuffer_->copyBytesDepthAttachment(*cmdQueue_, pixelsDepth.data(), rangeDesc);
  // ASSERT_EQ(pixels_depth[0], 0x80808080);

  // TODO: copyBytesStencilAttachment is not functioning property under Metal/Vulkan
  // due to unimplemented blitting
  // Refer to igl/metal/Framebuffer.mm
  framebuffer_->copyBytesStencilAttachment(*cmdQueue_, pixelsStencil.data(), rangeDesc);
  // ASSERT_EQ(pixels_stencil[0], 0x80808080);
#endif

  //-------------------------------------------------------------------------
  // Clear FB to {0, 0, 0, 0}, but this time bind a pipelineState, disable
  // the colorWriteMasks, and do a no-op draw
  //-------------------------------------------------------------------------
  cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(cmdBuf_ != nullptr);

  renderPipelineDesc_.targetDesc.colorAttachments[0].colorWriteMask = kColorWriteBitsDisabled;
  pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(pipelineState != nullptr);

  renderPass_.colorAttachments[0].clearColor = {0, 0, 0, 0};

  renderPass_.depthAttachment.clearDepth = 0.0;
  renderPass_.stencilAttachment.clearStencil = 0;

  cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);
  cmds->bindRenderPipelineState(pipelineState);
  cmds->bindDepthStencilState(depthStencilState);
  cmds->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmds->bindVertexBuffer(data::shader::kSimpleUvIndex, *uv_);
  cmds->bindIndexBuffer(*ib_, IndexFormat::UInt16);
  cmds->drawIndexed(0); // draw 0 indices
  cmds->endEncoding();

  cmdQueue_->submit(*cmdBuf_);

  cmdBuf_->waitUntilCompleted();

  //----------------------
  // Read back framebuffer
  //----------------------
  pixels = std::vector<uint32_t>(kOffscreenRtWidth * kOffscreenRtWidth);

  framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);
  ASSERT_EQ(pixels[0], 0);

#if !IGL_PLATFORM_IOS_SIMULATOR
  // IOS Simulator has specific restrictions on depth/stencil formats. It has to
  // be created with ".private" storage mode. This means we can't read it back
  // and verify the contents.
  // https://developer.apple.com/documentation/metal/developing_metal_apps_that_run_in_simulator
  framebuffer_->copyBytesDepthAttachment(*cmdQueue_, pixelsDepth.data(), rangeDesc);
  // ASSERT_EQ(pixels_depth[0], 0);

  // TODO: copyBytesStencilAttachment is not functioning property under Metal/Vulkan
  // due to unimplemented blitting
  // Refer to igl/metal/Framebuffer.mm
  framebuffer_->copyBytesStencilAttachment(*cmdQueue_, pixelsStencil.data(), rangeDesc);
  // ASSERT_EQ(pixels_stencil[0], 0);
#endif

  //-------------------------------------------------------------------------
  // Clear FB to {0.5, 0.5, 0.5, 0.5} again. We should not be impacted by the
  // colorWriteMask setting from the render pass above
  //-------------------------------------------------------------------------
  cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(cmdBuf_ != nullptr);

  renderPass_.colorAttachments[0].clearColor = {0.501f, 0.501f, 0.501f, 0.501f};

  cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);
  cmds->endEncoding();

  cmdQueue_->submit(*cmdBuf_);

  cmdBuf_->waitUntilCompleted();

  //----------------------
  // Read back framebuffer
  //----------------------
  pixels = std::vector<uint32_t>(kOffscreenRtWidth * kOffscreenRtWidth);

  framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);
  ASSERT_EQ(pixels[0], 0x80808080);
}

//
// Framebuffer Blit Test
//
// This test exercises OGL platform device's blitFramebufferColor API.
// We are including this here rather than making it OGL-specific to also test
// the device's getPlatformDevice() function, which should return nullptr
// on Metal
//
#if IGL_BACKEND_OPENGL
TEST_F(FramebufferTest, blitFramebufferColor) {
  auto* platformDevice = iglDev_->getPlatformDevice<opengl::PlatformDevice>();
  if (platformDevice) {
    ASSERT_TRUE(backend_ == util::kBackendOgl);

    /* Bootcamper: Your Code here
     * At the high level here are the steps:
     *  1. Create second IFramebuffer, call it framebuffer2 look at set up to
     *     see how to do this:
     *        a. create another texture
     *        b. attach it to a FB descriptor
     *        c. create the framebuffer object
     *  2. clear framebuffer_, and make sure it is cleared. See the Clear test
     *     above to see how to do this
     *  3. clear framebuffer2 to a different color than framebuffer_, and make
     *     sure it is cleared
     *  4. call platforDevice->blitFramebufferColor() from framebuffer2 to
     *     framebuffer_
     *  5. Verify that framebuffer_ now has the same color as framebuffer2
     */

    Result ret;

    //-----------------------------------------
    // Create an offscreen texture to render to
    //-----------------------------------------
    const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                   kOffscreenRtWidth,
                                                   kOffscreenRtHeight,
                                                   TextureDesc::TextureUsageBits::Sampled |
                                                       TextureDesc::TextureUsageBits::Attachment);

    const std::shared_ptr<ITexture> offscreenTexture2 = iglDev_->createTexture(texDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(offscreenTexture2 != nullptr);

    //-------------------------------------------------------------
    // Create second IFramebuffer framebuffer2 by offscreenTexture2
    //-------------------------------------------------------------
    const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = offscreenTexture2}}};
    const std::shared_ptr<IFramebuffer> framebuffer2 =
        iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(framebuffer2 != nullptr);

    //---------------------------------
    // Clear FB to {0.5, 0.5, 0.5, 0.5}
    //---------------------------------
    const auto rangeDesc = TextureRangeDesc::new2D(0, 0, kOffscreenRtWidth, kOffscreenRtHeight);

    cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(cmdBuf_ != nullptr);

    renderPass_.colorAttachments[0].clearColor = {0.501f, 0.501f, 0.501f, 0.501f};

    auto cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);
    cmds->endEncoding();

    cmdQueue_->submit(*cmdBuf_);

    cmdBuf_->waitUntilCompleted();

    //----------------------
    // Read back framebuffer
    //----------------------
    const auto dimensions = framebuffer_->getColorAttachment(0)->getDimensions();
    ASSERT_EQ(dimensions.width, static_cast<uint32_t>(kOffscreenRtWidth));
    ASSERT_EQ(dimensions.height, static_cast<uint32_t>(kOffscreenRtHeight));

    const auto dimensions2 = framebuffer2->getColorAttachment(0)->getDimensions();
    ASSERT_EQ(dimensions2.width, static_cast<uint32_t>(kOffscreenRtWidth));
    ASSERT_EQ(dimensions2.height, static_cast<uint32_t>(kOffscreenRtHeight));

    auto pixels = std::vector<uint32_t>(kOffscreenRtWidth * kOffscreenRtWidth);

    framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);
    ASSERT_EQ(pixels[0], 0x80808080);

    //-------------------------------------------------------------------------
    // Clear FB2 to {0, 0, 0, 0}
    //-------------------------------------------------------------------------
    cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(cmdBuf_ != nullptr);

    renderPass_.colorAttachments[0].clearColor = {0, 0, 0, 0};

    cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer2);
    cmds->endEncoding();

    cmdQueue_->submit(*cmdBuf_);

    cmdBuf_->waitUntilCompleted();

    //-----------------------
    // Read back framebuffer2
    //-----------------------
    auto pixels2 = std::vector<uint32_t>(kOffscreenRtWidth * kOffscreenRtWidth);

    framebuffer2->copyBytesColorAttachment(*cmdQueue_, 0, pixels2.data(), rangeDesc);
    ASSERT_EQ(pixels2[0], 0);

    if (platformDevice->getContext().deviceFeatures().hasInternalFeature(
            opengl::InternalFeatures::FramebufferBlit)) {
      //--------------------------------------------------------------
      // Call blitFramebuffer() from framebuffer_ to framebuffer2
      //--------------------------------------------------------------
      platformDevice->blitFramebuffer(framebuffer_,
                                      0,
                                      0,
                                      kOffscreenRtWidth,
                                      kOffscreenRtWidth,
                                      framebuffer2,
                                      0,
                                      0,
                                      kOffscreenRtWidth,
                                      kOffscreenRtWidth,
                                      GL_COLOR_BUFFER_BIT,
                                      &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

      //-------------------------------------------------------------
      // Read back framebuffer_ & framebuffer2, and examine if equal.
      //-------------------------------------------------------------
      framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);
      framebuffer2->copyBytesColorAttachment(*cmdQueue_, 0, pixels2.data(), rangeDesc);

      ASSERT_EQ(pixels[0], pixels2[0]);
      ASSERT_EQ(pixels2[0], 0x80808080);
    }
  } else {
    ASSERT_TRUE(backend_ != util::kBackendOgl);
  }
}
#endif // IGL_BACKEND_OPENGL

//
// Framebuffer Drawable Unbind Test
//
// This test checks that when updateDrawable is called with nullptr, the
// color attachment is no longer bound. It also checks that draw counts
// are properly updated when unbind and rebinding a drawable
//
TEST_F(FramebufferTest, DrawableUnbind) {
  // Currently the drawable is always bound to index 0
  auto colorAttachment = framebuffer_->getColorAttachment(0);
  auto numOfAttachments = framebuffer_->getColorAttachmentIndices().size();
  auto depthAttachment = framebuffer_->getDepthAttachment();
  auto stencilAttachment = framebuffer_->getStencilAttachment();

  ASSERT_TRUE(colorAttachment != nullptr);
  ASSERT_EQ(numOfAttachments, 1);
  ASSERT_TRUE(depthAttachment != nullptr);
  ASSERT_TRUE(stencilAttachment != nullptr);

  framebuffer_->updateDrawable(nullptr);
  numOfAttachments = framebuffer_->getColorAttachmentIndices().size();

  ASSERT_TRUE(framebuffer_->getColorAttachment(0) == nullptr);
  ASSERT_EQ(numOfAttachments, 0);

  // Restore framebuffer_ to its original state
  framebuffer_->updateDrawable(colorAttachment);
  numOfAttachments = framebuffer_->getColorAttachmentIndices().size();

  ASSERT_EQ(numOfAttachments, 1);
}

//
// Framebuffer Drawable Bind Count Test
//
// This test checks that when updateDrawable is called repeatedly with
// the same texture, the number of attachment stays the same.
//
TEST_F(FramebufferTest, DrawableBindCount) {
  // Currently the drawable is always bound to index 0
  auto colorAttachment = framebuffer_->getColorAttachment(0);
  auto numOfAttachments = framebuffer_->getColorAttachmentIndices().size();

  ASSERT_TRUE(colorAttachment != nullptr);
  ASSERT_EQ(numOfAttachments, 1);

  // Rebind the same texture, numAttachment should not change
  framebuffer_->updateDrawable(colorAttachment);
  framebuffer_->updateDrawable(colorAttachment);
  framebuffer_->updateDrawable(colorAttachment);
  numOfAttachments = framebuffer_->getColorAttachmentIndices().size();

  ASSERT_EQ(numOfAttachments, 1);

  // Create another texture
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 kOffscreenRtWidth,
                                                 kOffscreenRtHeight,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Attachment);

  Result ret;
  auto newTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(newTex != nullptr);

  // Updating one texture for another, numAttachment should not change
  framebuffer_->updateDrawable(newTex);
  numOfAttachments = framebuffer_->getColorAttachmentIndices().size();

  ASSERT_EQ(numOfAttachments, 1);

  // Restore framebuffer_ to its original state
  framebuffer_->updateDrawable(colorAttachment);
  numOfAttachments = framebuffer_->getColorAttachmentIndices().size();

  ASSERT_EQ(numOfAttachments, 1);
}

//
// Framebuffer Update Drawable Test With Depth And Stencil Attachment
//
// This test checks that updateDrawable can be called to bind and unbind depth and stencil
// attachments.
//
TEST_F(FramebufferTest, UpdateDrawableWithDepthAndStencilTest) {
  // Currently the drawable is always bound to index 0
  auto colorAttachment = framebuffer_->getColorAttachment(0);
  auto depthAttachment = framebuffer_->getDepthAttachment();
  auto stencilAttachment = framebuffer_->getStencilAttachment();

  ASSERT_EQ(framebuffer_->getColorAttachment(0), colorAttachment);
  ASSERT_EQ(framebuffer_->getDepthAttachment(), depthAttachment);
  ASSERT_EQ(framebuffer_->getStencilAttachment(), stencilAttachment);

  framebuffer_->updateDrawable(nullptr);

  ASSERT_EQ(framebuffer_->getColorAttachment(0), nullptr);
  ASSERT_EQ(framebuffer_->getDepthAttachment(), depthAttachment);
  ASSERT_EQ(framebuffer_->getStencilAttachment(), stencilAttachment);

  framebuffer_->updateDrawable(colorAttachment);

  ASSERT_EQ(framebuffer_->getColorAttachment(0), colorAttachment);
  ASSERT_EQ(framebuffer_->getDepthAttachment(), depthAttachment);
  ASSERT_EQ(framebuffer_->getStencilAttachment(), stencilAttachment);

  framebuffer_->updateDrawable(SurfaceTextures{.color = colorAttachment, .depth = nullptr});

  ASSERT_EQ(framebuffer_->getColorAttachment(0), colorAttachment);
  ASSERT_EQ(framebuffer_->getDepthAttachment(), nullptr);
  ASSERT_EQ(framebuffer_->getStencilAttachment(), nullptr);

  framebuffer_->updateDrawable(SurfaceTextures{.color = colorAttachment, .depth = depthAttachment});

  ASSERT_EQ(framebuffer_->getColorAttachment(0), colorAttachment);
  ASSERT_EQ(framebuffer_->getDepthAttachment(), depthAttachment);
  ASSERT_EQ(framebuffer_->getStencilAttachment(), stencilAttachment);

  framebuffer_->updateDrawable(SurfaceTextures{.color = nullptr, .depth = nullptr});

  ASSERT_EQ(framebuffer_->getColorAttachment(0), nullptr);
  ASSERT_EQ(framebuffer_->getDepthAttachment(), nullptr);
  ASSERT_EQ(framebuffer_->getStencilAttachment(), nullptr);

  framebuffer_->updateDrawable(SurfaceTextures{.color = colorAttachment, .depth = depthAttachment});

  ASSERT_EQ(framebuffer_->getColorAttachment(0), colorAttachment);
  ASSERT_EQ(framebuffer_->getDepthAttachment(), depthAttachment);
  ASSERT_EQ(framebuffer_->getStencilAttachment(), stencilAttachment);
}

TEST_F(FramebufferTest, GetColorAttachmentTest) {
  const bool backendOpenGL = iglDev_->getBackendType() == BackendType::OpenGL;

  if (backendOpenGL) {
#if IGL_BACKEND_OPENGL
    if (!iglDev_->getPlatformDevice<opengl::PlatformDevice>()
             ->getContext()
             .deviceFeatures()
             .hasInternalFeature(opengl::InternalFeatures::PackRowLength)) {
      GTEST_SKIP() << "Framebuffer PackRowLength is not supported";
    }
#endif
  }

  // Create a texture to be used as color attachment
  const int textureWidth = 3;
  const int textureHeight = 2;
  const int channelCount = 4;
  const int channelSize = sizeof(uint8_t);
  const int textureElementPerRow = textureWidth * channelCount;
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 textureWidth,
                                                 textureHeight,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Attachment);

  Result ret;
  auto outputTexture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(outputTexture != nullptr);

  // Create framebuffer using the texture
  const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = outputTexture}}};
  framebuffer_ = iglDev_->createFramebuffer(framebufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(framebuffer_ != nullptr);

  //----------------
  // Create Pipeline
  //----------------
  cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(cmdBuf_ != nullptr);

  renderPass_.colorAttachments[0].clearColor = {0.501f, 0.501f, 0.501f, 0.501f};

  auto cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);
  cmds->endEncoding();

  cmdQueue_->submit(*cmdBuf_);

  cmdBuf_->waitUntilCompleted();

  //----------------------
  // Read back framebuffer
  //----------------------
  const auto dimensions = framebuffer_->getColorAttachment(0)->getDimensions();
  ASSERT_EQ(dimensions.width, static_cast<uint32_t>(textureWidth));
  ASSERT_EQ(dimensions.height, static_cast<uint32_t>(textureHeight));

  const int outputImageWidth = textureWidth + 2;
  const int outputImageHeight = textureHeight;
  const int outputElementPerRow = outputImageWidth * channelCount;
  auto pixels = std::vector<uint8_t>(static_cast<size_t>(outputElementPerRow * textureHeight), 0);

  const auto rangeDesc = TextureRangeDesc::new2D(0, 0, textureWidth, textureHeight);
  framebuffer_->copyBytesColorAttachment(*cmdQueue_,
                                         0,
                                         pixels.data(),
                                         rangeDesc,
                                         static_cast<size_t>(outputElementPerRow) * channelSize);

  for (int i = 0; i < outputImageHeight; i++) {
    for (int j = 0; j < outputElementPerRow; j++) {
      if (j < textureElementPerRow) {
        EXPECT_EQ(pixels[i * outputElementPerRow + j], 128);
      } else {
        EXPECT_EQ(pixels[i * outputElementPerRow + j], 0);
      }
    }
  }
}

} // namespace igl::tests
