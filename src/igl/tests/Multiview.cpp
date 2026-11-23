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

#include <IGLU/managedUniformBuffer/ManagedUniformBuffer.h>
#include <glm/glm.hpp>
#include <string>

namespace igl::tests {

// Use a 1x1 Framebuffer for this test
constexpr size_t kOffScreenWidth = 1;
constexpr size_t kOffScreenHeight = 1;

using Colors = std::array<glm::vec4, 2>;

//
// MultiviewTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class MultiviewTest : public ::testing::Test {
 private:
 public:
  MultiviewTest() = default;

  std::shared_ptr<iglu::ManagedUniformBuffer> createVertexUniformBuffer(IDevice& device,
                                                                        Result* /*result*/) {
    std::shared_ptr<iglu::ManagedUniformBuffer> vertUniformBuffer = nullptr;

    const iglu::ManagedUniformBufferInfo ubInfo = {
        .index = 1,
        .length = sizeof(Colors),
        .uniforms =
            {
                {.name = "colors",
                 .location = -1,
                 .type = UniformType::Float4,
                 .numElements = 2,
                 .offset = 0,
                 .elementStride = sizeof(glm::vec4)},
            },
    };

    vertUniformBuffer = std::make_shared<iglu::ManagedUniformBuffer>(device, ubInfo);

    IGL_DEBUG_ASSERT(vertUniformBuffer->result.isOk());
    return vertUniformBuffer;
  }

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

    const std::vector<DeviceFeatures> requestedFeatures{DeviceFeatures::Multiview};

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    if (!iglDev_->hasFeature(DeviceFeatures::Multiview)) {
      GTEST_SKIP() << "Multiview is unsupported for this platform.";
    }
#if IGL_PLATFORM_WINDOWS || (IGL_PLATFORM_LINUX && !IGL_PLATFORM_LINUX_USE_EGL)
    if (iglDev_->getBackendType() == igl::BackendType::OpenGL) {
      GTEST_SKIP() << "Multiview is unsupported for this platform.";
      return;
    }
#endif

    // Create an offscreen texture to render to
    const TextureDesc texDesc = TextureDesc::new2DArray(
        TextureFormat::RGBA_UNorm8,
        kOffScreenWidth,
        kOffScreenHeight,
        2,
        TextureDesc::TextureUsageBits::Sampled | TextureDesc::TextureUsageBits::Attachment);

    auto depthFormat = TextureFormat::S8_UInt_Z32_UNorm;

#ifndef IGL_PLATFORM_MACOSX
    if (backend_ == util::kBackendVul) {
      depthFormat = TextureFormat::S8_UInt_Z24_UNorm;
    }
#endif // IGL_PLATFORM_MACOSX

    TextureDesc depthTexDesc = TextureDesc::new2DArray(depthFormat,
                                                       kOffScreenWidth,
                                                       kOffScreenHeight,
                                                       2,
                                                       TextureDesc::TextureUsageBits::Attachment);
    depthTexDesc.storage = ResourceStorage::Private;

    Result ret;
    offscreenTexture_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    ASSERT_NE(offscreenTexture_, nullptr);

    depthStencilTexture_ = iglDev_->createTexture(depthTexDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    ASSERT_NE(depthStencilTexture_, nullptr);

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

    // Initialize input to vertex shader
    VertexInputStateDesc inputDesc;

    inputDesc.attributes[0].format = VertexAttributeFormat::Float4;
    inputDesc.attributes[0].offset = 0;
    inputDesc.attributes[0].bufferIndex = data::shader::kSimplePosIndex;
    inputDesc.attributes[0].name = std::string(data::shader::kSimplePos);
    inputDesc.attributes[0].location = 0;
    inputDesc.inputBindings[0].stride = sizeof(float) * 4;

    // numAttributes has to equal to bindings when using more than 1 buffer
    inputDesc.numAttributes = inputDesc.numInputBindings = 1;

    vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(vertexInputState_, nullptr);

    // Initialize index buffer
    BufferDesc bufDesc;

    bufDesc.type = BufferDesc::BufferTypeBits::Index;
    bufDesc.data = data::vertex_index::kQuadInd.data();
    bufDesc.length = sizeof(data::vertex_index::kQuadInd);

    ib_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(ib_, nullptr);

    // Initialize vertex and sampler buffers
    bufDesc.type = BufferDesc::BufferTypeBits::Vertex;
    bufDesc.data = data::vertex_index::kQuadVert.data();
    bufDesc.length = sizeof(data::vertex_index::kQuadVert);

    vb_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(vb_, nullptr);

    // Initialize Render Pipeline Descriptor, but leave the creation
    // to the individual tests in case further customization is required
    renderPipelineDesc_ = {
        .vertexInputState = vertexInputState_,
        .targetDesc = {.colorAttachments = {{.textureFormat = offscreenTexture_->getFormat()}},
                       .depthAttachmentFormat = depthStencilTexture_->getFormat(),
                       .stencilAttachmentFormat = depthStencilTexture_->getFormat()},
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
  const std::string backend_ = IGL_BACKEND_TYPE;

  RenderPassDesc renderPass_;
  std::shared_ptr<ITexture> offscreenTexture_;
  std::shared_ptr<ITexture> depthStencilTexture_;

  std::shared_ptr<IFramebuffer> framebuffer_;

  std::shared_ptr<IShaderStages> shaderStages_;

  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::shared_ptr<IBuffer> vb_, ib_;

  RenderPipelineDesc renderPipelineDesc_;

  Colors colors_;
};

TEST_F(MultiviewTest, FramebufferMode) {
  FramebufferDesc desc;
  EXPECT_TRUE(desc.mode == FramebufferMode::Mono);

  desc.mode = FramebufferMode::Stereo;
  EXPECT_TRUE(desc.mode == FramebufferMode::Stereo);

  desc.mode = FramebufferMode::Multiview;
  EXPECT_TRUE(desc.mode == FramebufferMode::Multiview);
}

TEST_F(MultiviewTest, SinglePassStereo) {
  if (!iglDev_->hasFeature(DeviceFeatures::Multiview)) {
    GTEST_SKIP() << "Multiview is unsupported for this platform.";
  }

  std::unique_ptr<IShaderStages> stages;
  if (backend_ == util::kBackendOgl) {
    igl::tests::util::createShaderStages(iglDev_,
                                         data::shader::kOglSimpleVertShaderMultiviewEs3,
                                         data::shader::kShaderFunc,
                                         data::shader::kOglSimpleFragShaderMultiviewEs3,
                                         data::shader::kShaderFunc,
                                         stages);
  } else if (backend_ == util::kBackendVul) {
    igl::tests::util::createShaderStages(iglDev_,
                                         data::shader::kVulkanSimpleVertShaderMultiview,
                                         data::shader::kShaderFunc,
                                         data::shader::kVulkanSimpleFragShaderMultiview,
                                         data::shader::kShaderFunc,
                                         stages);
  }

  ASSERT_TRUE(stages);

  shaderStages_ = std::move(stages);
  renderPipelineDesc_.shaderStages = shaderStages_;

  FramebufferDesc framebufferDesc;
  framebufferDesc.mode = FramebufferMode::Stereo;

  framebufferDesc.colorAttachments[0].texture = offscreenTexture_;
  framebufferDesc.depthAttachment.texture = depthStencilTexture_;
  framebufferDesc.stencilAttachment.texture = depthStencilTexture_;

  Result ret;
  framebuffer_ = iglDev_->createFramebuffer(framebufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(framebuffer_, nullptr);

  Result result{};
  auto vertUniformBuffer = createVertexUniformBuffer(*iglDev_, &result);
  ASSERT_TRUE(result.isOk());

  colors_[0].r = 1.0f;
  colors_[0].g = 0.0f;
  colors_[0].b = 0.0f;
  colors_[0].a = 1.0f;

  colors_[1].r = 0.0f;
  colors_[1].g = 1.0f;
  colors_[1].b = 1.0f;
  colors_[1].a = 1.0f;

  *static_cast<Colors*>(vertUniformBuffer->getData()) = colors_;

  const auto pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipelineState, nullptr);

  const DepthStencilStateDesc desc{.isDepthWriteEnabled = true};
  const auto depthStencilState = iglDev_->createDepthStencilState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(depthStencilState, nullptr);

  cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cmdBuf_, nullptr);
  auto cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);

  cmds->bindRenderPipelineState(pipelineState);
  cmds->bindDepthStencilState(depthStencilState);

  cmds->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  vertUniformBuffer->bind(*iglDev_, *pipelineState, *cmds);

  cmds->bindIndexBuffer(*ib_, IndexFormat::UInt16);
  cmds->drawIndexed(6);

  cmds->endEncoding();
  cmdBuf_->present(framebuffer_->getColorAttachment(0));
  cmdQueue_->submit(*cmdBuf_);
  cmdBuf_->waitUntilCompleted();

  //----------------------
  // Read back framebuffer
  //----------------------
  const auto dimensions = framebuffer_->getColorAttachment(0)->getDimensions();
  EXPECT_EQ(dimensions.width, static_cast<uint32_t>(kOffScreenWidth));
  EXPECT_EQ(dimensions.height, static_cast<uint32_t>(kOffScreenHeight));

  auto pixels = std::vector<uint32_t>(kOffScreenWidth * kOffScreenHeight);
  auto rangeDesc = TextureRangeDesc::new2D(0, 0, kOffScreenWidth, kOffScreenHeight);

  framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);
  EXPECT_EQ(pixels[0], 0xff0000ff);

  rangeDesc.layer = 1;
  pixels[0] = 0;
  framebuffer_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), rangeDesc);
  EXPECT_EQ(pixels[0], 0xffffff00);
}
} // namespace igl::tests
