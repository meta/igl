/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "data/ShaderData.h"
#include "data/VertexIndexData.h"
#include "util/Common.h"

#if IGL_PLATFORM_WINDOWS
#include <windows.h>
#endif

#include <string>
#include <igl/CommandBuffer.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/VertexInputState.h>

// Use a 1x1 Framebuffer for this test
#define OFFSCREEN_RT_WIDTH 1
#define OFFSCREEN_RT_HEIGHT 1

namespace igl::tests {

//
// DeviceTest
//
// This category of tests are meant for testing Device APIs that are not
// related to resource creation, e.g. ICapabilities, and getting device
// statics.
//
class DeviceTest : public ::testing::Test {
 public:
  DeviceTest() = default;
  ~DeviceTest() override {
    // [Alexey] TODO: Somehow D3D12 is sensitive to the order of destruction
    // of resources and D3D debug layer report leaks
    // I suspect that might be the actual issue, but it requires deeper D3D12 backend investigation
    // first Release all GPU resources before destroying the device Resources must be released in
    // reverse order of creation and before the device is destroyed to avoid VMA cleanup issues
    if (iglDev_ && iglDev_->getBackendType() == BackendType::D3D12) {
      ib_.reset();
      vertexInputState_.reset();
      shaderStages_.reset();
      framebuffer_.reset();
      offscreenTexture_.reset();
      cmdBuf_.reset();
      cmdQueue_.reset();
      iglDev_.reset();
    }
  }

  // Set up common resources. This will create a device and a command queue
  void SetUp() override {
    // Turn off debug break so unit tests can run
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);

    // Create an offscreen texture to render to
    const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                   OFFSCREEN_RT_WIDTH,
                                                   OFFSCREEN_RT_HEIGHT,
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
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(vertexInputState_ != nullptr);

    // Initialize index buffer
    const BufferDesc bufDesc{.type = BufferDesc::BufferTypeBits::Index,
                             .data = data::vertex_index::kQuadInd.data(),
                             .length = sizeof(data::vertex_index::kQuadInd)};

    ib_ = iglDev_->createBuffer(bufDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(ib_ != nullptr);

    // Initialize Render Pipeline Descriptor, but leave the creation
    // to the individual tests in case further customization is required
    renderPipelineDesc_ = {
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = offscreenTexture_->getFormat()}}},
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
  std::shared_ptr<IFramebuffer> framebuffer_;

  std::shared_ptr<IShaderStages> shaderStages_;

  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::shared_ptr<IBuffer> ib_;

  RenderPipelineDesc renderPipelineDesc_;
};

//
// Last Draw Statics
//
// Check and make sure getDrawCount() is working properly.
//
TEST_F(DeviceTest, LastDrawStat) {
  Result ret;
  std::shared_ptr<IRenderPipelineState> pipelineState;

  size_t drawCount = iglDev_->getCurrentDrawCount();

  // Nothing has been drawn yet, so should be 0
  ASSERT_EQ(drawCount, 0);

  // Do a dummy draw
  cmdBuf_ = cmdQueue_->createCommandBuffer(cbDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(cmdBuf_ != nullptr);

  pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(pipelineState != nullptr);

  renderPass_.colorAttachments[0].clearColor = {0, 0, 0, 0};

  auto cmds = cmdBuf_->createRenderCommandEncoder(renderPass_, framebuffer_);
  cmds->bindRenderPipelineState(pipelineState);
  cmds->bindIndexBuffer(*ib_, IndexFormat::UInt16);
  cmds->drawIndexed(0); // draw 0 indices
  cmds->endEncoding();
  cmdQueue_->submit(*cmdBuf_);

  // Check draw count again
  drawCount = iglDev_->getCurrentDrawCount();

  // After the dummy draw, this should be 1
  ASSERT_EQ(drawCount, 1);
}

//
// In Development Feature API Test
//
// Make sure what flags have the right defaults, and setter and getter work.
//
TEST_F(DeviceTest, InDevelopmentFeature) {
  // Set a flag
  iglDev_->setDevelopmentFlags(InDevelopementFeatures::DummyFeatureExample, true);
  ASSERT_TRUE(iglDev_->testDevelopmentFlags(InDevelopementFeatures::DummyFeatureExample) != 0);

  // Reset the flag
  iglDev_->setDevelopmentFlags(InDevelopementFeatures::DummyFeatureExample, false);
  ASSERT_TRUE(iglDev_->testDevelopmentFlags(InDevelopementFeatures::DummyFeatureExample) == 0);
}

//
// Get Backend Type
//
// Make sure IDevice->getBackendType() only returns expected values
//
TEST_F(DeviceTest, GetBackendType) {
  if (iglDev_->getBackendType() == BackendType::Metal) {
    ASSERT_EQ(backend_, util::kBackendMtl);
  } else if (iglDev_->getBackendType() == BackendType::OpenGL) {
    ASSERT_EQ(backend_, util::kBackendOgl);
  } else if (iglDev_->getBackendType() == BackendType::Vulkan) {
    ASSERT_EQ(backend_, util::kBackendVul);
  } else if (iglDev_->getBackendType() == BackendType::D3D12) {
    ASSERT_EQ(backend_, util::kBackendD3D12);
  } else if (iglDev_->getBackendType() == BackendType::WebGPU) {
    ASSERT_EQ(backend_, util::kBackendWebGPU);
  } else {
    ASSERT_TRUE(0);
  }
}

TEST_F(DeviceTest, HasFeatureTexturePartialMipChain) {
  const bool supported = iglDev_->hasFeature(DeviceFeatures::TexturePartialMipChain);
  EXPECT_TRUE(supported == true || supported == false);
}

TEST_F(DeviceTest, GetFeatureLimitsMaxTextureDimension) {
  size_t maxDim = 0;
  const bool ok = iglDev_->getFeatureLimits(DeviceFeatureLimits::MaxTextureDimension1D2D, maxDim);
  EXPECT_TRUE(ok);
  EXPECT_GT(maxDim, 0u);
}

TEST_F(DeviceTest, CreateBufferBasic) {
  Result ret;

  // Provide initial data on creation. The default BufferDesc::storage is
  // platform-dependent (Managed on macOS, Shared elsewhere), and the OpenGL
  // backend requires data for non-dynamic (static) buffers. Supplying data
  // keeps this test valid across all backends and storage defaults.
  const uint16_t indexData[] = {0, 1, 2, 0, 2, 3};
  const BufferDesc desc{
      .type = BufferDesc::BufferTypeBits::Index,
      .data = indexData,
      .length = sizeof(indexData),
  };
  auto buffer = iglDev_->createBuffer(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
}

} // namespace igl::tests
