/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/metal/RenderPipelineState.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"

#include <igl/Common.h>
#include <igl/Device.h>
#include <igl/IGL.h>
#include <igl/metal/Shader.h>
#include <igl/metal/VertexInputState.h>
namespace igl::tests {

class RenderPipelineStateMTLTest : public ::testing::Test {
 public:
  RenderPipelineStateMTLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);
    Result ret;

    // Initialize vertex Shader
    auto shaderLibrary =
        ShaderLibraryCreator::fromStringInput(*iglDev_,
                                              std::string(data::shader::kMtlSimpleShader).c_str(),
                                              std::string(data::shader::kSimpleVertFunc),
                                              std::string(data::shader::kSimpleFragFunc),
                                              "",
                                              &ret);

    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(shaderLibrary != nullptr);

    vertShader_ = shaderLibrary->getShaderModule(std::string(data::shader::kSimpleVertFunc));
    ASSERT_TRUE(vertShader_ != nullptr);

    fragShader_ = shaderLibrary->getShaderModule(std::string(data::shader::kSimpleFragFunc));
    ASSERT_TRUE(fragShader_ != nullptr);

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

    NSError* error = nullptr;
    MTLRenderPipelineDescriptor* metalDesc = [MTLRenderPipelineDescriptor new];

    metalDesc.vertexDescriptor =
        static_cast<metal::VertexInputState*>(vertexInputState_.get())->get();
    metalDesc.vertexFunction = static_cast<metal::ShaderModule*>(vertShader_.get())->get();
    IGL_DEBUG_ASSERT(metalDesc.vertexFunction, "RenderPipeline requires non-null vertex function");
    metalDesc.fragmentFunction = static_cast<metal::ShaderModule*>(fragShader_.get())->get();
    metalDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    metalDesc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;

    MTLRenderPipelineReflection* reflection = nil;
    auto device = MTLCreateSystemDefaultDevice();

    RenderPipelineDesc pipelineDesc{};
    pipelineDesc.cullMode = CullMode::Back;
    pipelineDesc.frontFaceWinding = WindingMode::CounterClockwise;
    pipelineDesc.polygonFillMode = PolygonFillMode::Fill;

// Suppress warnings about MTLPipelineOptionArgumentInfo being deprecated
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    // Create reflection for use later in binding, etc.
    id<MTLRenderPipelineState> metalObject =
        [device newRenderPipelineStateWithDescriptor:metalDesc
                                             options:MTLPipelineOptionArgumentInfo
                                          reflection:&reflection
                                               error:&error];

    pipeState = std::make_shared<metal::RenderPipelineState>(metalObject, reflection, pipelineDesc);

    id<MTLRenderPipelineState> metalObjectWithoutRefl =
        [device newRenderPipelineStateWithDescriptor:metalDesc
                                             options:MTLPipelineOptionArgumentInfo
                                          reflection:nullptr
                                               error:&error];
#pragma GCC diagnostic pop

    pipeStateWithNoRefl_ =
        std::make_shared<metal::RenderPipelineState>(metalObjectWithoutRefl, nullptr, pipelineDesc);

    ASSERT_NE(pipeState, nullptr);
  }
  void TearDown() override {}

  std::shared_ptr<metal::RenderPipelineState> pipeState;

 protected:
  std::shared_ptr<metal::RenderPipelineState> pipeStateWithNoRefl_;

  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;

  std::shared_ptr<IShaderModule> vertShader_;
  std::shared_ptr<IShaderModule> fragShader_;
  std::shared_ptr<IVertexInputState> vertexInputState_;
};

TEST_F(RenderPipelineStateMTLTest, GetIndexByName) {
  auto index = pipeState->getIndexByName(std::string(data::shader::kSimpleUv), ShaderStage::Vertex);
  ASSERT_EQ(index, 1);
}

TEST_F(RenderPipelineStateMTLTest, GetNonexistentIndexByName) {
  auto index = pipeState->getIndexByName("", ShaderStage::Fragment);
  ASSERT_EQ(index, -1);
}

TEST_F(RenderPipelineStateMTLTest, GetIndexByNameWithoutRefl) {
  auto index = pipeStateWithNoRefl_->getIndexByName(std::string(data::shader::kSimpleUv),
                                                    ShaderStage::Vertex);
  ASSERT_EQ(index, -1);
}

TEST_F(RenderPipelineStateMTLTest, GetIndexByNameHandle) {
  auto index =
      pipeState->getIndexByName(IGL_NAMEHANDLE(data::shader::kSimpleUv), ShaderStage::Vertex);
  ASSERT_EQ(index, 1);
}

TEST_F(RenderPipelineStateMTLTest, GetNonexistentIndexByNameHandle) {
  auto index = pipeState->getIndexByName("", ShaderStage::Fragment);
  ASSERT_EQ(index, -1);
}

TEST_F(RenderPipelineStateMTLTest, GetIndexByNameHandleWithoutRefl) {
  auto index = pipeStateWithNoRefl_->getIndexByName(IGL_NAMEHANDLE(data::shader::kSimpleUv),
                                                    ShaderStage::Vertex);
  ASSERT_EQ(index, -1);
}

TEST_F(RenderPipelineStateMTLTest, ConvertColorWriteMaskRed) {
  auto mask = igl::metal::RenderPipelineState::convertColorWriteMask(kColorWriteBitsRed);
  ASSERT_EQ(mask, MTLColorWriteMaskRed);
}

TEST_F(RenderPipelineStateMTLTest, ConvertColorWriteMaskGreen) {
  auto mask = igl::metal::RenderPipelineState::convertColorWriteMask(kColorWriteBitsGreen);
  ASSERT_EQ(mask, MTLColorWriteMaskGreen);
}

TEST_F(RenderPipelineStateMTLTest, ConvertColorWriteMaskBlue) {
  auto mask = igl::metal::RenderPipelineState::convertColorWriteMask(kColorWriteBitsBlue);
  ASSERT_EQ(mask, MTLColorWriteMaskBlue);
}

TEST_F(RenderPipelineStateMTLTest, ConvertColorWriteMaskAlpha) {
  auto mask = igl::metal::RenderPipelineState::convertColorWriteMask(kColorWriteBitsAlpha);
  ASSERT_EQ(mask, MTLColorWriteMaskAlpha);
}

} // namespace igl::tests
