/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/RenderPipelineState.h>

#include <igl/Common.h>
#include <igl/ShaderCreator.h>
#include <igl/VertexInputState.h>
#include <igl/tests/util/device/TestDevice.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#ifdef __ANDROID__
#endif

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX
namespace igl::tests {

//
// RenderPipelineStateTest
//
// Unit tests for RenderPipelineState.cpp
//
class RenderPipelineStateTest : public ::testing::TestWithParam<std::tuple<PolygonFillMode,
                                                                           CullMode,
                                                                           WindingMode,
                                                                           VertexAttributeFormat,
                                                                           BlendOp,
                                                                           BlendFactor>> {
  void SetUp() override {
    // Turn off debug break so unit tests can run
    setDebugBreakEnabled(false);

    device_ = igl::tests::util::device::createTestDevice(BackendType::Vulkan);
    ASSERT_TRUE(device_ != nullptr);
    auto& device = static_cast<vulkan::Device&>(*device_);
    context_ = &device.getVulkanContext();
    ASSERT_TRUE(context_ != nullptr);
  }

 protected:
  std::shared_ptr<IDevice> device_;
  vulkan::VulkanContext* context_ = nullptr;
};

// polygonFillModeToVkPolygonMode *********************************************
TEST_P(RenderPipelineStateTest, PolygonFillModeToVkPolygonModeTest) {
  const auto polygonFillMode = std::get<0>(GetParam());
  const auto cullMode = std::get<1>(GetParam());
  const auto windingMode = std::get<2>(GetParam());
  const auto vertexFormat = std::get<3>(GetParam());
  const auto blendOp = std::get<4>(GetParam());
  const auto blendFactor = std::get<5>(GetParam());

  Result result;

  const VertexInputStateDesc inputDesc{.numAttributes = 1,
                                       .attributes = {{.format = vertexFormat}}};
  const auto inputState = device_->createVertexInputState(inputDesc, &result);
  EXPECT_TRUE(result.isOk());

  //   precision highp float;
  constexpr const char* codeVS = R"(

  void main() {
    gl_Position = vec4(0., 0., 0., 1.0);
  }
)";

  constexpr const char* codeFS = R"(
                layout(location = 0) out vec4 out_FragColor;

                void main() {
                  out_FragColor = vec4(0., 0., 0., 1.0);
                }
            )";

  const RenderPipelineDesc pipelineDesc{
      .vertexInputState = inputState,
      .shaderStages = ShaderStagesCreator::fromModuleStringInput(
          *device_, codeVS, "main", "", codeFS, "main", "", nullptr),
      .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8,
                                           .blendEnabled = true,
                                           .rgbBlendOp = blendOp,
                                           .srcRGBBlendFactor = blendFactor}}},
      .cullMode = cullMode,
      .frontFaceWinding = windingMode,
      .polygonFillMode = polygonFillMode};
  const auto renderPipeline = device_->createRenderPipeline(pipelineDesc, &result);
}

INSTANTIATE_TEST_SUITE_P(
    AllFormats,
    RenderPipelineStateTest,
    ::testing::Combine(::testing::Values(igl::PolygonFillMode::Line),
                       ::testing::Values(igl::CullMode::Front, igl::CullMode::Back),
                       ::testing::Values(igl::WindingMode::Clockwise),
                       ::testing::Values(VertexAttributeFormat::Float1,
                                         VertexAttributeFormat::Float2,
                                         VertexAttributeFormat::Float3,
                                         VertexAttributeFormat::Float4,
                                         VertexAttributeFormat::Byte1,
                                         VertexAttributeFormat::Byte2,
                                         VertexAttributeFormat::Byte3,
                                         VertexAttributeFormat::Byte4,
                                         VertexAttributeFormat::UByte1,
                                         VertexAttributeFormat::UByte2,
                                         VertexAttributeFormat::UByte3,
                                         VertexAttributeFormat::UByte4,
                                         VertexAttributeFormat::Short1,
                                         VertexAttributeFormat::Short2,
                                         VertexAttributeFormat::Short3,
                                         VertexAttributeFormat::Short4,
                                         VertexAttributeFormat::UShort1,
                                         VertexAttributeFormat::UShort2,
                                         VertexAttributeFormat::UShort3,
                                         VertexAttributeFormat::UShort4,
                                         VertexAttributeFormat::Byte1Norm,
                                         VertexAttributeFormat::Byte2Norm,
                                         VertexAttributeFormat::Byte3Norm,
                                         VertexAttributeFormat::Byte4Norm,
                                         VertexAttributeFormat::UByte1Norm,
                                         VertexAttributeFormat::UByte2Norm,
                                         VertexAttributeFormat::UByte3Norm,
                                         VertexAttributeFormat::UByte4Norm,
                                         VertexAttributeFormat::Short1Norm,
                                         VertexAttributeFormat::Short2Norm,
                                         VertexAttributeFormat::Short3Norm,
                                         VertexAttributeFormat::Short4Norm,
                                         VertexAttributeFormat::UShort1Norm,
                                         VertexAttributeFormat::UShort2Norm,
                                         VertexAttributeFormat::UShort3Norm,
                                         VertexAttributeFormat::UShort4Norm,
                                         VertexAttributeFormat::Int1,
                                         VertexAttributeFormat::Int2,
                                         VertexAttributeFormat::Int3,
                                         VertexAttributeFormat::Int4,
                                         VertexAttributeFormat::UInt1,
                                         VertexAttributeFormat::UInt2,
                                         VertexAttributeFormat::UInt3,
                                         VertexAttributeFormat::UInt4, // Half-float
                                         VertexAttributeFormat::HalfFloat1,
                                         VertexAttributeFormat::HalfFloat2,
                                         VertexAttributeFormat::HalfFloat3,
                                         VertexAttributeFormat::HalfFloat4,
                                         VertexAttributeFormat::Int_2_10_10_10_REV),
                       ::testing::Values(BlendOp::Add),
                       ::testing::Values(BlendFactor::Zero)));
INSTANTIATE_TEST_SUITE_P(AllBlendOps,
                         RenderPipelineStateTest,
                         ::testing::Combine(::testing::Values(igl::PolygonFillMode::Line),
                                            ::testing::Values(igl::CullMode::Front,
                                                              igl::CullMode::Back),
                                            ::testing::Values(igl::WindingMode::Clockwise),
                                            ::testing::Values(VertexAttributeFormat::Float1),
                                            ::testing::Values(BlendOp::Add,
                                                              BlendOp::Subtract,
                                                              BlendOp::ReverseSubtract,
                                                              BlendOp::Min,
                                                              BlendOp::Max),
                                            ::testing::Values(BlendFactor::Zero)));
INSTANTIATE_TEST_SUITE_P(AllBlendFactors,
                         RenderPipelineStateTest,
                         ::testing::Combine(::testing::Values(igl::PolygonFillMode::Line),
                                            ::testing::Values(igl::CullMode::Front,
                                                              igl::CullMode::Back),
                                            ::testing::Values(igl::WindingMode::Clockwise),
                                            ::testing::Values(VertexAttributeFormat::Float1),
                                            ::testing::Values(BlendOp::Add),
                                            ::testing::Values(BlendFactor::Zero,
                                                              BlendFactor::One,
                                                              BlendFactor::SrcColor,
                                                              BlendFactor::OneMinusSrcColor,
                                                              BlendFactor::DstColor,
                                                              BlendFactor::OneMinusDstColor,
                                                              BlendFactor::SrcAlpha,
                                                              BlendFactor::OneMinusSrcAlpha,
                                                              BlendFactor::DstAlpha,
                                                              BlendFactor::OneMinusDstAlpha,
                                                              BlendFactor::BlendColor,
                                                              BlendFactor::OneMinusBlendColor,
                                                              BlendFactor::BlendAlpha,
                                                              BlendFactor::OneMinusBlendAlpha,
                                                              BlendFactor::SrcAlphaSaturated,
                                                              BlendFactor::Src1Color,
                                                              BlendFactor::OneMinusSrc1Color,
                                                              BlendFactor::Src1Alpha,
                                                              BlendFactor::OneMinusSrc1Alpha)));
} // namespace igl::tests

#endif
