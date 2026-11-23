/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/ShaderData.h"
#include "../util/TestDevice.h"

#include <igl/DepthStencilState.h>
#include <igl/Device.h>
#include <igl/Shader.h>
#include <igl/ShaderCreator.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class DeviceExtendedVulkanTest : public ::testing::Test {
 public:
  DeviceExtendedVulkanTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
};

TEST_F(DeviceExtendedVulkanTest, CreateDepthStencilState) {
  Result ret;

  const DepthStencilStateDesc desc{.compareFunction = CompareFunction::Less,
                                   .isDepthWriteEnabled = true};

  auto state = iglDev_->createDepthStencilState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(state, nullptr);
}

TEST_F(DeviceExtendedVulkanTest, CreateDepthStencilStateWithStencil) {
  Result ret;

  constexpr StencilStateDesc kStencil{.stencilFailureOperation = StencilOperation::Keep,
                                      .depthFailureOperation = StencilOperation::Keep,
                                      .depthStencilPassOperation = StencilOperation::Replace,
                                      .stencilCompareFunction = CompareFunction::AlwaysPass};
  const DepthStencilStateDesc desc{.compareFunction = CompareFunction::LessEqual,
                                   .isDepthWriteEnabled = true,
                                   .backFaceStencil = kStencil,
                                   .frontFaceStencil = kStencil};

  auto state = iglDev_->createDepthStencilState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(state, nullptr);
}

TEST_F(DeviceExtendedVulkanTest, GetShaderVersion) {
  auto version = iglDev_->getShaderVersion();
  EXPECT_GT(version.majorVersion, 0u);
}

TEST_F(DeviceExtendedVulkanTest, GetBackendVersion) {
  auto version = iglDev_->getBackendVersion();
  EXPECT_GT(version.majorVersion, 0u);
}

TEST_F(DeviceExtendedVulkanTest, GetCurrentDrawCount) {
  auto drawCount = iglDev_->getCurrentDrawCount();
  EXPECT_EQ(drawCount, 0u);
}

TEST_F(DeviceExtendedVulkanTest, GetBackendType) {
  EXPECT_EQ(iglDev_->getBackendType(), BackendType::Vulkan);
}

TEST_F(DeviceExtendedVulkanTest, HasFeatureTexture3D) {
  EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Texture3D));
}

TEST_F(DeviceExtendedVulkanTest, GetFeatureLimitsMaxTextureDimension) {
  size_t maxDim = 0;
  bool result = iglDev_->getFeatureLimits(DeviceFeatureLimits::MaxTextureDimension1D2D, maxDim);
  EXPECT_TRUE(result);
  EXPECT_GT(maxDim, 0u);
}

TEST_F(DeviceExtendedVulkanTest, CreateShaderModuleNullBinaryDataReturnsArgumentInvalid) {
  const ShaderModuleDesc desc{
      .info = {.stage = ShaderStage::Vertex, .entryPoint = "main"},
      .input = {.data = nullptr, .length = 16, .type = ShaderInputType::Binary},
      .debugName = "nullData"};

  Result ret;
  auto module = iglDev_->createShaderModule(desc, &ret);
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

TEST_F(DeviceExtendedVulkanTest, CreateShaderModuleZeroLengthBinaryDataReturnsArgumentInvalid) {
  const uint32_t dummy = 0;
  const ShaderModuleDesc desc{
      .info = {.stage = ShaderStage::Vertex, .entryPoint = "main"},
      .input = {.data = &dummy, .length = 0, .type = ShaderInputType::Binary},
      .debugName = "zeroLength"};

  Result ret;
  auto module = iglDev_->createShaderModule(desc, &ret);
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

TEST_F(DeviceExtendedVulkanTest, HasFeatureCompute) {
  EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Compute));
}

TEST_F(DeviceExtendedVulkanTest, HasFeatureMultiview) {
  // Multiview is an optional Vulkan capability (gated behind VK_KHR_multiview / the multiview
  // feature bit), so the device may legitimately report either value depending on the underlying
  // implementation. Verify only that the query executes without crashing.
  const bool hasMultiview = iglDev_->hasFeature(DeviceFeatures::Multiview);
  (void)hasMultiview;
}

TEST_F(DeviceExtendedVulkanTest, HasFeatureExplicitBinding) {
  EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ExplicitBinding));
}

TEST_F(DeviceExtendedVulkanTest, HasFeatureDynamicVertexBufferStride) {
  // DynamicVertexBufferStride is gated on both vkCmdBindVertexBuffers2 availability and the
  // enableDynamicVertexBufferStride config flag (disabled by default), so the device may report
  // either value. Verify only that the query executes without crashing.
  const bool hasDynamicVertexBufferStride =
      iglDev_->hasFeature(DeviceFeatures::DynamicVertexBufferStride);
  (void)hasDynamicVertexBufferStride;
}

TEST_F(DeviceExtendedVulkanTest, TextureFormatCapabilitiesRGBA) {
  const ICapabilities::TextureFormatCapabilities caps =
      iglDev_->getTextureFormatCapabilities(TextureFormat::RGBA_UNorm8);
  EXPECT_TRUE(contains(caps, ICapabilities::TextureFormatCapabilityBits::Sampled));
  EXPECT_TRUE(contains(caps, ICapabilities::TextureFormatCapabilityBits::Attachment));
  EXPECT_TRUE(contains(caps, ICapabilities::TextureFormatCapabilityBits::SampledFiltered));
}

TEST_F(DeviceExtendedVulkanTest, GetNormalizedZRange) {
  EXPECT_EQ(iglDev_->getNormalizedZRange(), NormalizedZRange::NegOneToOne);
}

TEST_F(DeviceExtendedVulkanTest, GetShaderCompilationCount) {
  // Don't assume a specific starting count: device setup may compile internal shaders. Instead
  // verify the counter increases after we explicitly compile a shader.
  const size_t countBefore = iglDev_->getShaderCompilationCount();

  Result ret;
  const std::string vertexSource(data::shader::kVulkanSimpleVertShader);
  auto vertexModule =
      ShaderModuleCreator::fromStringInput(*iglDev_,
                                           vertexSource.c_str(),
                                           {.stage = ShaderStage::Vertex, .entryPoint = "main"},
                                           "GetShaderCompilationCount",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(vertexModule, nullptr);

  EXPECT_GT(iglDev_->getShaderCompilationCount(), countBefore);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
