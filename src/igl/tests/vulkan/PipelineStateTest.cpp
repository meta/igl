/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <igl/DepthStencilState.h>
#include <igl/SamplerState.h>
#include <igl/VertexInputState.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class PipelineStateTest : public ::testing::Test {
 public:
  PipelineStateTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;

  vulkan::VulkanContext& getVulkanContext() {
    auto& device = static_cast<vulkan::Device&>(*iglDev_);
    return device.getVulkanContext();
  }
};

TEST_F(PipelineStateTest, PipelineLayoutCreation) {
  auto& ctx = getVulkanContext();
  EXPECT_TRUE(ctx.getVkDevice() != VK_NULL_HANDLE);
  EXPECT_TRUE(ctx.pipelineCache_ != VK_NULL_HANDLE);
}

TEST_F(PipelineStateTest, CreateSamplerState) {
  Result ret;

  const SamplerStateDesc desc{
      .minFilter = SamplerMinMagFilter::Linear,
      .magFilter = SamplerMinMagFilter::Linear,
      .mipFilter = SamplerMipFilter::Linear,
      .addressModeU = SamplerAddressMode::Repeat,
      .addressModeV = SamplerAddressMode::Repeat,
  };

  const auto sampler = iglDev_->createSamplerState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(sampler, nullptr);
}

TEST_F(PipelineStateTest, CreateVertexInputState) {
  Result ret;

  const VertexInputStateDesc desc{
      .numAttributes = 1,
      .attributes =
          {
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float4,
               .offset = 0,
               .name = "position",
               .location = 0},
          },
      .numInputBindings = 1,
      .inputBindings =
          {
              {.stride = 16, .sampleFunction = VertexSampleFunction::PerVertex},
          },
  };

  const auto vis = iglDev_->createVertexInputState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(vis, nullptr);
}

TEST_F(PipelineStateTest, PipelineCacheExists) {
  auto& ctx = getVulkanContext();
  EXPECT_TRUE(ctx.pipelineCache_ != VK_NULL_HANDLE);
}

TEST_F(PipelineStateTest, CreateSamplerStateClampAddressMode) {
  Result ret;

  const SamplerStateDesc desc{
      .minFilter = SamplerMinMagFilter::Nearest,
      .magFilter = SamplerMinMagFilter::Nearest,
      .mipFilter = SamplerMipFilter::Disabled,
      .addressModeU = SamplerAddressMode::Clamp,
      .addressModeV = SamplerAddressMode::Clamp,
      .addressModeW = SamplerAddressMode::Clamp,
  };
  const auto sampler = iglDev_->createSamplerState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(sampler, nullptr);
}

TEST_F(PipelineStateTest, CreateVertexInputStateMultipleAttributes) {
  Result ret;

  const VertexInputStateDesc desc{
      .numAttributes = 2,
      .attributes =
          {
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float4,
               .offset = 0,
               .name = "position",
               .location = 0},
              {.bufferIndex = 1,
               .format = VertexAttributeFormat::Float2,
               .offset = 0,
               .name = "uv",
               .location = 1},
          },
      .numInputBindings = 2,
      .inputBindings =
          {
              {.stride = sizeof(float) * 4},
              {.stride = sizeof(float) * 2},
          },
  };

  const auto vis = iglDev_->createVertexInputState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(vis, nullptr);
}

TEST_F(PipelineStateTest, CreateDepthStencilStateWithDepthWrite) {
  Result ret;

  const DepthStencilStateDesc desc{
      .compareFunction = CompareFunction::Less,
      .isDepthWriteEnabled = true,
  };
  const auto dss = iglDev_->createDepthStencilState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(dss, nullptr);
}

TEST_F(PipelineStateTest, CreateDepthStencilStateWithStencilOps) {
  Result ret;

  const DepthStencilStateDesc desc{
      .compareFunction = CompareFunction::LessEqual,
      .isDepthWriteEnabled = true,
      .backFaceStencil = {.depthStencilPassOperation = StencilOperation::Replace,
                          .stencilCompareFunction = CompareFunction::AlwaysPass},
      .frontFaceStencil = {.depthStencilPassOperation = StencilOperation::Replace,
                           .stencilCompareFunction = CompareFunction::AlwaysPass},
  };
  const auto dss = iglDev_->createDepthStencilState(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(dss, nullptr);
}

TEST_F(PipelineStateTest, VulkanDeviceHasComputeSupport) {
  EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Compute));
}

TEST_F(PipelineStateTest, VulkanDeviceHasShaderLibrarySupport) {
  EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderLibrary));
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
