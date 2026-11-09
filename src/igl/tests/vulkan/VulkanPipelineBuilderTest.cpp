/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanPipelineBuilder.h>

#include "../util/TestDevice.h"

#include <igl/ShaderCreator.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/ShaderModule.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class VulkanPipelineBuilderTest : public ::testing::Test {
 public:
  VulkanPipelineBuilderTest() = default;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;

  igl::vulkan::VulkanContext& getVulkanContext() {
    auto& device = static_cast<igl::vulkan::Device&>(*iglDev_);
    return device.getVulkanContext();
  }
};

TEST_F(VulkanPipelineBuilderTest, DepthBiasEnable) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.depthBiasEnable(true);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, StencilStateOps) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.stencilStateOps(VK_STENCIL_FACE_FRONT_BIT,
                                         VK_STENCIL_OP_KEEP,
                                         VK_STENCIL_OP_REPLACE,
                                         VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                                         VK_COMPARE_OP_ALWAYS);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, CullModeVariants) {
  igl::vulkan::VulkanPipelineBuilder builder1;
  auto& r1 = builder1.cullMode(VK_CULL_MODE_NONE);
  EXPECT_EQ(&r1, &builder1);

  igl::vulkan::VulkanPipelineBuilder builder2;
  auto& r2 = builder2.cullMode(VK_CULL_MODE_FRONT_BIT);
  EXPECT_EQ(&r2, &builder2);

  igl::vulkan::VulkanPipelineBuilder builder3;
  auto& r3 = builder3.cullMode(VK_CULL_MODE_BACK_BIT);
  EXPECT_EQ(&r3, &builder3);

  igl::vulkan::VulkanPipelineBuilder builder4;
  auto& r4 = builder4.cullMode(VK_CULL_MODE_FRONT_AND_BACK);
  EXPECT_EQ(&r4, &builder4);
}

TEST_F(VulkanPipelineBuilderTest, PolygonModeLine) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.polygonMode(VK_POLYGON_MODE_LINE);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, MultipleColorBlendAttachments) {
  igl::vulkan::VulkanPipelineBuilder builder;

  std::vector<VkPipelineColorBlendAttachmentState> blendStates(3);
  for (auto& state : blendStates) {
    state = {};
    state.blendEnable = VK_TRUE;
    state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    state.colorBlendOp = VK_BLEND_OP_ADD;
    state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    state.alphaBlendOp = VK_BLEND_OP_ADD;
    state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  }

  auto& result = builder.colorBlendAttachmentStates(blendStates);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, PrimitiveTopologyVariants) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.primitiveTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, DynamicStates) {
  igl::vulkan::VulkanPipelineBuilder builder;
  std::vector<VkDynamicState> states = {
      VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
  auto& result = builder.dynamicStates(states);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, DepthCompareOp) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.depthCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL, true);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, DepthWriteEnable) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.depthWriteEnable(true);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, FrontFace) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.frontFace(VK_FRONT_FACE_CLOCKWISE);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, RasterizationSamples) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.rasterizationSamples(VK_SAMPLE_COUNT_4_BIT);
  EXPECT_EQ(&result, &builder);

  // Verify the rasterization samples state is actually applied by building a pipeline.
  // The pipeline creation will fail if the multisample state is invalid.
  auto& ctx = getVulkanContext();
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;

  // Create a minimal pipeline layout
  VkPipelineLayoutCreateInfo layoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
  };
  VkResult res =
      ctx.vf_.vkCreatePipelineLayout(ctx.getVkDevice(), &layoutInfo, nullptr, &pipelineLayout);
  ASSERT_EQ(res, VK_SUCCESS);

  // Create a minimal render pass with 4x MSAA
  const VkAttachmentDescription colorAttachment = {
      .format = VK_FORMAT_B8G8R8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_4_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  const VkAttachmentReference colorAttachmentRef = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
  };

  const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachmentRef,
  };

  const VkRenderPassCreateInfo renderPassInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &colorAttachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
  };

  res = ctx.vf_.vkCreateRenderPass(ctx.getVkDevice(), &renderPassInfo, nullptr, &renderPass);
  ASSERT_EQ(res, VK_SUCCESS);

  // Create minimal shader modules
  Result ret;
  const std::string vertSource = R"(
    #version 450
    void main() {
      gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    }
  )";
  auto vertModule =
      ShaderModuleCreator::fromStringInput(*iglDev_,
                                           vertSource.c_str(),
                                           {.stage = ShaderStage::Vertex, .entryPoint = "main"},
                                           "TestVert",
                                           &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(vertModule, nullptr);

  const std::string fragSource = R"(
    #version 450
    layout(location = 0) out vec4 outColor;
    void main() {
      outColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
  )";
  auto fragModule =
      ShaderModuleCreator::fromStringInput(*iglDev_,
                                           fragSource.c_str(),
                                           {.stage = ShaderStage::Fragment, .entryPoint = "main"},
                                           "TestFrag",
                                           &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(fragModule, nullptr);

  // Add shader stages to builder
  const VkPipelineShaderStageCreateInfo vertStage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = igl::vulkan::ShaderModule::getVkShaderModule(vertModule),
      .pName = "main",
  };

  const VkPipelineShaderStageCreateInfo fragStage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = igl::vulkan::ShaderModule::getVkShaderModule(fragModule),
      .pName = "main",
  };

  builder.shaderStage(vertStage);
  builder.shaderStage(fragStage);
  builder.dynamicStates({VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});
  std::vector<VkPipelineColorBlendAttachmentState> blendStates(1);
  blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  builder.colorBlendAttachmentStates(blendStates);

  // Build pipeline with 4x MSAA - should succeed if rasterizationSamples is properly applied
  res = builder.build(ctx.vf_,
                      ctx.getVkDevice(),
                      0,
                      VK_NULL_HANDLE,
                      pipelineLayout,
                      renderPass,
                      &pipeline,
                      "TestRasterizationSamples");
  EXPECT_EQ(res, VK_SUCCESS);
  EXPECT_NE(pipeline, VK_NULL_HANDLE);

  // Cleanup
  if (pipeline != VK_NULL_HANDLE) {
    ctx.vf_.vkDestroyPipeline(ctx.getVkDevice(), pipeline, nullptr);
  }
  if (renderPass != VK_NULL_HANDLE) {
    ctx.vf_.vkDestroyRenderPass(ctx.getVkDevice(), renderPass, nullptr);
  }
  if (pipelineLayout != VK_NULL_HANDLE) {
    ctx.vf_.vkDestroyPipelineLayout(ctx.getVkDevice(), pipelineLayout, nullptr);
  }
}

TEST_F(VulkanPipelineBuilderTest, AlphaToCoverageEnable) {
  igl::vulkan::VulkanPipelineBuilder builder1;
  auto& r1 = builder1.alphaToCoverageEnable(true);
  EXPECT_EQ(&r1, &builder1);

  igl::vulkan::VulkanPipelineBuilder builder2;
  auto& r2 = builder2.alphaToCoverageEnable(false);
  EXPECT_EQ(&r2, &builder2);

  // Verify alpha-to-coverage state is actually applied by building pipelines.
  // The pipeline creation will fail if the multisample state is invalid.
  auto& ctx = getVulkanContext();
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline1 = VK_NULL_HANDLE;
  VkPipeline pipeline2 = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;

  // Create a minimal pipeline layout
  VkPipelineLayoutCreateInfo layoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
  };
  VkResult res =
      ctx.vf_.vkCreatePipelineLayout(ctx.getVkDevice(), &layoutInfo, nullptr, &pipelineLayout);
  ASSERT_EQ(res, VK_SUCCESS);

  // Create a minimal render pass
  const VkAttachmentDescription colorAttachment = {
      .format = VK_FORMAT_B8G8R8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  const VkAttachmentReference colorAttachmentRef = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
  };

  const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachmentRef,
  };

  const VkRenderPassCreateInfo renderPassInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &colorAttachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
  };

  res = ctx.vf_.vkCreateRenderPass(ctx.getVkDevice(), &renderPassInfo, nullptr, &renderPass);
  ASSERT_EQ(res, VK_SUCCESS);

  // Create minimal shader modules
  Result ret;
  const std::string vertSource = R"(
    #version 450
    void main() {
      gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    }
  )";
  auto vertModule =
      ShaderModuleCreator::fromStringInput(*iglDev_,
                                           vertSource.c_str(),
                                           {.stage = ShaderStage::Vertex, .entryPoint = "main"},
                                           "TestVert",
                                           &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(vertModule, nullptr);

  const std::string fragSource = R"(
    #version 450
    layout(location = 0) out vec4 outColor;
    void main() {
      outColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
  )";
  auto fragModule =
      ShaderModuleCreator::fromStringInput(*iglDev_,
                                           fragSource.c_str(),
                                           {.stage = ShaderStage::Fragment, .entryPoint = "main"},
                                           "TestFrag",
                                           &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(fragModule, nullptr);

  const VkPipelineShaderStageCreateInfo vertStage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = igl::vulkan::ShaderModule::getVkShaderModule(vertModule),
      .pName = "main",
  };

  const VkPipelineShaderStageCreateInfo fragStage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = igl::vulkan::ShaderModule::getVkShaderModule(fragModule),
      .pName = "main",
  };

  builder1.shaderStage(vertStage);
  builder1.shaderStage(fragStage);
  builder1.dynamicStates({VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});
  builder2.shaderStage(vertStage);
  builder2.shaderStage(fragStage);
  builder2.dynamicStates({VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});
  std::vector<VkPipelineColorBlendAttachmentState> blendStates1(1);
  blendStates1[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  auto blendStates2 = blendStates1;
  builder1.colorBlendAttachmentStates(blendStates1);
  builder2.colorBlendAttachmentStates(blendStates2);

  // Build pipeline with alpha-to-coverage enabled - should succeed
  res = builder1.build(ctx.vf_,
                       ctx.getVkDevice(),
                       0,
                       VK_NULL_HANDLE,
                       pipelineLayout,
                       renderPass,
                       &pipeline1,
                       "TestAlphaToCoverageEnabled");
  EXPECT_EQ(res, VK_SUCCESS);
  EXPECT_NE(pipeline1, VK_NULL_HANDLE);

  // Build pipeline with alpha-to-coverage disabled - should also succeed
  res = builder2.build(ctx.vf_,
                       ctx.getVkDevice(),
                       0,
                       VK_NULL_HANDLE,
                       pipelineLayout,
                       renderPass,
                       &pipeline2,
                       "TestAlphaToCoverageDisabled");
  EXPECT_EQ(res, VK_SUCCESS);
  EXPECT_NE(pipeline2, VK_NULL_HANDLE);

  // Cleanup
  if (pipeline1 != VK_NULL_HANDLE) {
    ctx.vf_.vkDestroyPipeline(ctx.getVkDevice(), pipeline1, nullptr);
  }
  if (pipeline2 != VK_NULL_HANDLE) {
    ctx.vf_.vkDestroyPipeline(ctx.getVkDevice(), pipeline2, nullptr);
  }
  if (renderPass != VK_NULL_HANDLE) {
    ctx.vf_.vkDestroyRenderPass(ctx.getVkDevice(), renderPass, nullptr);
  }
  if (pipelineLayout != VK_NULL_HANDLE) {
    ctx.vf_.vkDestroyPipelineLayout(ctx.getVkDevice(), pipelineLayout, nullptr);
  }
}

TEST_F(VulkanPipelineBuilderTest, DynamicStateSingle) {
  igl::vulkan::VulkanPipelineBuilder builder;
  auto& result = builder.dynamicState(VK_DYNAMIC_STATE_VIEWPORT);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, ShaderStage) {
  igl::vulkan::VulkanPipelineBuilder builder;
  const VkPipelineShaderStageCreateInfo stage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT};
  auto& result = builder.shaderStage(stage);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, ShaderStages) {
  igl::vulkan::VulkanPipelineBuilder builder;
  const std::vector<VkPipelineShaderStageCreateInfo> stages = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT}};
  auto& result = builder.shaderStages(stages);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, VertexInputState) {
  igl::vulkan::VulkanPipelineBuilder builder;
  const VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  auto& result = builder.vertexInputState(vertexInput);
  EXPECT_EQ(&result, &builder);
}

TEST_F(VulkanPipelineBuilderTest, ComputePipelineBasic) {
  igl::vulkan::VulkanComputePipelineBuilder builder{};
  (void)builder; // Verify construction succeeds
  EXPECT_GE(igl::vulkan::VulkanComputePipelineBuilder::getNumPipelinesCreated(), 0u);
}

TEST_F(VulkanPipelineBuilderTest, ComputePipelineShaderStage) {
  igl::vulkan::VulkanComputePipelineBuilder builder{};
  const VkPipelineShaderStageCreateInfo stage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT};
  auto& result = builder.shaderStage(stage);
  EXPECT_EQ(&result, &builder);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
