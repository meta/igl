/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanHelpers.h>

namespace igl::vulkan {

class VulkanPipelineBuilder final {
 public:
  VulkanPipelineBuilder();
  ~VulkanPipelineBuilder() = default;

  VulkanPipelineBuilder& depthBiasEnable(bool enable);
  VulkanPipelineBuilder& depthWriteEnable(bool enable);
  VulkanPipelineBuilder& depthCompareOp(VkCompareOp compareOp, bool writeDepthEnable);
  VulkanPipelineBuilder& dynamicState(VkDynamicState state);
  VulkanPipelineBuilder& dynamicStates(const std::vector<VkDynamicState>& states);
  VulkanPipelineBuilder& primitiveTopology(VkPrimitiveTopology topology);
  VulkanPipelineBuilder& rasterizationSamples(VkSampleCountFlagBits samples);
  VulkanPipelineBuilder& alphaToCoverageEnable(bool enable);
  VulkanPipelineBuilder& shaderStage(VkPipelineShaderStageCreateInfo stage);
  VulkanPipelineBuilder& shaderStages(const std::vector<VkPipelineShaderStageCreateInfo>& stages);
  VulkanPipelineBuilder& stencilStateOps(VkStencilFaceFlags faceMask,
                                         VkStencilOp failOp,
                                         VkStencilOp passOp,
                                         VkStencilOp depthFailOp,
                                         VkCompareOp compareOp);
  VulkanPipelineBuilder& cullMode(VkCullModeFlags mode);
  VulkanPipelineBuilder& frontFace(VkFrontFace mode);
  VulkanPipelineBuilder& polygonMode(VkPolygonMode mode);
  VulkanPipelineBuilder& vertexInputState(const VkPipelineVertexInputStateCreateInfo& state);
  VulkanPipelineBuilder& colorBlendAttachmentStates(
      std::vector<VkPipelineColorBlendAttachmentState>& states);
  /// @brief The index of the subpass of the render pass that the pipeline is used in. The default
  /// is 0.
  VulkanPipelineBuilder& subpass(uint32_t subpassIndex);

  [[nodiscard]] VkResult build(const VulkanFunctionTable& vf,
                               VkDevice device,
                               VkPipelineCreateFlags flags,
                               VkPipelineCache pipelineCache,
                               VkPipelineLayout pipelineLayout,
                               VkRenderPass renderPass,
                               VkPipeline* outPipeline,
                               const char* debugName = nullptr) noexcept;

  [[nodiscard]] static uint32_t getNumPipelinesCreated();

 private:
  std::vector<VkDynamicState> dynamicStates_;
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages_;
  VkPipelineVertexInputStateCreateInfo vertexInputState_;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly_;
  VkPipelineRasterizationStateCreateInfo rasterizationState_;
  VkPipelineMultisampleStateCreateInfo multisampleState_;
  VkPipelineDepthStencilStateCreateInfo depthStencilState_;
  std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentStates_;
  uint32_t subpass_ = 0;
  static uint32_t numPipelinesCreated;
};

[[nodiscard]] inline uint32_t VulkanPipelineBuilder::getNumPipelinesCreated() {
  return numPipelinesCreated;
}

class VulkanComputePipelineBuilder final {
 public:
  VulkanComputePipelineBuilder() = default;
  ~VulkanComputePipelineBuilder() = default;

  VulkanComputePipelineBuilder& shaderStage(VkPipelineShaderStageCreateInfo stage);

  [[nodiscard]] VkResult build(const VulkanFunctionTable& vf,
                               VkDevice device,
                               VkPipelineCreateFlags flags,
                               VkPipelineCache pipelineCache,
                               VkPipelineLayout pipelineLayout,
                               VkPipeline* outPipeline,
                               const char* debugName = nullptr) noexcept;

  [[nodiscard]] static uint32_t getNumPipelinesCreated();

 private:
  VkPipelineShaderStageCreateInfo shaderStage_;
  static uint32_t numPipelinesCreated;
};

[[nodiscard]] inline uint32_t VulkanComputePipelineBuilder::getNumPipelinesCreated() {
  return numPipelinesCreated;
}

} // namespace igl::vulkan
