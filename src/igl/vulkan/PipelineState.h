/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/vulkan/Common.h>
#include <igl/vulkan/util/SpvReflection.h>

namespace igl::vulkan {

class VulkanContext;
class VulkanDescriptorSetLayout;

class PipelineState {
 public:
  PipelineState(const VulkanContext& ctx,
                IShaderStages* stages,
                std::shared_ptr<ISamplerState> immutableSamplers[IGL_TEXTURE_SAMPLERS_MAX],
                uint32_t isDynamicBufferMask,
                const char* debugName);
  virtual ~PipelineState() = default;

  PipelineState(const PipelineState&) = delete;
  PipelineState(PipelineState&&) = delete;
  PipelineState& operator=(const PipelineState&) = delete;
  PipelineState& operator=(PipelineState&&) = delete;

  [[nodiscard]] VkPipelineLayout getVkPipelineLayout() const;

  [[nodiscard]] const util::SpvModuleInfo& getSpvModuleInfo() const {
    return info;
  }

 private:
  void initializeSpvModuleInfoFromShaderStages(const VulkanContext& ctx, IShaderStages* stages);

 public:
  util::SpvModuleInfo info;

  VkPushConstantRange pushConstantRange = {};
  VkShaderStageFlags stageFlags = 0;

  mutable VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  // the last seen VkDescriptorSetLayout from VulkanContext::dslBindless_
  mutable VkDescriptorSetLayout lastBindlessVkDescriptorSetLayout = VK_NULL_HANDLE;

  std::unique_ptr<VulkanDescriptorSetLayout> dslCombinedImageSamplers;
  std::unique_ptr<VulkanDescriptorSetLayout> dslBuffers;
  std::unique_ptr<VulkanDescriptorSetLayout> dslStorageImages;
};

} // namespace igl::vulkan
