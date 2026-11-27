/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#if !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES
#endif // !defined(VK_NO_PROTOTYPES)
#include <vulkan/vulkan_core.h>
#include <igl/Shader.h>

namespace igl::vulkan {

class VulkanShaderModule;

/// @brief Implements the igl::IShaderModule interface
class ShaderModule final : public IShaderModule {
 public:
  ShaderModule(ShaderModuleInfo info, std::shared_ptr<VulkanShaderModule> shaderModule);
  ~ShaderModule() override = default;

  [[nodiscard]] const VulkanShaderModule& getVulkanShaderModule() const;

  static VkShaderModule getVkShaderModule(const std::shared_ptr<IShaderModule>& shaderModule);

 private:
  std::shared_ptr<VulkanShaderModule> module_;
};

[[nodiscard]] inline const VulkanShaderModule& ShaderModule::getVulkanShaderModule() const {
  return *module_;
}

/// @brief Implements the igl::IShaderStages interface
class ShaderStages final : public IShaderStages {
 public:
  explicit ShaderStages(ShaderStagesDesc desc);
  ~ShaderStages() override = default;
};

/// @brief Implements the igl::IShaderLibrary interface
class ShaderLibrary : public IShaderLibrary {
 public:
  explicit ShaderLibrary(std::vector<std::shared_ptr<IShaderModule>> modules);
};

} // namespace igl::vulkan
