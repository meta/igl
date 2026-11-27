/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES
#endif // !defined(VK_NO_PROTOTYPES)

#include <vulkan/vulkan_core.h>

namespace igl::vulkan {

class VulkanContext;

/// @brief A wrapper around a Vulkan Framebuffer object
class VulkanFramebuffer final {
 public:
  /// @brief Constructs a VulkanFramebuffer object with the parameters provided and an optional
  /// debug name
  VulkanFramebuffer(const VulkanContext& ctx,
                    VkDevice device,
                    uint32_t width,
                    uint32_t height,
                    VkRenderPass renderPass,
                    size_t numAttachments,
                    const VkImageView* attachments,
                    const char* debugName = nullptr);

  /// @brief Queues the destruction of the framebuffer on the Vulkan context via a deferred task.
  /// For more details about deferred tasks, please refer to the igl::vulkan::VulkanContext class
  ~VulkanFramebuffer();

  VulkanFramebuffer(const VulkanFramebuffer&) = delete;
  VulkanFramebuffer& operator=(const VulkanFramebuffer&) = delete;
  VulkanFramebuffer(VulkanFramebuffer&&) = delete;
  VulkanFramebuffer& operator=(VulkanFramebuffer&&) = delete;

  [[nodiscard]] VkFramebuffer getVkFramebuffer() const;

 public:
  const VulkanContext& ctx;
  VkDevice vkDevice = VK_NULL_HANDLE;
  VkFramebuffer vkFramebuffer = VK_NULL_HANDLE;
};

[[nodiscard]] inline VkFramebuffer VulkanFramebuffer::getVkFramebuffer() const {
  return vkFramebuffer;
}

} // namespace igl::vulkan
