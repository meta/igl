/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <utility>
#include <igl/vulkan/VulkanFunctionTable.h> // IWYU pragma: keep

namespace igl::vulkan {

class VulkanContext;

struct VulkanImageViewCreateInfo {
  VkImage image = VK_NULL_HANDLE;
  VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkComponentMapping components = {
      .r = VK_COMPONENT_SWIZZLE_IDENTITY,
      .g = VK_COMPONENT_SWIZZLE_IDENTITY,
      .b = VK_COMPONENT_SWIZZLE_IDENTITY,
      .a = VK_COMPONENT_SWIZZLE_IDENTITY,
  };
  VkImageSubresourceRange subresourceRange = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
  };
};

/**
 * @brief VulkanImageView is a RAII wrapper for VkImageView.
 * The device member is not managed by this class (it is used to destroy the imageView).
 */
class VulkanImageView final {
 public:
  explicit VulkanImageView() = default;
  /**
   * @brief Creates the VulkanImageView object which stores a handle to a newly created VkImageView.
   * The imageView is created from the device, image, and other parameters with a name that can be
   * used for debugging.
   */
  VulkanImageView(const VulkanContext& ctx,
                  const VulkanImageViewCreateInfo& ci,
                  const char* debugName = nullptr);

  VulkanImageView(const VulkanContext& ctx,
                  const VkImageViewCreateInfo& ci,
                  const char* debugName = nullptr);

  ~VulkanImageView();

  VulkanImageView(const VulkanImageView&) = delete;
  VulkanImageView& operator=(const VulkanImageView&) = delete;

  VulkanImageView(VulkanImageView&& other) noexcept {
    *this = std::move(other);
  }
  VulkanImageView& operator=(VulkanImageView&& other) noexcept;

  /**
   * @brief Returns Vulkan's opaque handle to the imageView object
   */
  [[nodiscard]] VkImageView getVkImageView() const {
    return vkImageView;
  }
  /**
   * @brief Returns true if the object is valid: its VkImageView was created.
   */
  [[nodiscard]] bool valid() const;
  /**
   * @brief The result of the failed `vkCreateImageView()` (`VK_SUCCESS` for a valid view). A view
   * whose creation failed owns no Vulkan object and was not named.
   */
  [[nodiscard]] VkResult initResult() const noexcept {
    return initResult_;
  }
  /**
   * @brief Destroys the view at once instead of through a deferred task. Only for a view that no
   * GPU work can reference because it was refused right after it was created. No-op if invalid.
   */
  void destroyNeverSubmitted();
  /**
   * @brief Returns the VkImageAspectFlags used to create the imageView
   */
  [[nodiscard]] VkImageAspectFlags getVkImageAspectFlags() const {
    return aspectMask;
  }

 public:
  const VulkanContext* ctx = nullptr;
  VkImageView vkImageView = VK_NULL_HANDLE;
  VkImageAspectFlags aspectMask = 0;

 private:
  VkResult initResult_ = VK_SUCCESS;
  void destroy();
};

} // namespace igl::vulkan
