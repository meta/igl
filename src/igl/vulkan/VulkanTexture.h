/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanImageView.h>

namespace igl::vulkan {

class VulkanTexture final {
 public:
  VulkanTexture(VulkanImage&& image, VulkanImageView&& imageView);
  ~VulkanTexture() = default;

  VulkanTexture(const VulkanTexture&) = delete;
  VulkanTexture& operator=(const VulkanTexture&) = delete;
  VulkanTexture(VulkanTexture&&) noexcept = default;
  VulkanTexture& operator=(VulkanTexture&&) noexcept = default;

 public:
  // Set on a texture view, whose `image` does not own the underlying VkImage: the texture that owns
  // it. Declared first so it is released after this view's image view.
  // NOLINTNEXTLINE(readability-identifier-naming)
  std::shared_ptr<VulkanTexture> baseTexture_;
  VulkanImage image;
  // NOLINTNEXTLINE(readability-identifier-naming)
  VulkanImageView imageView_;
  // an index into VulkanContext::textures_
  // NOLINTNEXTLINE(readability-identifier-naming)
  uint32_t textureId_ = 0;
};

} // namespace igl::vulkan
