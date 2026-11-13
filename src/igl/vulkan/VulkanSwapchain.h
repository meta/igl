/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanFence.h>
#include <igl/vulkan/VulkanFramebuffer.h>
#include <igl/vulkan/VulkanHelpers.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanImageView.h>
#include <igl/vulkan/VulkanTexture.h>

namespace igl::vulkan {

class VulkanContext;
class VulkanSemaphore;

class VulkanSwapchain final {
 public:
  VulkanSwapchain(VulkanContext& ctx, uint32_t width, uint32_t height);
  ~VulkanSwapchain();
  VulkanSwapchain(const VulkanSwapchain&) = delete;
  VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;
  VulkanSwapchain(VulkanSwapchain&&) = delete;
  VulkanSwapchain& operator=(VulkanSwapchain&&) = delete;

  Result acquireNextImage();
  /// False when an image view of a swapchain image could not be created: the swapchain is unusable.
  [[nodiscard]] bool valid() const noexcept {
    return valid_;
  }
  Result present(VkSemaphore waitSemaphore);
  [[nodiscard]] VkImage getCurrentVkImage() const {
    if (IGL_DEBUG_VERIFY(currentImageIndex_ < numSwapchainImages_)) {
      return swapchainTextures_[currentImageIndex_]->image.getVkImage();
    }
    return VK_NULL_HANDLE;
  }
  [[nodiscard]] VkImageView getCurrentVkImageView() const {
    if (IGL_DEBUG_VERIFY(currentImageIndex_ < numSwapchainImages_)) {
      return swapchainTextures_[currentImageIndex_]->imageView_.getVkImageView();
    }
    return VK_NULL_HANDLE;
  }

  std::shared_ptr<VulkanTexture> getCurrentDepthTexture() {
    if (!depthTexture_) {
      lazyAllocateDepthBuffer();
    }

    return depthTexture_;
  }

  std::shared_ptr<VulkanTexture> getCurrentVulkanTexture() {
    if (getNextImage_) {
      if (const Result result = acquireNextImage(); !result.isOk()) {
        return nullptr;
      }
    }

    if (IGL_DEBUG_VERIFY(currentImageIndex_ < numSwapchainImages_)) {
      return swapchainTextures_[currentImageIndex_];
    }

    return nullptr;
  }

  [[nodiscard]] VkImage getDepthVkImage() const;
  [[nodiscard]] VkImageView getDepthVkImageView() const;
  [[nodiscard]] uint32_t getWidth() const {
    return width_;
  }
  [[nodiscard]] uint32_t getHeight() const {
    return height_;
  }
  [[nodiscard]] VkExtent2D getExtent() const {
    return VkExtent2D{.width = width_, .height = height_};
  }

  [[nodiscard]] VkFormat getFormatColor() const {
    return surfaceFormat_.format;
  }

  [[nodiscard]] uint32_t getNumSwapchainImages() const {
    return numSwapchainImages_;
  }

  [[nodiscard]] uint32_t getCurrentImageIndex() const {
    return currentImageIndex_;
  }

  [[nodiscard]] VkSemaphore getSemaphore() const noexcept;

  [[nodiscard]] uint64_t getFrameNumber() const {
    return frameNumber_;
  }

 private:
  friend class CommandQueue;
  friend class VulkanContext;
  [[nodiscard]] VulkanFence createAcquireFence(uint32_t index) const;
  [[nodiscard]] VkSemaphore getPresentSemaphore() const noexcept;
  void lazyAllocateDepthBuffer() const;

 public:
  std::vector<VulkanSemaphore> acquireSemaphores;
  // Used to check whether the acquire semaphore can be used for acquiring
  // Based on
  // https://github.com/corporateshark/lightweightvk/blob/36597fae5c79ad6b310e4ed8c00e8acfa38b5aca/lvk/vulkan/VulkanClasses.h#L146
  std::vector<VulkanFence> acquireFences; // this can be removed once we switch to timeline
                                          // semaphores
  std::vector<uint64_t> timelineWaitValues;

 private:
  const VulkanContext& ctx_;
  VkQueue graphicsQueue_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t numSwapchainImages_ = 0;
  uint32_t currentImageIndex_ = 0;
  // Because the next acquired image's index is obtained _after_ requesting it (along with
  // semaphores and fences), the index of the semaphore and fence used to synchronize the current
  // swapchain image is different than the `currentImageIndex_`
  uint32_t currentSemaphoreIndex_ = 0;
  uint64_t frameNumber_ = 0; // increasing continuously without bound
  bool getNextImage_ = true;
  bool valid_ = true;
  VkSwapchainKHR swapchain_{};
  std::vector<VulkanSemaphore> presentSemaphores_;
  std::unique_ptr<std::shared_ptr<VulkanTexture>[]> swapchainTextures_;
  mutable std::shared_ptr<VulkanTexture> depthTexture_;
  VkSurfaceFormatKHR surfaceFormat_{};
};

} // namespace igl::vulkan
