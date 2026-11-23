/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <shell/openxr/XrPlatform.h>
#include <shell/openxr/impl/XrSwapchainProviderImpl.h>
#include <igl/vulkan/VulkanTexture.h>

namespace igl::shell::openxr::mobile {
class XrSwapchainProviderImplVulkan final : public impl::XrSwapchainProviderImpl {
 public:
  explicit XrSwapchainProviderImplVulkan(TextureFormat preferredColorFormat);

  // NOLINTNEXTLINE(bugprone-exception-escape)
  [[nodiscard]] std::vector<int64_t> preferredColorFormats() const noexcept final {
    return {preferredColorFormat_};
  }
  // NOLINTNEXTLINE(bugprone-exception-escape)
  [[nodiscard]] std::vector<int64_t> preferredDepthFormats() const noexcept final {
    return {VK_FORMAT_D16_UNORM, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
  }

  void enumerateImages(IDevice& device,
                       XrSwapchain colorSwapchain,
                       XrSwapchain depthSwapchain,
                       const impl::SwapchainImageInfo& swapchainImageInfo,
                       uint8_t numViews) noexcept final;

  [[nodiscard]] SurfaceTextures getSurfaceTextures(
      IDevice& device,
      XrSwapchain colorSwapchain,
      XrSwapchain depthSwapchain,
      const impl::SwapchainImageInfo& swapchainImageInfo,
      uint8_t numViews) noexcept final;

 private:
  int64_t preferredColorFormat_;
  std::vector<std::shared_ptr<vulkan::VulkanTexture>> vulkanColorTextures_;
  std::vector<std::shared_ptr<vulkan::VulkanTexture>> vulkanDepthTextures_;
};
} // namespace igl::shell::openxr::mobile
