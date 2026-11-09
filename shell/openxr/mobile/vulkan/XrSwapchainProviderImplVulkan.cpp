/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/openxr/mobile/vulkan/XrSwapchainProviderImplVulkan.h>

#include <fmt/format.h>
#include <shell/openxr/XrLog.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanImageView.h>

namespace igl::shell::openxr::mobile {
namespace {
void enumerateSwapchainImages(
    IDevice& device,
    XrSwapchain swapchain,
    VkFormat format,
    const impl::SwapchainImageInfo& swapchainImageInfo,
    uint8_t numViews,
    VkImageUsageFlags usageFlags,
    VkImageAspectFlags aspectMask,
    std::vector<std::shared_ptr<igl::vulkan::VulkanTexture>>& outVulkanTextures) {
  uint32_t numImages = 0;
  XR_CHECK(xrEnumerateSwapchainImages(swapchain, 0, &numImages, nullptr));

  IGL_LOG_INFO("XRSwapchain numImages: %d\n", numImages);

  std::vector<XrSwapchainImageVulkanKHR> images(
      numImages, {.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR, .next = nullptr});
  XR_CHECK(
      xrEnumerateSwapchainImages(swapchain,
                                 numImages,
                                 &numImages,
                                 reinterpret_cast<XrSwapchainImageBaseHeader*>(images.data())));

  const auto& actualDevice = static_cast<igl::vulkan::Device&>(device);
  const auto& ctx = actualDevice.getVulkanContext();
  outVulkanTextures.reserve(numImages);

  const bool isDepth = ((aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0);

  for (uint32_t i = 0; i < numImages; ++i) {
    auto image = igl::vulkan::VulkanImage(
        ctx,
        images[i].image,
        fmt::format("Image: swapchain {} #{}", isDepth ? "depth" : "color", i).c_str(),
        usageFlags,
        true,
        VkExtent3D{.width = swapchainImageInfo.imageWidth,
                   .height = swapchainImageInfo.imageHeight,
                   .depth = 0},
        VK_IMAGE_TYPE_2D,
        format,
        1,
        numViews);

    auto imageView = image.createImageView(
        numViews > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
        format,
        aspectMask,
        0,
        VK_REMAINING_MIP_LEVELS,
        0,
        numViews,
        fmt::format("Image View: swapchain {} #{}", isDepth ? "depth" : "color", i).c_str());
    outVulkanTextures.emplace_back(
        std::make_shared<igl::vulkan::VulkanTexture>(std::move(image), std::move(imageView)));
  }
}

std::shared_ptr<ITexture> getSurfaceTexture(
    IDevice& device,
    const XrSwapchain& swapchain,
    const impl::SwapchainImageInfo& swapchainImageInfo,
    uint8_t numViews,
    const std::vector<std::shared_ptr<igl::vulkan::VulkanTexture>>& vulkanTextures,
    VkFormat externalTextureFormat,
    std::vector<std::shared_ptr<ITexture>>& inOutTextures) {
  uint32_t imageIndex = 0;
  const XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
  XR_CHECK(xrAcquireSwapchainImage(swapchain, &acquireInfo, &imageIndex));

  XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
  waitInfo.timeout = XR_INFINITE_DURATION;
  XR_CHECK(xrWaitSwapchainImage(swapchain, &waitInfo));

  const auto& vulkanTexture = vulkanTextures[imageIndex];

  if (imageIndex >= inOutTextures.size()) {
    inOutTextures.resize(static_cast<size_t>(imageIndex) + 1, nullptr);
  }

  auto& actualDevice = static_cast<igl::vulkan::Device&>(device);
  const auto iglFormat = vulkan::vkFormatToTextureFormat(externalTextureFormat);
  const auto texture = inOutTextures[imageIndex];
  // allocate new drawable textures if its null or mismatches in size or format
  if (!texture || swapchainImageInfo.imageWidth != texture->getSize().width ||
      swapchainImageInfo.imageHeight != texture->getSize().height ||
      iglFormat != texture->getProperties().format) {
    const TextureDesc textureDesc =
        numViews > 1 ? TextureDesc::new2DArray(iglFormat,
                                               swapchainImageInfo.imageWidth,
                                               swapchainImageInfo.imageHeight,
                                               numViews,
                                               TextureDesc::TextureUsageBits::Attachment,
                                               "SwapChain Texture")
                     : TextureDesc::new2D(iglFormat,
                                          swapchainImageInfo.imageWidth,
                                          swapchainImageInfo.imageHeight,
                                          TextureDesc::TextureUsageBits::Attachment,
                                          "SwapChain Texture");

    inOutTextures[imageIndex] =
        std::make_shared<igl::vulkan::Texture>(actualDevice, vulkanTexture, textureDesc);
  }

  return inOutTextures[imageIndex];
}
} // namespace

XrSwapchainProviderImplVulkan::XrSwapchainProviderImplVulkan(TextureFormat preferredColorFormat) {
  preferredColorFormat_ = vulkan::textureFormatToVkFormat(preferredColorFormat);
}

void XrSwapchainProviderImplVulkan::enumerateImages(
    IDevice& device,
    XrSwapchain colorSwapchain,
    XrSwapchain depthSwapchain,
    const impl::SwapchainImageInfo& swapchainImageInfo,
    uint8_t numViews) noexcept { // NOLINT(bugprone-exception-escape)
  enumerateSwapchainImages(device,
                           colorSwapchain,
                           static_cast<VkFormat>(swapchainImageInfo.colorFormat),
                           swapchainImageInfo,
                           numViews,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT,
                           vulkanColorTextures_);

  const VkFormat vkDepthFormat = static_cast<VkFormat>(swapchainImageInfo.depthFormat);
  VkImageAspectFlags depthAspectFlags = 0;
  if (igl::vulkan::hasDepth(vkDepthFormat)) {
    depthAspectFlags |= VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  if (igl::vulkan::hasStencil(vkDepthFormat)) {
    depthAspectFlags |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  enumerateSwapchainImages(device,
                           depthSwapchain,
                           vkDepthFormat,
                           swapchainImageInfo,
                           numViews,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                           depthAspectFlags,
                           vulkanDepthTextures_);
}

SurfaceTextures XrSwapchainProviderImplVulkan::getSurfaceTextures(
    IDevice& device,
    XrSwapchain colorSwapchain,
    XrSwapchain depthSwapchain,
    const impl::SwapchainImageInfo& swapchainImageInfo,
    uint8_t numViews) noexcept { // NOLINT(bugprone-exception-escape)
  auto colorTexture = getSurfaceTexture(device,
                                        colorSwapchain,
                                        swapchainImageInfo,
                                        numViews,
                                        vulkanColorTextures_,
                                        static_cast<VkFormat>(swapchainImageInfo.colorFormat),
                                        colorTextures_);
  auto depthTexture = getSurfaceTexture(device,
                                        depthSwapchain,
                                        swapchainImageInfo,
                                        numViews,
                                        vulkanDepthTextures_,
                                        static_cast<VkFormat>(swapchainImageInfo.depthFormat),
                                        depthTextures_);

  return {.color = colorTexture, .depth = depthTexture};
}
} // namespace igl::shell::openxr::mobile
