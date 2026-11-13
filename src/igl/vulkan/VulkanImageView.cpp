/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanImageView.h"

#include <type_traits>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanContext.h>

namespace igl::vulkan {

static_assert(std::is_trivially_copyable_v<VulkanImageViewCreateInfo>);

VulkanImageView::VulkanImageView(const VulkanContext& ctx,
                                 const VulkanImageViewCreateInfo& ci,
                                 const char* debugName) :
  VulkanImageView(ctx,
                  VkImageViewCreateInfo{
                      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                      .pNext = nullptr,
                      .flags = 0,
                      .image = ci.image,
                      .viewType = ci.viewType,
                      .format = ci.format,
                      .components = ci.components,
                      .subresourceRange = ci.subresourceRange,
                  },
                  debugName) {}

VulkanImageView::~VulkanImageView() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  destroy();
}

VulkanImageView::VulkanImageView(const VulkanContext& ctx,
                                 const VkImageViewCreateInfo& ci,
                                 const char* debugName) :
  ctx(&ctx), aspectMask(ci.subresourceRange.aspectMask) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  VkImageViewCreateInfo ciCopy(ci);

  VkSamplerYcbcrConversionInfo info{};

  if (!ci.pNext && igl::vulkan::getNumImagePlanes(ci.format) > 1) {
    info = ctx.getOrCreateYcbcrConversionInfo(ci.format);
    ciCopy.pNext = &info;
  }

  const VkDevice device = this->ctx->getVkDevice();
  const VkResult result = this->ctx->vf_.vkCreateImageView(device, &ciCopy, nullptr, &vkImageView);
  if (result != VK_SUCCESS) {
    IGL_LOG_ERROR("VulkanImageView: vkCreateImageView() failed (VkResult %d)\n",
                  static_cast<int>(result));
    // The view owns nothing and is not named.
    vkImageView = VK_NULL_HANDLE;
    this->ctx = nullptr;
    aspectMask = 0;
    initResult_ = result;
    return;
  }

  VK_ASSERT(ivkSetDebugObjectName(
      &this->ctx->vf_, device, VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)vkImageView, debugName));
}

VulkanImageView& VulkanImageView::operator=(VulkanImageView&& other) noexcept {
  destroy();
  ctx = other.ctx;
  vkImageView = other.vkImageView;
  aspectMask = other.aspectMask;
  initResult_ = other.initResult_;
  other.ctx = nullptr;
  other.vkImageView = VK_NULL_HANDLE;
  other.aspectMask = 0;
  other.initResult_ = VK_SUCCESS;
  return *this;
}

[[nodiscard]] bool VulkanImageView::valid() const {
  return ctx != nullptr;
}

void VulkanImageView::destroyNeverSubmitted() {
  if (!valid()) {
    return;
  }

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx);

  ctx->vf_.vkDestroyImageView(ctx->getVkDevice(), vkImageView, nullptr);
  vkImageView = VK_NULL_HANDLE;
  ctx = nullptr;
}

void VulkanImageView::destroy() {
  if (!valid()) {
    return;
  }

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx);

  ctx->deferredTask(std::packaged_task<void()>(
      [vf = &ctx->vf_, device = ctx->getVkDevice(), imageView = vkImageView]() {
        vf->vkDestroyImageView(device, imageView, nullptr);
      }));

  vkImageView = VK_NULL_HANDLE;
  ctx = nullptr;
}

} // namespace igl::vulkan
