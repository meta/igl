/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanSemaphore.h"

#include <igl/vulkan/Common.h>

namespace igl::vulkan {

VulkanSemaphore::VulkanSemaphore(const VulkanFunctionTable& vf,
                                 VkDevice device,
                                 bool exportable,
                                 const char* debugName) :
  vf_(&vf), device_(device), exportable_(exportable) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  const VkExportSemaphoreCreateInfo exportInfo = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
  };
  const VkSemaphoreCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = exportable ? &exportInfo : nullptr,
      .flags = 0,
  };
  initResult_ = vf_->vkCreateSemaphore(device_, &ci, nullptr, &vkSemaphore_);
  VK_ASSERT(initResult_);
  if (initResult_ != VK_SUCCESS) {
    // The handle a failed `vkCreateSemaphore()` writes is undefined: the destructor must not use
    // it.
    vkSemaphore_ = VK_NULL_HANDLE;
    return;
  }
  VK_ASSERT(ivkSetDebugObjectName(
      vf_, device_, VK_OBJECT_TYPE_SEMAPHORE, reinterpret_cast<uint64_t>(vkSemaphore_), debugName));
}

VulkanSemaphore::VulkanSemaphore(const VulkanFunctionTable& vf,
                                 VkDevice device,
                                 uint64_t initialValue,
                                 bool exportable,
                                 const char* debugName) :
  vf_(&vf), device_(device), exportable_(exportable) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  const VkExportSemaphoreCreateInfo exportInfo = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
  };
  const VkSemaphoreTypeCreateInfo semaphoreTypeCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .pNext = exportable ? &exportInfo : nullptr,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = initialValue,
  };
  const VkSemaphoreCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &semaphoreTypeCreateInfo,
      .flags = 0,
  };
  initResult_ = vf_->vkCreateSemaphore(device_, &ci, nullptr, &vkSemaphore_);
  VK_ASSERT(initResult_);
  if (initResult_ != VK_SUCCESS) {
    // The handle a failed `vkCreateSemaphore()` writes is undefined: the destructor must not use
    // it.
    vkSemaphore_ = VK_NULL_HANDLE;
    return;
  }
  VK_ASSERT(ivkSetDebugObjectName(
      vf_, device_, VK_OBJECT_TYPE_SEMAPHORE, reinterpret_cast<uint64_t>(vkSemaphore_), debugName));
}

VulkanSemaphore ::~VulkanSemaphore() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);

  if (device_ != VK_NULL_HANDLE) {
    // lifetimes of all VkSemaphore objects are managed explicitly
    // we do not use deferredTask() for them
    vf_->vkDestroySemaphore(device_, vkSemaphore_, nullptr);
  }
}

VulkanSemaphore::VulkanSemaphore(VulkanSemaphore&& other) noexcept {
  std::swap(vf_, other.vf_);
  std::swap(device_, other.device_);
  std::swap(vkSemaphore_, other.vkSemaphore_);
  std::swap(exportable_, other.exportable_);
  std::swap(initResult_, other.initResult_);
}

VulkanSemaphore& VulkanSemaphore::operator=(VulkanSemaphore&& other) noexcept {
  VulkanSemaphore tmp(std::move(other));
  std::swap(vf_, tmp.vf_);
  std::swap(device_, tmp.device_);
  std::swap(vkSemaphore_, tmp.vkSemaphore_);
  std::swap(exportable_, tmp.exportable_);
  std::swap(initResult_, tmp.initResult_);
  return *this;
}

VkSemaphore VulkanSemaphore::getVkSemaphore() const noexcept {
  return vkSemaphore_;
}

// Exportable semaphores are not used right now, so exclude from coverage
// FIXME_DEPRECATED_COVERAGE_EXCLUDE_START
int VulkanSemaphore::getFileDescriptor() const noexcept {
  if (!exportable_) {
    return -1;
  }
  const VkSemaphoreGetFdInfoKHR fdInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
      .pNext = nullptr,
      .semaphore = vkSemaphore_,
      .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
  };
  int fd = -1;
  const VkResult ok = vf_->vkGetSemaphoreFdKHR(device_, &fdInfo, &fd);
  if (ok == VK_SUCCESS) {
    return fd;
  }
  return -1;
}
// FIXME_DEPRECATED_COVERAGE_EXCLUDE_END

} // namespace igl::vulkan
