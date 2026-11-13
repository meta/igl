/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanHelpers.h>

namespace igl::vulkan {

/**
 * @brief Fences are used to synchronize CPU-GPU tasks. The VulkanFence class encapsulates the
 * creation and destruction of a vulkan fence object (VkFence). It stores an opaque handle for a
 * newly created fence object and for a device object.
 */
class VulkanFence final {
 public:
  VulkanFence(const VulkanFunctionTable& vf,
              VkDevice device,
              VkFlags flags,
              bool exportable = false,
              const char* debugName = nullptr);
  ~VulkanFence();

  VulkanFence(VulkanFence&& other) noexcept;
  VulkanFence& operator=(VulkanFence&& other) noexcept;

  VulkanFence(const VulkanFence&) = delete;
  VulkanFence& operator=(const VulkanFence&) = delete;

  bool reset() noexcept;
  bool wait(uint64_t timeoutNs = UINT64_MAX) noexcept;

  /** @brief Signals the fence on the provided queue.
   *
   * This does not wait for completion of the signal, it merely
   * executes the vkQueueSubmit() with the fence and no actual workload
   * so that the fence is signaled as soon as the queue workload executes
   * on the GPU.
   */
  bool signal(VkQueue queue);

  [[nodiscard]] bool exportable() const noexcept;

  /// The result of `vkCreateFence()`. A fence that failed to be created has no `VkFence`.
  [[nodiscard]] VkResult initResult() const noexcept {
    return initResult_;
  }

 private:
  VkResult initResult_ = VK_SUCCESS;

 public:
  // NOLINTBEGIN(readability-identifier-naming)
  const VulkanFunctionTable* vf_{};
  VkDevice device_ = VK_NULL_HANDLE;
  VkFence vkFence_ = VK_NULL_HANDLE;
  bool exportable_ = false;
  // NOLINTEND(readability-identifier-naming)
};

} // namespace igl::vulkan
