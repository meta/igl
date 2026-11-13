/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/vulkan/Common.h>

namespace igl::vulkan {

class VulkanContext;

/// @brief A wrapper around a Vulkan Buffer object that provides convenience functions for
/// uploading/downloading data to/from the GPU.
class VulkanBuffer {
 public:
  /** @brief Creates a new VulkanBuffer with a given size, usage flags, memory property flags, and
   * an optional debug name. Uses VMA if IGL is built with VMA support. If memory flags specify
   * that the buffer is visible by the host (the CPU), then the buffer's memory will be mapped into
   * the application's address space and can be accessed directly.
   *
   * The constructor stops at the first failing step and records its result: check `valid()` before
   * using the buffer. An invalid buffer owns no Vulkan object and is not mapped, so its upload,
   * download, flush and invalidate operations return early on their existing mapped-pointer
   * checks; debug builds also assert.
   */
  VulkanBuffer(const VulkanContext& ctx,
               VkDevice device,
               VkDeviceSize bufferSize,
               VkBufferUsageFlags usageFlags,
               VkMemoryPropertyFlags memFlags,
               const char* debugName = nullptr);
  ~VulkanBuffer();

  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;
  VulkanBuffer(VulkanBuffer&&) = delete;
  VulkanBuffer& operator=(VulkanBuffer&&) = delete;

  /** @brief Uploads the data located at `data` into the buffer on the device with the provided
   * `offset`. Only mapped host-visible buffers can be uploaded this way. All other GPU buffers
   * should use a temporary staging buffer. If the buffer's data has not been mapped, this function
   * is a no-op. This function is synchronous and the data is expected to be available when the
   * function returns
   */
  void bufferSubData(size_t offset, size_t size, const void* data);

  /** @brief Downloads the data located at `offset` from the buffer on the device to the location
   * pointed by `data`. Only mapped host-visible buffers can be downloaded this way. All other GPU
   * buffers should use a temporary staging buffer. If the buffer's data has not been mapped, this
   * function is a no-op. This function is synchronous and the data is expected to be available
   * when the function returns.
   */
  void getBufferSubData(size_t offset, size_t size, void* data) const;
  [[nodiscard]] uint8_t* getMappedPtr() const {
    return static_cast<uint8_t*>(mappedPtr_);
  }

  /// @brief Whether the buffer's memory has been mapped.
  [[nodiscard]] bool isMapped() const {
    return mappedPtr_ != nullptr;
  }

  /// @brief Flushes the mapped memory range to make it visible to the GPU.
  void flushMappedMemory(VkDeviceSize offset, VkDeviceSize size) const;

  /// @brief Invalidates the mapped memory range to make it visible to the CPU.
  void invalidateMappedMemory(VkDeviceSize offset, VkDeviceSize size) const;

  [[nodiscard]] VkBuffer getVkBuffer() const {
    return vkBuffer_;
  }
  [[nodiscard]] VkDeviceAddress getVkDeviceAddress() const {
    IGL_DEBUG_ASSERT(valid());
    IGL_DEBUG_ASSERT(vkDeviceAddress_, "Make sure config.enableBufferDeviceAddress is enabled");
    return vkDeviceAddress_;
  }
  [[nodiscard]] VkDeviceSize getSize() const {
    return bufferSize_;
  }
  [[nodiscard]] VkMemoryPropertyFlags getMemoryPropertyFlags() const {
    return memFlags_;
  }
  [[nodiscard]] VkBufferUsageFlags getBufferUsageFlags() const {
    return usageFlags_;
  }
  [[nodiscard]] bool isCoherentMemory() const {
    return isCoherentMemory_;
  }

  /// @brief Whether construction succeeded: the buffer and its memory exist.
  [[nodiscard]] bool valid() const noexcept {
    return initResult_ == VK_SUCCESS;
  }

  /// @brief The result of the first construction step that failed, `VK_SUCCESS` for a valid buffer.
  [[nodiscard]] VkResult initResult() const noexcept {
    return initResult_;
  }

  /// @brief Destroys the buffer and its memory at once, instead of through a deferred task, and
  /// leaves the object invalid. Only for a buffer that no GPU work can reference because it was
  /// created and then refused before any use (a failed aggregate). No-op for an invalid buffer.
  void discardNeverSubmitted();

 private:
  /// Clears the handles and records `result`. Only a failed allocation fails the constructor, so
  /// there is nothing to destroy.
  void failInit(VkResult result);

  const VulkanContext& ctx_;
  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer vkBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory vkMemory_ = VK_NULL_HANDLE;
  VmaAllocation vmaAllocation_ = VK_NULL_HANDLE;
  VkDeviceAddress vkDeviceAddress_ = 0;
  VkDeviceSize bufferSize_ = 0;
  VkBufferUsageFlags usageFlags_ = 0;
  VkMemoryPropertyFlags memFlags_ = 0;
  void* mappedPtr_ = nullptr;
  bool isCoherentMemory_ = false;
  VkResult initResult_ = VK_SUCCESS;
};

} // namespace igl::vulkan
