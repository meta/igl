/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <deque>
#include <memory>
#include <vector>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanHelpers.h>
#include <igl/vulkan/VulkanImmediateCommands.h>

namespace igl::vulkan {

class VulkanBuffer;
class VulkanContext;
class VulkanImage;

/** @brief Manages data transfers between the CPU and the GPU.
 * This class automatically allocates and uses a staging buffer when transferring data between the
 * CPU and device-local resources. Transfers between the CPU and host-visible resources are copied
 * directly to the device, without the intermediary copy to the staging buffer. The staging buffer
 * is lazily allocated and grows as needed when the uploaded data cannot be transferred in small
 * chunks and is larger than the current staging buffer's size. The maximum size of the buffer is
 * determined at runtime and is the minimum between VkPhysicalDeviceLimits::VkPhysicalDeviceLimits
 * and 256 MB. Some architectures limit the size of staging buffers to 256MB (buffers that are both
 * host and device visible).
 */
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class VulkanStagingDevice final {
 public:
  explicit VulkanStagingDevice(VulkanContext& ctx);
  ~VulkanStagingDevice() = default;

  VulkanStagingDevice(const VulkanStagingDevice&) = delete;
  VulkanStagingDevice& operator=(const VulkanStagingDevice&) = delete;

  std::unique_ptr<VulkanImmediateCommands> immediate;

  /** @brief Uploads the data at location `data` with the provided size (in bytes) to the
   * VulkanBuffer object on the device at offset `dstOffset`. The upload operation is asynchronous
   * and the data may or may not be available to the GPU when the function returns
   */
  void bufferSubData(VulkanBuffer& buffer, size_t dstOffset, size_t size, const void* data);

  /** @brief Downloads the data with the provided size (in bytes) from the VulkanBuffer object on
   * the device, and at the offset provided, to the location referenced by the pointer `data`. The
   * function is synchronous and the data donwloaded from the device is expected to be available in
   * the location pointed by `data` upon return
   */
  void getBufferSubData(const VulkanBuffer& buffer, size_t srcOffset, size_t size, void* data);

  /// @brief Uploads the texture data pointed by `data` to the VulkanImage object on the device. The
  /// data may span the entire texture or just part of it. The upload operation is asynchronous and
  /// the data may or may not be available to the GPU when the function returns
  void imageData(const VulkanImage& image,
                 TextureType type,
                 const TextureRangeDesc& range,
                 const TextureFormatProperties& properties,
                 uint32_t bytesPerRow,
                 VkImageAspectFlags aspectFlags,
                 const void* data);

  /** @brief Downloads the texture data from the VulkanImage object on the device to the location
   * pointed by `data`. The data requested may span the entire texture or just part of it. The
   * download operation is synchronous and the data is expected to be available at location `data`
   * upon return
   */
  void getImageData2D(VkImage srcImage,
                      uint32_t level,
                      uint32_t layer,
                      const VkRect2D& imageRegion,
                      TextureFormatProperties properties,
                      VkFormat format,
                      VkImageLayout layout,
                      VkImageAspectFlags aspectFlags,
                      void* data,
                      uint32_t bytesPerRow,
                      bool flipImageVertical);

  /// @brief Returns the size of staging buffer available for use
  [[nodiscard]] VkDeviceSize getFreeStagingBufferSize() const {
    return freeStagingBufferSize_;
  }

  /// @brief Returns the maximum possible size of the staging buffer in bytes
  [[nodiscard]] VkDeviceSize getMaxStagingBufferSize() const {
    return maxStagingBufferSize_;
  }

  /// @brief Returns the alignment required for staging copies of this image format.
  [[nodiscard]] VkDeviceSize getImageCopyAlignment(const TextureFormatProperties& properties) const;

  /// @brief Function to merge regions of the staging buffer that are contiguous, and deallocate
  /// unused staging buffers.
  void mergeRegionsAndFreeBuffers();

 private:
  struct MemoryRegion {
    VkDeviceSize offset = 0u;
    VkDeviceSize size = 0u;
    VkDeviceSize alignedSize = 0u;
    VulkanImmediateCommands::SubmitHandle handle;
    uint32_t stagingBufferIndex = 0u;
  };

  /**
   * @brief Searches for an available block in the staging buffer that is as large as the size
   * requested. If the only contiguous block of memory available is smaller than the requested size,
   * the function returns the amount of memory it was able to find.
   *
   * @param contiguous
   * if true, the function will return a region big enough to accommodate full requested size.
   * if false, the function may return a region smaller than the requested size.
   * @return The offset of the free memory block on the staging buffer and the size of the block
   * found.
   */
  [[nodiscard]] MemoryRegion nextFreeBlock(VkDeviceSize size, bool contiguous);

  [[nodiscard]] VkDeviceSize getAlignedSize(VkDeviceSize size) const;

  /// @brief Waits for all memory blocks to become available and resets the staging device's
  /// internal state
  void waitAndReset();

  /**
   * @brief Returns true if the staging buffer cannot store the size requested
   * @param sizeNeeded the size of the memory block requested
   * @param contiguous if true, the function returns true if a contiguous block of memory cannot
   * accommodate sizeNeeded
   **/
  [[nodiscard]] bool shouldAllocateStagingBuffer(VkDeviceSize sizeNeeded,
                                                 bool contiguous) const noexcept;

  /// @brief Returns the next size to allocate for the staging buffer given the requested size
  [[nodiscard]] VkDeviceSize nextSize(VkDeviceSize requestedSize) const;

  /// @brief Allocates a new staging buffer to a size that is at least as large as the requested
  /// size
  void allocateStagingBuffer(VkDeviceSize minimumSize);

 private:
  VulkanContext& ctx_;
  std::vector<std::unique_ptr<VulkanBuffer>> stagingBuffers_;

  /// @brief available free memory in staging buffer
  VkDeviceSize freeStagingBufferSize_ = 0;
  /// @brief Maximum staging buffer size, limited by some architectures
  VkDeviceSize maxStagingBufferSize_ = 0;
  /// @brief Used to track the current staging buffer's id. Updated every time the staging buffer
  /// grows, it is used as the debug name for the staging buffer for easily tracking it during
  /// debugging
  uint32_t stagingBufferCounter_ = 0;

  /**
   * @brief Stores the used and unused blocks of memory in the staging buffer. There is no
   * distinction between used and unused blocks in the deque, as we always `wait` on each block
   * before using them. Older blocks are stored at the front of the deque, while used/busy
   * ones are stored at the end. Older blocks have a higher chance of being unused (not waiting for
   * the associated command buffer to finish)
   */
  std::deque<MemoryRegion> regions_;
};

} // namespace igl::vulkan
