/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/Buffer.h>
#include <igl/vulkan/Common.h>

namespace igl::vulkan {

class Device;
class VulkanBuffer;

/// @brief Implements the igl::IBuffer interface for Vulkan. Contains one or more VulkanBuffers,
/// depending on the type of buffer this class represents. If this class represents a ring buffer,
/// then there will be multiple VulkanBuffers, each with its own index. Otherwise it contains only
/// one VulkanBuffer object.
class Buffer final : public IBuffer {
  friend class Device;

 public:
  explicit Buffer(const Device& device);
  ~Buffer() override = default;

  Result upload(const void* data, const BufferRange& range) override;

  void* map(const BufferRange& range, Result* outResult) override;

  void unmap() override;

  [[nodiscard]] BufferDesc::BufferAPIHint requestedApiHints() const noexcept override;
  [[nodiscard]] BufferDesc::BufferAPIHint acceptedApiHints() const noexcept override;
  [[nodiscard]] ResourceStorage storage() const noexcept override;

  [[nodiscard]] size_t getSizeInBytes() const override;
  [[nodiscard]] uint64_t gpuAddress(size_t offset) const override;

  [[nodiscard]] BufferDesc::BufferType getBufferType() const override {
    return desc_.type;
  }

  /// @brief Returns the current VkBuffer. For a Ring buffer, this and gpuAddress() mark the current
  /// copy as read by GPU work (see VulkanContext::markRingBufferRead()), so GPU work must get the
  /// buffer through one of them.
  [[nodiscard]] VkBuffer getVkBuffer() const;
  [[nodiscard]] VkBufferUsageFlags getBufferUsageFlags() const;

  [[nodiscard]] bool isRingBuffer() const noexcept {
    return isRingBuffer_;
  }

  /// @brief Returns the current active VulkanBuffer object managed by this class. Since this class
  /// may be used as a Ring Buffer, the active buffer is the buffer currently being accessed.
  /// Unlike getVkBuffer(), this does not mark a Ring buffer copy as read.
  [[nodiscard]] const std::unique_ptr<VulkanBuffer>& currentVulkanBuffer() const;

 private:
  const Device& device_;
  BufferDesc desc_;
  bool isRingBuffer_ = false;
  uint32_t previousBufferIndex_ = UINT32_MAX;
  std::unique_ptr<std::unique_ptr<VulkanBuffer>[]> buffers_;
  std::unique_ptr<uint8_t[]> localData_;
  std::unique_ptr<BufferRange[]> bufferPatches_;
  uint32_t bufferCount_ = 0;

  Result create(const BufferDesc& desc);

  // Function determines smallest starting and largest ending offset by iterating over all
  // bufferPatches_, and returns it in the form of buffer range
  [[nodiscard]] BufferRange getUpdateRange() const;
  void extendUpdateRange(uint32_t ringBufferIndex, const BufferRange& range);
  void resetUpdateRange(uint32_t ringBufferIndex, const BufferRange& range);

  // Used for map/unmap API for DEVICE_LOCAL buffers
  std::vector<uint8_t> tmpBuffer_;
  BufferRange mappedRange_ = {};
  BufferDesc::BufferAPIHint requestedApiHints_ = 0;
};

} // namespace igl::vulkan
