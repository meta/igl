/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/Buffer.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <igl/IGLSafeC.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanStagingDevice.h>

namespace igl::vulkan {

Buffer::Buffer(const Device& device) : device_(device) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
}

Result Buffer::create(const BufferDesc& desc) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  desc_ = desc;

  if (desc_.storage == igl::ResourceStorage::Invalid) {
    desc_.storage = igl::ResourceStorage::Private;
  }

  const VulkanContext& ctx = device_.getVulkanContext();

  if (!ctx.useStagingForBuffers_ && (desc_.storage == ResourceStorage::Private)) {
    desc_.storage = ResourceStorage::Shared;
  }

  /* Use staging device to transfer data into the buffer when the storage is private to the device
   */
  VkBufferUsageFlags usageFlags =
      (desc_.storage == ResourceStorage::Private)
          ? VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
          : 0;

  const VkBufferUsageFlags optionalBDA = ctx.features().has_VK_KHR_buffer_device_address
                                             ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR
                                             : 0;

  if (desc_.type == 0) {
    return Result(Result::Code::InvalidOperation, "Invalid buffer type");
  }

  if ((desc_.type & BufferDesc::BufferTypeBits::Index) != 0) {
    usageFlags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  if ((desc_.type & BufferDesc::BufferTypeBits::Vertex) != 0) {
    usageFlags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if ((desc_.type & BufferDesc::BufferTypeBits::Uniform) != 0) {
    // TRANSFER_DST lets ICommandBuffer::updateBuffer() write any uniform buffer, including host
    // visible ones. Ring-buffer updates also copy from the previous allocation.
    usageFlags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | optionalBDA;
  }

  if ((desc_.type & BufferDesc::BufferTypeBits::Storage) != 0) {
    usageFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | optionalBDA;
  }

  if ((desc_.type & BufferDesc::BufferTypeBits::Indirect) != 0) {
    usageFlags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | optionalBDA;
  }

  const VkMemoryPropertyFlags memFlags = resourceStorageToVkMemoryPropertyFlags(desc_.storage);

  // Store the flag that determines if this buffer contains sub-allocations (i.e. is a ring-buffer)
  isRingBuffer_ = ((desc_.hint & BufferDesc::BufferAPIHintBits::Ring) != 0);

  bufferCount_ = isRingBuffer_ ? device_.getVulkanContext().config_.maxResourceCount : 1u;

  buffers_ = std::make_unique<std::unique_ptr<VulkanBuffer>[]>(bufferCount_);
  bufferPatches_ = std::make_unique<BufferRange[]>(bufferCount_);

  // This is used to generate a unique number for unnamed buffers
  static uint32_t bufferId = 0;
  Result result;
  for (size_t bufferIndex = 0; bufferIndex < bufferCount_; ++bufferIndex) {
    const std::string subBufferName =
        bufferCount_ > 1 ? " - sub-buffer " + std::to_string(bufferIndex) : "";
    const std::string bufferName = !desc_.debugName.empty()
                                       ? IGL_FORMAT("Buffer: {}", desc_.debugName + subBufferName)
                                       : IGL_FORMAT("Buffer {}{}", bufferId, subBufferName);
    buffers_[bufferIndex] =
        ctx.createBuffer(desc_.length, usageFlags, memFlags, &result, bufferName.c_str());
    if (!result.isOk() || buffers_[bufferIndex] == nullptr) {
      // Nothing is half-built: the sub-buffers created so far were never used, so they are
      // destroyed at once and the buffer has none.
      IGL_DEBUG_ASSERT(!result.isOk(), "createBuffer() returned null with an OK result");
      for (size_t created = 0; created < bufferIndex; ++created) {
        buffers_[created]->discardNeverSubmitted();
      }
      buffers_.reset();
      bufferPatches_.reset();
      return result.isOk() ? Result(Result::Code::RuntimeError, "Cannot create the buffer")
                           : result;
    }
  }
  if (desc_.debugName.empty()) {
    ++bufferId;
  }

  // allocate local data for ring-buffer only if Vulkan Buffers are not mapped to the CPU
  if (isRingBuffer_ && !buffers_[0]->isMapped()) {
    // Resize the local copy of the data
    localData_ = std::make_unique<uint8_t[]>(desc_.length);
  }

  return result;
}

const std::unique_ptr<VulkanBuffer>& Buffer::currentVulkanBuffer() const {
  IGL_DEBUG_ASSERT(buffers_, "There are no sub-allocations available for this buffer");
  return buffers_[isRingBuffer_ ? device_.getVulkanContext().currentSyncIndex() : 0u];
}

BufferRange Buffer::getUpdateRange() const {
  size_t start = std::numeric_limits<size_t>::max();
  size_t end = 0;

  for (uint32_t i = 0; i < bufferCount_; ++i) {
    const auto& bufferPatch = bufferPatches_[i];
    // skip this buffer, if update size is zero
    if (bufferPatch.size == 0) {
      continue;
    }
    end = std::max(end, bufferPatch.offset + bufferPatch.size);
    start = std::min(start, bufferPatch.offset);
  }

  // If there is no new data, return an empty range to indicate that no data is available
  if (start == std::numeric_limits<size_t>::max()) {
    return {};
  }

  return {end - start, start};
}

void Buffer::extendUpdateRange(uint32_t ringBufferIndex, const BufferRange& range) {
  auto& bufferPatch = bufferPatches_[ringBufferIndex];
  const size_t start = std::min(bufferPatch.offset, range.offset);
  const size_t end = std::max(bufferPatch.offset + bufferPatch.size, range.size + range.offset);

  bufferPatch.offset = start;
  bufferPatch.size = end - start;
}

void Buffer::resetUpdateRange(uint32_t ringBufferIndex, const BufferRange& range) {
  bufferPatches_[ringBufferIndex] = range;
}

Result Buffer::upload(const void* data, const BufferRange& range) {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(data)) {
    return Result();
  }

  if (!IGL_DEBUG_VERIFY(range.offset + range.size <= desc_.length)) {
    return Result(Result::Code::ArgumentOutOfRange, "Out of range");
  }

  // To handle an upload to a ring-buffer, we update the local copy first and upload the entire
  // local data to the device below
  const VulkanContext& ctx = device_.getVulkanContext();
  if (isRingBuffer_) {
    // get the current ring buffer index
    const uint32_t currentBufferIndex = device_.getVulkanContext().currentSyncIndex();
    uint8_t* prevDataPtr = nullptr; // pointer to the previous local copy of the data
    BufferRange currentUpdateRange = range;
    if (currentBufferIndex != previousBufferIndex_) {
      prevDataPtr = previousBufferIndex_ < bufferCount_
                        ? buffers_[previousBufferIndex_]->getMappedPtr()
                        : nullptr;
      // if the index has changed update the index
      previousBufferIndex_ = currentBufferIndex;
      // reset update range at the current index, using input range
      resetUpdateRange(currentBufferIndex, range);
      // get full update range for this index, based on updates made in all the other buffers
      currentUpdateRange = getUpdateRange();
    } else {
      // increase buffer update range at the current index, based on new range
      extendUpdateRange(currentBufferIndex, range);
    }

    // If ring buffer's Vulkan Buffers are CPU mapped
    if (buffers_[0]->isMapped()) { // if the buffer is mapped
      // if current updated range differs from the input range, copy data outside of the input range
      // from previous buffer
      if ((currentUpdateRange.offset != range.offset || currentUpdateRange.size != range.size) &&
          prevDataPtr) {
        uint8_t* currDataPtr = currentVulkanBuffer()->getMappedPtr();
        // this block is not required for non-mapped buffers, because in that case localData_ always
        // contains the latest data; and staging device is used to copy data from localData_ to the
        // device.

        // this block is needed for mapped buffer, because device buffer data will be updated based
        // on CPU accessible portion of the currentVulkanBuffer (which is in currDataPtr). And so
        // data changes outside the input range will be copied from the previous buffer.

        // this should never happen, but check just in case
        IGL_DEBUG_ASSERT(currentUpdateRange.offset <= range.offset);

        // copy data from starting of current update range to range offset
        const auto frontCopySize = range.offset - currentUpdateRange.offset;
        if (frontCopySize > 0) {
          checked_memcpy(currDataPtr + currentUpdateRange.offset,
                         getSizeInBytes() - currentUpdateRange.offset,
                         prevDataPtr + currentUpdateRange.offset,
                         frontCopySize);
        }

        // copy data from range end to current update range end
        const auto rangeEnd = range.offset + range.size;
        const auto currentUpdateRangeEnd = currentUpdateRange.offset + currentUpdateRange.size;

        // this should never happen, but check just in case
        IGL_DEBUG_ASSERT(currentUpdateRangeEnd >= rangeEnd);

        const auto backCopySize = currentUpdateRangeEnd - rangeEnd;
        if (backCopySize > 0) {
          checked_memcpy(currDataPtr + rangeEnd,
                         getSizeInBytes() - rangeEnd,
                         prevDataPtr + rangeEnd,
                         backCopySize);
        }
      }
      currentVulkanBuffer()->bufferSubData(range.offset, range.size, data);
    } else {
      // update local data copy
      checked_memcpy(localData_.get() + range.offset, range.size, data, range.size);
      // use staging to upload data to device-local buffers
      ctx.stagingDevice_->bufferSubData(*currentVulkanBuffer(),
                                        currentUpdateRange.offset,
                                        currentUpdateRange.size,
                                        localData_.get() + currentUpdateRange.offset);
    }
  } else {
    // use staging to upload data to device-local buffers
    ctx.stagingDevice_->bufferSubData(*currentVulkanBuffer(), range.offset, range.size, data);
  }
  return Result();
}

size_t Buffer::getSizeInBytes() const {
  return desc_.length;
}

uint64_t Buffer::gpuAddress(size_t offset) const {
  IGL_DEBUG_ASSERT((offset & 7) == 0,
                   "Buffer offset must be 8 bytes aligned as per GLSL_EXT_buffer_reference spec.");

  if (isRingBuffer_) {
    device_.getVulkanContext().markRingBufferRead();
  }
  return static_cast<uint64_t>(currentVulkanBuffer()->getVkDeviceAddress()) + offset;
}

VkBuffer Buffer::getVkBuffer() const {
  if (isRingBuffer_) {
    device_.getVulkanContext().markRingBufferRead();
  }
  return currentVulkanBuffer()->getVkBuffer();
}

VkBufferUsageFlags Buffer::getBufferUsageFlags() const {
  return currentVulkanBuffer()->getBufferUsageFlags();
}

void* IGL_NULLABLE Buffer::map(const BufferRange& range, Result* outResult) {
  IGL_PROFILER_FUNCTION();

  IGL_DEBUG_ASSERT(!isRingBuffer_, "Buffer::map() operation not supported for ring buffer");

  // Sanity check
  if ((range.size > desc_.length) || (range.offset > desc_.length - range.size)) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Range exceeds buffer length");
    // @fb-only
    // @lint-ignore CLANGTIDY
    return nullptr;
  }

  // If the buffer is currently mapped, then unmap it first
  if (mappedRange_.size &&
      (mappedRange_.size != range.size || mappedRange_.offset != range.offset)) {
    IGL_DEBUG_ABORT("Buffer::map() is called more than once without Buffer::unmap()");
    unmap();
  }

  mappedRange_ = range;

  Result::setOk(outResult);

  const auto& buffer = currentVulkanBuffer();
  if (!buffer->isMapped()) {
    // handle DEVICE_LOCAL buffers
    tmpBuffer_.resize(range.size);
    const VulkanContext& ctx = device_.getVulkanContext();
    ctx.stagingDevice_->getBufferSubData(*buffer, range.offset, range.size, tmpBuffer_.data());
    return tmpBuffer_.data();
  }

  return buffer->getMappedPtr() + range.offset;
}

void Buffer::unmap() {
  IGL_PROFILER_FUNCTION();

  IGL_DEBUG_ASSERT(!isRingBuffer_, "Buffer::unmap() operation not supported for ring buffer");
  IGL_DEBUG_ASSERT(mappedRange_.size, "Called Buffer::unmap() without Buffer::map()");

  const auto& buffer = currentVulkanBuffer();
  const BufferRange range(tmpBuffer_.size(), mappedRange_.offset);
  if (!buffer->isMapped()) {
    // handle DEVICE_LOCAL buffers
    upload(tmpBuffer_.data(), range);
  } else if (!buffer->isCoherentMemory()) {
    buffer->flushMappedMemory(range.offset, range.size);
  }
  mappedRange_.size = 0;
}

BufferDesc::BufferAPIHint Buffer::requestedApiHints() const noexcept {
  return desc_.hint;
}

BufferDesc::BufferAPIHint Buffer::acceptedApiHints() const noexcept {
  if ((desc_.type & BufferDesc::BufferTypeBits::Uniform) != 0) {
    return BufferDesc::BufferAPIHintBits::UniformBlock;
  }

  return 0;
}

ResourceStorage Buffer::storage() const noexcept {
  return desc_.storage;
}

} // namespace igl::vulkan
