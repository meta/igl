/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanBuffer.h"

#include <cstring>
#include <igl/IGLSafeC.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanContext.h>

namespace igl::vulkan {

VulkanBuffer::VulkanBuffer(const VulkanContext& ctx,
                           VkDevice device,
                           VkDeviceSize bufferSize,
                           VkBufferUsageFlags usageFlags,
                           VkMemoryPropertyFlags memFlags,
                           const char* debugName) :
  ctx_(ctx),
  device_(device),
  bufferSize_(bufferSize),
  usageFlags_(usageFlags),
  memFlags_(memFlags) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_DEBUG_ASSERT(bufferSize > 0);

  // Initialize Buffer Info
  const VkBufferCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bufferSize,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  if (IGL_VULKAN_USE_VMA) {
    VmaAllocationCreateInfo ciAlloc = {
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    // Initialize VmaAllocation Info
    if ((memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      ciAlloc.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      ciAlloc.preferredFlags =
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      ciAlloc.flags =
          VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }

    auto* const vma = static_cast<VmaAllocator>(ctx_.getVmaAllocator());
    VkResult result = vmaCreateBuffer(vma, &ci, &ciAlloc, &vkBuffer_, &vmaAllocation_, nullptr);

    if (result != VK_SUCCESS || vmaAllocation_ == nullptr) {
      // Allocation failed - possibly due to memory pressure from deferred tasks not being
      // processed. Drain deferred tasks queue to free memory, then retry once.
      const_cast<VulkanContext&>(ctx_).waitDeferredTasks();
      vkBuffer_ = VK_NULL_HANDLE;
      vmaAllocation_ = VK_NULL_HANDLE;
      result = vmaCreateBuffer(vma, &ci, &ciAlloc, &vkBuffer_, &vmaAllocation_, nullptr);
      if (result != VK_SUCCESS || vmaAllocation_ == nullptr) {
        IGL_LOG_ERROR(
            "VulkanBuffer: allocation failed after draining deferred tasks (VkResult %d)\n",
            static_cast<int>(result));
        failInit(result != VK_SUCCESS ? result : VK_ERROR_UNKNOWN);
        return;
      }
    }

    vmaSetAllocationName(vma, vmaAllocation_, IGL_FORMAT("VMA Allocation: {}", debugName).c_str());

    // handle memory-mapped buffers
    if ((memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      VkMemoryPropertyFlags allocMemFlags = 0;
      vmaGetAllocationMemoryProperties(vma, vmaAllocation_, &allocMemFlags);
      isCoherentMemory_ = (allocMemFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

      vmaMapMemory(vma, vmaAllocation_, &mappedPtr_);
    }
  } else {
    // create buffer
    VK_ASSERT(ctx_.vf_.vkCreateBuffer(device_, &ci, nullptr, &vkBuffer_));

    // back the buffer with some memory
    {
      VkMemoryRequirements requirements = {};
      ctx_.vf_.vkGetBufferMemoryRequirements(device_, vkBuffer_, &requirements);

      VkMemoryPropertyFlags allocMemFlags = 0;
      VK_ASSERT(ivkAllocateMemory(&ctx_.vf_,
                                  ctx_.getVkPhysicalDevice(),
                                  device_,
                                  &requirements,
                                  memFlags,
                                  ctx.features().has_VK_KHR_buffer_device_address,
                                  &vkMemory_,
                                  &allocMemFlags));
      VK_ASSERT(ctx_.vf_.vkBindBufferMemory(device_, vkBuffer_, vkMemory_, 0));

      // Derive coherence from the memory type actually used for the allocation above. Only
      // meaningful for host-visible buffers; DEVICE_LOCAL-only buffers stay non-coherent even when
      // the chosen memory type happens to also be host-coherent (e.g. on UMA GPUs).
      if ((memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        isCoherentMemory_ = (allocMemFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
      }

      VK_ASSERT(ivkSetDebugObjectName(&ctx_.vf_,
                                      device_,
                                      VK_OBJECT_TYPE_DEVICE_MEMORY,
                                      (uint64_t)vkMemory_,
                                      IGL_FORMAT("Buffer memory: {}", debugName).c_str()));
    }

    // handle memory-mapped buffers
    if ((memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      VK_ASSERT(ctx_.vf_.vkMapMemory(device_, vkMemory_, 0, bufferSize_, 0, &mappedPtr_));
    }
  }

  IGL_DEBUG_ASSERT(vkBuffer_ != VK_NULL_HANDLE);

  // set debug name
  VK_ASSERT(ivkSetDebugObjectName(
      &ctx_.vf_, device_, VK_OBJECT_TYPE_BUFFER, (uint64_t)vkBuffer_, debugName));

  // handle shader access
  if ((usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR) != 0) {
    const VkBufferDeviceAddressInfo ai = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR,
        .buffer = vkBuffer_,
    };
    vkDeviceAddress_ = ctx_.vf_.vkGetBufferDeviceAddressKHR(device_, &ai);
    IGL_DEBUG_ASSERT(vkDeviceAddress_);
  }
}

void VulkanBuffer::failInit(VkResult result) {
  // Only a failed allocation reaches here, so nothing was created and there is nothing to destroy.
  vkBuffer_ = VK_NULL_HANDLE;
  vkMemory_ = VK_NULL_HANDLE;
  vmaAllocation_ = VK_NULL_HANDLE;
  mappedPtr_ = nullptr;
  isCoherentMemory_ = false;
  initResult_ = result != VK_SUCCESS ? result : VK_ERROR_UNKNOWN;
}

void VulkanBuffer::discardNeverSubmitted() {
  if (!valid()) {
    return;
  }

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx_);

  if (IGL_VULKAN_USE_VMA) {
    auto* const vma = static_cast<VmaAllocator>(ctx_.getVmaAllocator());
    if (mappedPtr_ != nullptr) {
      vmaUnmapMemory(vma, vmaAllocation_);
    }
    vmaDestroyBuffer(vma, vkBuffer_, vmaAllocation_);
  } else {
    if (mappedPtr_ != nullptr) {
      ctx_.vf_.vkUnmapMemory(device_, vkMemory_);
    }
    ctx_.vf_.vkDestroyBuffer(device_, vkBuffer_, nullptr);
    ctx_.vf_.vkFreeMemory(device_, vkMemory_, nullptr);
  }
  failInit(VK_ERROR_UNKNOWN);
}

VulkanBuffer::~VulkanBuffer() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);

  if (!valid()) {
    return; // owns nothing: the failing constructor already destroyed its partial objects
  }

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx_);

  if (IGL_VULKAN_USE_VMA) {
    if (mappedPtr_) {
      vmaUnmapMemory(static_cast<VmaAllocator>(ctx_.getVmaAllocator()), vmaAllocation_);
    }
    ctx_.deferredTask(std::packaged_task<void()>(
        [vma = ctx_.getVmaAllocator(), buffer = vkBuffer_, allocation = vmaAllocation_]() {
          vmaDestroyBuffer(static_cast<VmaAllocator>(vma), buffer, allocation);
        }));
  } else {
    if (mappedPtr_) {
      ctx_.vf_.vkUnmapMemory(device_, vkMemory_);
    }
    ctx_.deferredTask(std::packaged_task<void()>(
        [vf = &ctx_.vf_, device = device_, buffer = vkBuffer_, memory = vkMemory_]() {
          vf->vkDestroyBuffer(device, buffer, nullptr);
          vf->vkFreeMemory(device, memory, nullptr);
        }));
  }
}

void VulkanBuffer::flushMappedMemory(VkDeviceSize offset, VkDeviceSize size) const {
  IGL_DEBUG_ASSERT(valid());
  if (!IGL_DEBUG_VERIFY(isMapped())) {
    return;
  }

  if (IGL_VULKAN_USE_VMA) {
    vmaFlushAllocation(
        static_cast<VmaAllocator>(ctx_.getVmaAllocator()), vmaAllocation_, offset, size);
  } else {
    const VkMappedMemoryRange memoryRange = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = vkMemory_,
        .offset = offset,
        .size = size,
    };
    ctx_.vf_.vkFlushMappedMemoryRanges(device_, 1, &memoryRange);
  }
}

void VulkanBuffer::invalidateMappedMemory(VkDeviceSize offset, VkDeviceSize size) const {
  IGL_DEBUG_ASSERT(valid());
  if (!IGL_DEBUG_VERIFY(isMapped())) {
    return;
  }

  if (IGL_VULKAN_USE_VMA) {
    vmaInvalidateAllocation(
        static_cast<VmaAllocator>(ctx_.getVmaAllocator()), vmaAllocation_, offset, size);
  } else {
    const VkMappedMemoryRange memoryRange = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = vkMemory_,
        .offset = offset,
        .size = size,
    };
    ctx_.vf_.vkInvalidateMappedMemoryRanges(device_, 1, &memoryRange);
  }
}

void VulkanBuffer::getBufferSubData(size_t offset, size_t size, void* data) const {
  IGL_PROFILER_FUNCTION();

  IGL_DEBUG_ASSERT(valid());

  // Only mapped host-visible buffers can be downloaded this way. All other
  // GPU buffers should use a temporary staging buffer

  IGL_DEBUG_ASSERT(mappedPtr_);

  if (!mappedPtr_) {
    return;
  }

  IGL_DEBUG_ASSERT(offset + size <= bufferSize_);

  if (!isCoherentMemory_) {
    invalidateMappedMemory(offset, size);
  }

  const uint8_t* src = static_cast<uint8_t*>(mappedPtr_) + offset;
  checked_memcpy(data, size, src, size);
}

void VulkanBuffer::bufferSubData(size_t offset, size_t size, const void* data) {
  IGL_PROFILER_FUNCTION();

  IGL_DEBUG_ASSERT(valid());

  // Only mapped host-visible buffers can be uploaded this way. All other GPU buffers should use a
  // temporary staging buffer

  IGL_DEBUG_ASSERT(mappedPtr_);

  if (!mappedPtr_) {
    return;
  }

  IGL_DEBUG_ASSERT(offset + size <= bufferSize_);

  if (data) {
    checked_memcpy(static_cast<uint8_t*>(mappedPtr_) + offset, bufferSize_ - offset, data, size);
  } else {
    std::memset(static_cast<uint8_t*>(mappedPtr_) + offset, 0, size);
  }
  if (!isCoherentMemory_) {
    flushMappedMemory(offset, size);
  }
}

} // namespace igl::vulkan
