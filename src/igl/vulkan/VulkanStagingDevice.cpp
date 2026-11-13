/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/VulkanStagingDevice.h>

#include <algorithm>
#include <numeric>
#include <igl/IGLSafeC.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanImmediateCommands.h>

#define IGL_VULKAN_DEBUG_STAGING_DEVICE 0

using VulkanSubmitHandle = igl::vulkan::VulkanImmediateCommands::SubmitHandle;

constexpr VkDeviceSize kMinStagingBufferSize = static_cast<const VkDeviceSize>(1024u) * 1024u;

namespace igl::vulkan {

VulkanStagingDevice::VulkanStagingDevice(VulkanContext& ctx) : ctx_(ctx) {
  IGL_PROFILER_FUNCTION();

  const auto& limits = ctx_.getVkPhysicalDeviceProperties().limits;

  // Use value of 256MB (limited by some architectures), and clamp it to the max limits
  maxStagingBufferSize_ = std::min(limits.maxStorageBufferRange, 256u * 1024u * 1024u);

  immediate =
      std::make_unique<VulkanImmediateCommands>(ctx_.vf_,
                                                ctx_.getVkDevice(),
                                                ctx_.deviceQueues_.graphicsQueueFamilyIndex,
                                                ctx_.config_.exportableFences,
                                                ctx_.features_.has_VK_KHR_timeline_semaphore,
                                                ctx_.features_.has_VK_KHR_synchronization2,
                                                "VulkanStagingDevice::immediate_",
                                                ctx_.config_.chainSubmitsWithSemaphores);
  IGL_DEBUG_ASSERT(immediate.get());
}

bool VulkanStagingDevice::bufferSubData(VulkanBuffer& buffer,
                                        size_t dstOffset,
                                        size_t size,
                                        const void* data) {
  IGL_PROFILER_FUNCTION();
  if (buffer.isMapped()) {
    buffer.bufferSubData(dstOffset, size, data);
    return true;
  }

  // Use vkCmdUpdateBuffer() for small buffers (up to 64KB) when alignment requirements are met.
  // Per Vulkan spec: dstOffset and dataSize must be multiples of 4, and dataSize <= 65536.
  // This avoids staging buffer allocation and an extra memcpy for small uploads.
  constexpr size_t kMaxUpdateBufferSize = 65536;
  if (data && size <= kMaxUpdateBufferSize && (dstOffset % 4 == 0) && (size % 4 == 0)) {
    IGL_DEBUG_ASSERT((buffer.getBufferUsageFlags() & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0);
    const auto& wrapper = immediate->acquire();
    ctx_.vf_.vkCmdUpdateBuffer(wrapper.cmdBuf,
                               buffer.getVkBuffer(),
                               static_cast<VkDeviceSize>(dstOffset),
                               static_cast<VkDeviceSize>(size),
                               data);
    immediate->submit(wrapper);
    return true;
  }

  VkDeviceSize chunkDstOffset = dstOffset;
  void* copyData = const_cast<void*>(data);

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
  IGL_LOG_INFO("Upload requested for data with %u bytes\n", size);
#endif

  while (size) {
    // finds a free memory block to store the data in the staging buffer
    MemoryRegion memoryChunk = nextFreeBlock(size, false);
    if (memoryChunk.size == 0) {
      IGL_LOG_ERROR("VulkanStagingDevice: no staging block is available for %zu bytes\n", size);
      return false;
    }
    const VkDeviceSize copySize = std::min(static_cast<VkDeviceSize>(size), memoryChunk.size);

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
    IGL_LOG_INFO("\tUploading %u bytes\n", copySize);
#endif

    auto& stagingBuffer = stagingBuffers_[memoryChunk.stagingBufferIndex];

    // copy data into the staging buffer
    stagingBuffer->bufferSubData(memoryChunk.offset, copySize, copyData);

    // do the transfer
    const VkBufferCopy copy = {
        .srcOffset = memoryChunk.offset,
        .dstOffset = chunkDstOffset,
        .size = copySize,
    };

    const auto& wrapper = immediate->acquire();
    ctx_.vf_.vkCmdCopyBuffer(
        wrapper.cmdBuf, stagingBuffer->getVkBuffer(), buffer.getVkBuffer(), 1, &copy);
    memoryChunk.handle = immediate->submit(wrapper); // store the submit handle with the allocation
    regions_.push_back(memoryChunk);

    size -= copySize;
    copyData = static_cast<uint8_t*>(copyData) + copySize;
    chunkDstOffset += copySize;
  }
  return true;
}

void VulkanStagingDevice::mergeRegionsAndFreeBuffers() {
  uint32_t regionIndex = 0;
  while (regionIndex < regions_.size() && immediate->isReady(regions_[regionIndex].handle)) {
    auto& currRegion = regions_[regionIndex];

    // set empty handle for a region, if it has finished processing
    // so handle.empty() check can be done later
    if (!currRegion.handle.empty() && immediate->isReady(currRegion.handle)) {
      currRegion.handle = VulkanImmediateCommands::SubmitHandle();
      freeStagingBufferSize_ += currRegion.size;
    }

    // if a next region exist and is not busy
    if ((regionIndex + 1) < regions_.size() && regions_[regionIndex + 1].handle.empty()) {
      auto& nextRegion = regions_[regionIndex + 1];
      // if current and next region are parts of the same buffer
      if (currRegion.stagingBufferIndex == nextRegion.stagingBufferIndex) {
        // if current and next region are adjacent memory blocks, merge them into one
        const bool adjacentRegions = (currRegion.size + currRegion.offset) == nextRegion.offset ||
                                     (nextRegion.size + nextRegion.offset) == currRegion.offset;

        if (adjacentRegions) {
          const MemoryRegion newRegion = {
              .offset = std::min(currRegion.offset, nextRegion.offset),
              .size = currRegion.size + nextRegion.size,
              .alignedSize = currRegion.alignedSize,
              .handle = VulkanImmediateCommands::SubmitHandle(),
              .stagingBufferIndex = currRegion.stagingBufferIndex,
          };
          nextRegion = newRegion;
          regions_.erase(regions_.begin() + regionIndex);
          continue;
        }
      } else {
        // if current and next region are not parts of the same buffer
        // move region with smaller staging buffer index first, so regions would eventually line up
        // if they have been split
        if (currRegion.stagingBufferIndex > nextRegion.stagingBufferIndex) {
          std::swap(currRegion, nextRegion);
        }
      }
    }

    // if a staging buffer is completely recovered
    if (currRegion.size == currRegion.alignedSize) {
      freeStagingBufferSize_ -= currRegion.size;
      // free the staging buffer
      stagingBuffers_[currRegion.stagingBufferIndex].reset();
      // remove the region
      regions_.erase(regions_.begin() + regionIndex);

      // remove trailing empty staging buffers
      while (!stagingBuffers_.empty() && !stagingBuffers_.back()) {
        stagingBuffers_.pop_back();
      }
      continue;
    }

    regionIndex++;
  }

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
  IGL_LOG_INFO("Regions: %d Staging buffers: %d Free space: %d\n",
               regions_.size(),
               stagingBuffers_.size(),
               freeStagingBufferSize_);
#endif
}

VulkanStagingDevice::MemoryRegion VulkanStagingDevice::nextFreeBlock(VkDeviceSize size,
                                                                     bool contiguous) {
  IGL_PROFILER_FUNCTION();

  const VkDeviceSize requestedAlignedSize = getAlignedSize(size);

  if (shouldAllocateStagingBuffer(requestedAlignedSize, contiguous)) {
    // A failed allocation publishes nothing; the existing regions may still serve the request.
    if (!allocateStagingBuffer(nextSize(requestedAlignedSize))) {
      IGL_LOG_ERROR("VulkanStagingDevice: a staging buffer of %llu bytes could not be allocated\n",
                    static_cast<unsigned long long>(nextSize(requestedAlignedSize)));
    }
  }

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
  IGL_LOG_INFO("nextFreeBlock() with %u bytes, aligned %u bytes\n", size, requestedAlignedSize);
#endif

  // Splits `allocatedSize` bytes off the front of `*regionItr`, updates the bookkeeping, and
  // returns the allocated block. Any remainder is written back into `regions_` so it can be reused,
  // and a fully-consumed region is erased. This is the single point at which a region is handed
  // out, so a returned block is never also left in `regions_` as a free region.
  const auto splitAndReturn = [this](const std::deque<MemoryRegion>::iterator& regionItr,
                                     VkDeviceSize allocatedSize) -> MemoryRegion {
    if (!regionItr->handle.empty()) {
      // A block returned with its submission, which has retired since (the scan takes only ready
      // blocks): it counts as free first, as `mergeRegionsAndFreeBuffers()` counts it. The split
      // below rewrites or removes the region, handle included.
      freeStagingBufferSize_ += regionItr->size;
    }
    const VkDeviceSize newSize = regionItr->size - allocatedSize;
    const VkDeviceSize newOffset = regionItr->offset + allocatedSize;
    const uint32_t stagingBufferIndex = regionItr->stagingBufferIndex;

    const MemoryRegion allocatedRegion = {
        .offset = regionItr->offset,
        .size = allocatedSize,
        .alignedSize = regionItr->alignedSize,
        .handle = VulkanImmediateCommands::SubmitHandle(),
        .stagingBufferIndex = stagingBufferIndex,
    };

    if (newSize > 0) {
      *regionItr = {
          .offset = newOffset,
          .size = newSize,
          .alignedSize = regionItr->alignedSize,
          .handle = VulkanImmediateCommands::SubmitHandle(),
          .stagingBufferIndex = stagingBufferIndex,
      };
    } else {
      regions_.erase(regionItr);
    }

    freeStagingBufferSize_ -= allocatedSize;
    return allocatedRegion;
  };

  // at this point, there should be a free region that can fit the requested size
  auto regionItr = regions_.begin();
  while (regionItr != regions_.end()) {
    // if requested size is available or if contiguous memory is not requested
    if ((regionItr->size >= requestedAlignedSize || !contiguous) &&
        (immediate->isReady(regionItr->handle))) {
      return splitAndReturn(regionItr, std::min(regionItr->size, requestedAlignedSize));
    }
    regionItr++;
  }

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
  IGL_LOG_INFO(
      "Could not find an available block. Waiting for the staging device to become fully "
      "available\n");
#endif

  // Nothing was available. Let's wait for the entire staging buffer to become free
  waitAndReset();

  // try to allocate a new staging buffer
  if (!allocateStagingBuffer(nextSize(requestedAlignedSize))) {
    IGL_LOG_ERROR(
        "VulkanStagingDevice: no staging block is available (the allocation failed and nothing "
        "else became usable)\n");
    return {};
  }

  // After waitAndReset()/allocateStagingBuffer() the deque holds a single whole, ready region.
  // Split it through the same path so the returned block is not also left behind in `regions_` as
  // free. std::min() also covers a contiguous request larger than the maximum staging buffer size
  // (only reachable via alignment padding), handing out the whole buffer instead of nothing.
  regionItr = regions_.begin();
  return splitAndReturn(regionItr, std::min(regionItr->size, requestedAlignedSize));
}

bool VulkanStagingDevice::getBufferSubData(const VulkanBuffer& buffer,
                                           size_t srcOffset,
                                           size_t size,
                                           void* data) {
  IGL_PROFILER_FUNCTION();
  if (buffer.isMapped()) {
    buffer.getBufferSubData(srcOffset, size, data);
    return true;
  }

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
  IGL_LOG_INFO("Download requested for data with %u bytes\n", size);
#endif

  size_t chunkSrcOffset = srcOffset;
  auto* dstData = static_cast<uint8_t*>(data);

  while (size) {
    const MemoryRegion memoryChunk = nextFreeBlock(size, false);
    if (memoryChunk.size == 0) {
      IGL_LOG_ERROR("VulkanStagingDevice: no staging block is available for %zu bytes\n", size);
      return false;
    }
    const VkDeviceSize copySize = std::min(static_cast<VkDeviceSize>(size), memoryChunk.size);

    // do the transfer
    const VkBufferCopy copy = {
        .srcOffset = chunkSrcOffset,
        .dstOffset = memoryChunk.offset,
        .size = copySize,
    };

    const auto& wrapper = immediate->acquire();

    auto& stagingBuffer = stagingBuffers_[memoryChunk.stagingBufferIndex];

    ctx_.vf_.vkCmdCopyBuffer(
        wrapper.cmdBuf, buffer.getVkBuffer(), stagingBuffer->getVkBuffer(), 1, &copy);

    // Wait for command to finish
    immediate->wait(immediate->submit(wrapper), ctx_.config_.fenceTimeoutNanoseconds);

    if (!stagingBuffer->isCoherentMemory()) {
      stagingBuffer->invalidateMappedMemory(memoryChunk.offset, memoryChunk.size);
    }

    // Copy data into data. `size` is the number of bytes still to be written into `data`, which is
    // exactly the remaining capacity of `dstData` (it starts at the full requested size and is
    // decremented by `copySize` each iteration). `copySize` is always <= `size`, so this bound is
    // correct even when `srcOffset` is non-zero.
    const uint8_t* src = stagingBuffer->getMappedPtr() + memoryChunk.offset;
    checked_memcpy(dstData, size, src, copySize);

    size -= copySize;
    dstData += copySize;
    chunkSrcOffset += copySize;

    // the transfer has completed above, so the staging block is free again
    regions_.push_back(memoryChunk);
    freeStagingBufferSize_ += memoryChunk.size;
  }
  return true;
}

bool VulkanStagingDevice::imageData(const VulkanImage& image,
                                    TextureType type,
                                    const TextureRangeDesc& range,
                                    const TextureFormatProperties& properties,
                                    uint32_t bytesPerRow,
                                    VkImageAspectFlags aspectFlags,
                                    const void* data) {
  IGL_PROFILER_FUNCTION();

  const bool is420 = (image.imageFormat_ == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) ||
                     (image.imageFormat_ == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM);

  const uint32_t storageSize =
      is420 ? image.extent_.width * image.extent_.height * 3u / 2u
            : static_cast<uint32_t>(properties.getBytesPerRange(range, bytesPerRow));

  IGL_DEBUG_ASSERT(storageSize);

  // We don't support uploading image data in small chunks. If the total upload size exceeds the
  // the maximum allowed staging buffer size, we can't upload it
  IGL_DEBUG_ASSERT(storageSize <= maxStagingBufferSize_,
                   "Image size exceeds maximum size of staging buffer");

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
  IGL_LOG_INFO("Image upload requested for data with %u bytes\n", storageSize);
#endif

  // Keep alignment padding in the allocated region for reclamation after submission.
  const VkDeviceSize alignment = getImageCopyAlignment(properties);
  MemoryRegion memoryChunk = nextFreeBlock(
      std::min(static_cast<VkDeviceSize>(storageSize) + alignment - 1, maxStagingBufferSize_),
      true);
  if (memoryChunk.size == 0) {
    IGL_LOG_ERROR("VulkanStagingDevice: no staging block is available for %u bytes\n", storageSize);
    return false;
  }
  const VkDeviceSize stagingOffset = (memoryChunk.offset + alignment - 1) / alignment * alignment;

  IGL_DEBUG_ASSERT(memoryChunk.size >= stagingOffset - memoryChunk.offset + storageSize);
  auto& stagingBuffer = stagingBuffers_[memoryChunk.stagingBufferIndex];

  // 1. Copy the pixel data into the host visible staging buffer
  stagingBuffer->bufferSubData(stagingOffset, storageSize, data);

  const auto& wrapper = immediate->acquire();
  const uint32_t initialLayer = getVkLayer(type, range.face, range.layer);
  const uint32_t numLayers = getVkLayer(type, range.numFaces, range.numLayers);

  std::vector<VkBufferImageCopy> copyRegions;
  copyRegions.reserve(range.numMipLevels);

  if (is420) {
    // this is a prototype support implemented for a couple of multiplanar image formats
    IGL_DEBUG_ASSERT(range.face == 0 && range.layer == 0 && range.mipLevel == 0);
    IGL_DEBUG_ASSERT(range.numFaces == 1 && range.numLayers == 1 && range.numMipLevels == 1);
    IGL_DEBUG_ASSERT(range.x == 0 && range.y == 0 && range.z == 0);
    IGL_DEBUG_ASSERT(image.type_ == VK_IMAGE_TYPE_2D);
    IGL_DEBUG_ASSERT(image.extent_.width == range.width && image.extent_.height == range.height);
    const uint32_t w = image.extent_.width;
    const uint32_t h = image.extent_.height;
    ivkCmdBeginDebugUtilsLabel(&ctx_.vf_,
                               wrapper.cmdBuf,
                               "VulkanStagingDevice::imageData (upload YUV image data)",
                               K_COLOR_UPLOAD_IMAGE.toFloatPtr());
    VkImageAspectFlags imageAspect = VK_IMAGE_ASPECT_PLANE_0_BIT;

    // Luminance (1 plane)
    copyRegions.emplace_back(VkBufferImageCopy{
        .bufferOffset = stagingOffset,
        .imageSubresource =
            VkImageSubresourceLayers{
                .aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {.x = 0, .y = 0, .z = 0},
        .imageExtent = {.width = w, .height = h, .depth = 1u},
    });
    // Chrominance (in 1 or 2 planes, 420 subsampled)
    const VkDeviceSize planeSize0 = static_cast<VkDeviceSize>(w) * static_cast<VkDeviceSize>(h);
    const VkDeviceSize planeSize1 = planeSize0 / 4; // subsampled
    if (image.imageFormat_ == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
      imageAspect |= VK_IMAGE_ASPECT_PLANE_1_BIT;
      copyRegions.emplace_back(VkBufferImageCopy{
          .bufferOffset = stagingOffset + planeSize0,
          .imageSubresource =
              VkImageSubresourceLayers{
                  .aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT,
                  .mipLevel = 0,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
          .imageOffset = {.x = 0, .y = 0, .z = 0},
          .imageExtent = {.width = w / 2, .height = h / 2, .depth = 1u},
      });
    } else if (image.imageFormat_ == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM) {
      imageAspect |= VK_IMAGE_ASPECT_PLANE_1_BIT | VK_IMAGE_ASPECT_PLANE_2_BIT;
      copyRegions.emplace_back(VkBufferImageCopy{
          .bufferOffset = stagingOffset + planeSize0,
          .imageSubresource =
              VkImageSubresourceLayers{
                  .aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT,
                  .mipLevel = 0,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
          .imageOffset = {.x = 0, .y = 0, .z = 0},
          .imageExtent = {.width = w / 2, .height = h / 2, .depth = 1u},
      });
      copyRegions.emplace_back(VkBufferImageCopy{
          .bufferOffset = stagingOffset + planeSize0 + planeSize1,
          .imageSubresource =
              VkImageSubresourceLayers{
                  .aspectMask = VK_IMAGE_ASPECT_PLANE_2_BIT,
                  .mipLevel = 0,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
          .imageOffset = {.x = 0, .y = 0, .z = 0},
          .imageExtent = {.width = w / 2, .height = h / 2, .depth = 1u},
      });

    } else {
      IGL_DEBUG_ABORT("Unimplemented multiplanar image format");
      return false;
    }

    const VkImageSubresourceRange subresourceRange = {
        .aspectMask = imageAspect,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
    // 1. Transition initial image layout into TRANSFER_DST_OPTIMAL
    ivkImageMemoryBarrier(&ctx_.vf_,
                          wrapper.cmdBuf,
                          image.getVkImage(),
                          0,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          subresourceRange);

    // 2. Copy the pixel data from the staging buffer into the image
#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkCmdCopyBufferToImage()\n", wrapper.cmdBuf);
#endif // IGL_VULKAN_PRINT_COMMANDS
    ctx_.vf_.vkCmdCopyBufferToImage(wrapper.cmdBuf,
                                    stagingBuffer->getVkBuffer(),
                                    image.getVkImage(),
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    static_cast<uint32_t>(copyRegions.size()),
                                    copyRegions.data());

    const VkImageLayout targetLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // 3. Transition TRANSFER_DST_OPTIMAL into `targetLayout`
    ivkImageMemoryBarrier(&ctx_.vf_,
                          wrapper.cmdBuf,
                          image.getVkImage(),
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          targetLayout,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                              VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          subresourceRange);

    ctx_.journalImageLayout(wrapper.cmdBuf, image);
    image.imageLayout_ = targetLayout;

    ivkCmdEndDebugUtilsLabel(&ctx_.vf_, wrapper.cmdBuf);

    // Store the allocated block with the SubmitHandle at the end of the deque
    memoryChunk.handle = immediate->submit(wrapper);
    regions_.push_back(memoryChunk);

    return true;

    // end of VK_FORMAT_G8_B8R8_2PLANE_420_UNORM code path
  }

  // vkCmdCopyBufferToImage() can have only one single bit set for image aspect flags (IGL has no
  // way to distinguish between Depth and Stencil for combined depth/stencil image formats)
  const VkImageAspectFlags aspectMask =
      image.isDepthFormat_ ? VK_IMAGE_ASPECT_DEPTH_BIT
                           : (image.isStencilFormat_ ? VK_IMAGE_ASPECT_STENCIL_BIT : aspectFlags);

  ivkCmdBeginDebugUtilsLabel(&ctx_.vf_,
                             wrapper.cmdBuf,
                             "VulkanStagingDevice::imageData (upload image data)",
                             K_COLOR_UPLOAD_IMAGE.toFloatPtr());

  for (uint32_t mipLevel = range.mipLevel; mipLevel < range.mipLevel + range.numMipLevels;
       ++mipLevel) {
    const auto mipRange = range.atMipLevel(mipLevel);
    const uint32_t offset =
        static_cast<uint32_t>(properties.getSubRangeByteOffset(range, mipRange, bytesPerRow));
    const uint32_t texelsPerRow = bytesPerRow / static_cast<uint32_t>(properties.bytesPerBlock);

    if (image.type_ == VK_IMAGE_TYPE_2D) {
      copyRegions.emplace_back(VkBufferImageCopy{
          .bufferOffset = stagingOffset + offset,
          .bufferRowLength = texelsPerRow,
          .imageSubresource =
              VkImageSubresourceLayers{
                  .aspectMask = aspectMask,
                  .mipLevel = static_cast<uint32_t>(mipLevel),
                  .baseArrayLayer = initialLayer,
                  .layerCount = numLayers,
              },
          .imageOffset = {.x = static_cast<int32_t>(mipRange.x),
                          .y = static_cast<int32_t>(mipRange.y),
                          .z = 0},
          .imageExtent = {.width = static_cast<uint32_t>(mipRange.width),
                          .height = static_cast<uint32_t>(mipRange.height),
                          .depth = 1u},
      });
    } else {
      copyRegions.emplace_back(VkBufferImageCopy{
          .bufferOffset = stagingOffset + offset,
          .bufferRowLength = texelsPerRow,
          .bufferImageHeight = 0,
          .imageSubresource =
              VkImageSubresourceLayers{
                  .aspectMask = aspectMask,
                  .mipLevel = static_cast<uint32_t>(mipLevel),
                  .baseArrayLayer = initialLayer,
                  .layerCount = numLayers,
              },
          .imageOffset = VkOffset3D{.x = static_cast<int32_t>(mipRange.x),
                                    .y = static_cast<int32_t>(mipRange.y),
                                    .z = static_cast<int32_t>(mipRange.z)},
          .imageExtent = VkExtent3D{.width = static_cast<uint32_t>(mipRange.width),
                                    .height = static_cast<uint32_t>(mipRange.height),
                                    .depth = static_cast<uint32_t>(mipRange.depth)}});
    }
  }

  // image memory barriers should have combined image aspect flags (depth/stencil)
  const VkImageSubresourceRange subresourceRange = {
      .aspectMask = aspectFlags,
      .baseMipLevel = 0,
      .levelCount = VK_REMAINING_MIP_LEVELS,
      .baseArrayLayer = initialLayer,
      .layerCount = numLayers,
  };
  // 1. Transition initial image layout into TRANSFER_DST_OPTIMAL
  ivkImageMemoryBarrier(&ctx_.vf_,
                        wrapper.cmdBuf,
                        image.getVkImage(),
                        0,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        subresourceRange);

  // 2. Copy the pixel data from the staging buffer into the image
#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("%p vkCmdCopyBufferToImage()\n", wrapper.cmdBuf);
#endif // IGL_VULKAN_PRINT_COMMANDS
  ctx_.vf_.vkCmdCopyBufferToImage(wrapper.cmdBuf,
                                  stagingBuffer->getVkBuffer(),
                                  image.getVkImage(),
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  static_cast<uint32_t>(copyRegions.size()),
                                  copyRegions.data());

  const bool isSampled = (image.getVkImageUsageFlags() & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;
  const bool isStorage = (image.getVkImageUsageFlags() & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
  const bool isColorAttachment =
      (image.getVkImageUsageFlags() & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0;
  const bool isDepthStencilAttachment =
      (image.getVkImageUsageFlags() & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;

  // a ternary cascade...
  const VkImageLayout targetLayout =
      isSampled ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : (isStorage ? VK_IMAGE_LAYOUT_GENERAL
                             : (isColorAttachment
                                    ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                    : (isDepthStencilAttachment
                                           ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                           : VK_IMAGE_LAYOUT_UNDEFINED)));

  IGL_DEBUG_ASSERT(targetLayout != VK_IMAGE_LAYOUT_UNDEFINED, "Missing usage flags");

  const VkAccessFlags dstAccessMask =
      isSampled
          ? VK_ACCESS_SHADER_READ_BIT
          : (isStorage ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                       : (isColorAttachment ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                            : (isDepthStencilAttachment
                                                   ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                                                   : 0)));

  // Priority ordering mirrors the targetLayout/dstAccessMask cascades above: when multiple usage
  // bits are set (e.g. SAMPLED | STORAGE), isSampled wins and the layout becomes
  // SHADER_READ_ONLY_OPTIMAL. The image will require a subsequent transition before use as an
  // attachment or storage, so covering only the shader stages is correct for the post-upload case.
  const VkPipelineStageFlags dstStageMask =
      isSampled
          ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
          : (isStorage
                 ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                 : (isColorAttachment
                        ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        : (isDepthStencilAttachment ? VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                          VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                                                    : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)));

  // 3. Transition TRANSFER_DST_OPTIMAL into `targetLayout`
  ivkImageMemoryBarrier(&ctx_.vf_,
                        wrapper.cmdBuf,
                        image.getVkImage(),
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        targetLayout,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        dstStageMask,
                        subresourceRange);

  ctx_.journalImageLayout(wrapper.cmdBuf, image);
  image.imageLayout_ = targetLayout;

  ivkCmdEndDebugUtilsLabel(&ctx_.vf_, wrapper.cmdBuf);

  // Store the allocated block with the SubmitHandle at the end of the deque
  memoryChunk.handle = immediate->submit(wrapper);
  regions_.push_back(memoryChunk);
  return true;
}

bool VulkanStagingDevice::getImageData2D(VkImage srcImage,
                                         const uint32_t level,
                                         const uint32_t layer,
                                         const VkRect2D& imageRegion,
                                         TextureFormatProperties properties,
                                         VkFormat /*format*/,
                                         VkImageLayout layout,
                                         VkImageAspectFlags aspectFlags,
                                         void* data,
                                         uint32_t bytesPerRow,
                                         bool flipImageVertical) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(layout != VK_IMAGE_LAYOUT_UNDEFINED);

  const bool mustRepack = bytesPerRow != 0 && bytesPerRow % properties.bytesPerBlock != 0;

  const auto range =
      TextureRangeDesc::new2D(0, 0, imageRegion.extent.width, imageRegion.extent.height);
  const uint32_t storageSize = static_cast<uint32_t>(
      properties.getBytesPerRange(range.atMipLevel(0), mustRepack ? 0 : bytesPerRow));

  // We don't support uploading image data in small chunks. If the total upload size exceeds the
  // the maximum allowed staging buffer size, we can't upload it
  IGL_DEBUG_ASSERT(storageSize <= maxStagingBufferSize_,
                   "Image size exceeds maximum size of staging buffer");

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
  IGL_LOG_INFO("Image download requested for data with %u bytes\n", storageSize);
#endif

  // Keep alignment padding in the allocated region for reclamation after the copy.
  const VkDeviceSize alignment = getImageCopyAlignment(properties);
  const MemoryRegion memoryChunk = nextFreeBlock(
      std::min(static_cast<VkDeviceSize>(storageSize) + alignment - 1, maxStagingBufferSize_),
      true);
  if (memoryChunk.size == 0) {
    IGL_LOG_ERROR("VulkanStagingDevice: no staging block is available for %u bytes\n", storageSize);
    return false;
  }
  const VkDeviceSize stagingOffset = (memoryChunk.offset + alignment - 1) / alignment * alignment;

  IGL_DEBUG_ASSERT(memoryChunk.size >= stagingOffset - memoryChunk.offset + storageSize);
  const auto& wrapper1 = immediate->acquire();

  // 1. Transition to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
  ivkImageMemoryBarrier(&ctx_.vf_,
                        wrapper1.cmdBuf,
                        srcImage,
                        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, // srcAccessMask
                        VK_ACCESS_TRANSFER_READ_BIT, // dstAccessMask
                        layout,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, // wait for any previous operation
                        VK_PIPELINE_STAGE_TRANSFER_BIT, // dstStageMask
                        VkImageSubresourceRange{
                            .aspectMask = aspectFlags,
                            .baseMipLevel = level,
                            .levelCount = 1,
                            .baseArrayLayer = layer,
                            .layerCount = 1,
                        });

  auto& stagingBuffer = stagingBuffers_[memoryChunk.stagingBufferIndex];

  // 2.  Copy the pixel data from the image into the staging buffer
  const VkBufferImageCopy copy = {
      .bufferOffset = stagingOffset,
      .bufferRowLength = mustRepack ? 0
                                    : bytesPerRow / static_cast<uint32_t>(properties.bytesPerBlock),
      .imageSubresource =
          VkImageSubresourceLayers{
              .aspectMask = aspectFlags,
              .mipLevel = level,
              .baseArrayLayer = layer,
              .layerCount = 1,
          },
      .imageOffset = {.x = imageRegion.offset.x, .y = imageRegion.offset.y, .z = 0},
      .imageExtent = {.width = imageRegion.extent.width,
                      .height = imageRegion.extent.height,
                      .depth = 1u},
  };
  ctx_.vf_.vkCmdCopyImageToBuffer(wrapper1.cmdBuf,
                                  srcImage,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  stagingBuffer->getVkBuffer(),
                                  1,
                                  &copy);

  // Wait for command to finish
  immediate->wait(immediate->submit(wrapper1), ctx_.config_.fenceTimeoutNanoseconds);

  // 4 (below). The transition back to the initial image layout.
  const auto transitionBack = [&]() {
    const auto& wrapper2 = immediate->acquire();
    ivkImageMemoryBarrier(&ctx_.vf_,
                          wrapper2.cmdBuf,
                          srcImage,
                          VK_ACCESS_TRANSFER_READ_BIT, // srcAccessMask
                          0, // dstAccessMask
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          layout,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, // srcStageMask
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // dstStageMask
                          VkImageSubresourceRange{
                              .aspectMask = aspectFlags,
                              .baseMipLevel = level,
                              .levelCount = 1,
                              .baseArrayLayer = layer,
                              .layerCount = 1,
                          });
    // the data should be available as we get out of this function
    immediate->wait(immediate->submit(wrapper2), ctx_.config_.fenceTimeoutNanoseconds);
  };

  // 3. Copy data from staging buffer into data
  if (!IGL_DEBUG_VERIFY(stagingBuffer->getMappedPtr())) {
    // The copy finished, so the block is free; the image still goes back to its layout.
    regions_.push_back(memoryChunk);
    freeStagingBufferSize_ += memoryChunk.size;
    transitionBack();
    return false;
  }

  if (!stagingBuffer->isCoherentMemory()) {
    stagingBuffer->invalidateMappedMemory(memoryChunk.offset, memoryChunk.size);
  }

  const uint8_t* src = stagingBuffer->getMappedPtr() + stagingOffset;
  uint8_t* dst = static_cast<uint8_t*>(data);

  // Vulkan only handles cases where row lengths are multiples of texel block size.
  // Must repack the data if the output data does not conform to this.
  if (mustRepack) {
    // Must repack the data.
    ITexture::repackData(properties, range, src, 0, dst, bytesPerRow, flipImageVertical);
  } else {
    if (flipImageVertical) {
      ITexture::repackData(properties, range, src, bytesPerRow, dst, bytesPerRow, true);
    } else {
      checked_memcpy(dst, storageSize, src, storageSize);
    }
  }

  transitionBack();

  regions_.push_back(memoryChunk);
  freeStagingBufferSize_ += memoryChunk.size;
  return true;
}

VkDeviceSize VulkanStagingDevice::getAlignedSize(VkDeviceSize size) const {
  constexpr VkDeviceSize kStagingBufferAlignment = 16; // updated to support BC7 compressed image
  return (size + kStagingBufferAlignment - 1) & ~(kStagingBufferAlignment - 1);
}

VkDeviceSize VulkanStagingDevice::getImageCopyAlignment(
    const TextureFormatProperties& properties) const {
  IGL_PROFILER_FUNCTION();
  // Multiplanar formats encode plane storage separately and have bytesPerBlock == 0.
  const VkDeviceSize texelBlockAlignment =
      properties.isDepthOrStencil() || properties.numPlanes > 1 ? 4 : properties.bytesPerBlock;
  return std::lcm(std::lcm(getAlignedSize(1), texelBlockAlignment),
                  ctx_.getVkPhysicalDeviceProperties().limits.optimalBufferCopyOffsetAlignment);
}

void VulkanStagingDevice::waitAndReset() {
  IGL_PROFILER_FUNCTION();

  for (const auto region : regions_) {
    immediate->wait(region.handle, ctx_.config_.fenceTimeoutNanoseconds);
  }

  regions_.clear();
  stagingBuffers_.clear();
  freeStagingBufferSize_ = 0;
}

bool VulkanStagingDevice::shouldAllocateStagingBuffer(VkDeviceSize sizeNeeded,
                                                      bool contiguous) const noexcept {
  if (regions_.empty()) {
    return true;
  }

  // if contiguous memory is not requested
  if (!contiguous) {
    // return true if there is no free space
    return freeStagingBufferSize_ == 0;
  }

  // if contiguous memory is requested and we have enough free space
  if (sizeNeeded <= freeStagingBufferSize_) {
    // loop over empty regions to find a block that is large enough
    auto regionItr = regions_.begin();
    while (regionItr != regions_.end()) {
      if (regionItr->size >= sizeNeeded && regionItr->handle.empty()) {
        return false;
      }
      regionItr++;
    }
  }

  // return true if no single block can hold the requested size
  return true;
}

VkDeviceSize VulkanStagingDevice::nextSize(VkDeviceSize requestedSize) const {
  return std::min(std::max(getAlignedSize(requestedSize), kMinStagingBufferSize),
                  maxStagingBufferSize_);
}

bool VulkanStagingDevice::allocateStagingBuffer(VkDeviceSize minimumSize) {
  IGL_PROFILER_FUNCTION();

  IGL_DEBUG_ASSERT(minimumSize <= maxStagingBufferSize_);

#if IGL_VULKAN_DEBUG_STAGING_DEVICE
  IGL_LOG_INFO("Allocating a new staging buffer of size %u bytes\n", minimumSize);
#endif

  const VkDeviceSize stagingBufferSize = minimumSize;

  // Increment the id used for naming the staging buffer
  ++stagingBufferCounter_;

  // Create a new staging buffer with the new size; it is published only if it is valid
  auto stagingBuffer = std::make_unique<VulkanBuffer>(
      ctx_,
      ctx_.getVkDevice(),
      stagingBufferSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
      IGL_FORMAT("Buffer: staging buffer #{} with {}B", stagingBufferCounter_, stagingBufferSize)
          .c_str());
  if (!stagingBuffer->valid()) {
    IGL_LOG_ERROR(
        "VulkanStagingDevice: allocating a staging buffer of %llu bytes failed (VkResult %d)\n",
        static_cast<unsigned long long>(stagingBufferSize),
        static_cast<int>(stagingBuffer->initResult()));
    return false;
  }
  stagingBuffers_.emplace_back(std::move(stagingBuffer));

  // Add region that represents the entire buffer
  regions_.push_front({
      .offset = 0,
      .size = stagingBufferSize,
      .alignedSize = stagingBufferSize,
      .handle = VulkanImmediateCommands::SubmitHandle(),
      .stagingBufferIndex = static_cast<uint32_t>(stagingBuffers_.size()) - 1,
  });

  freeStagingBufferSize_ += stagingBufferSize;
  return true;
}

} // namespace igl::vulkan
