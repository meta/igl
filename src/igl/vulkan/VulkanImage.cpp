/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanImage.h"

#include <array>
// NOLINTNEXTLINE(facebook-unused-include-check)
#include <cinttypes>
#include <type_traits>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImageView.h>

#ifndef VK_USE_PLATFORM_WIN32_KHR
#include <unistd.h>
#endif

// any image layout transition causes a full barrier
#define IGL_DEBUG_ENFORCE_FULL_IMAGE_BARRIER 0

namespace {
uint32_t ivkGetMemoryTypeIndex(const VkPhysicalDeviceMemoryProperties& memProps,
                               const uint32_t typeBits,
                               VkMemoryPropertyFlags requiredProperties) {
  // Search memory types to find the index with the requested properties.
  for (uint32_t type = 0; type < memProps.memoryTypeCount; type++) {
    if ((typeBits & (1 << type)) != 0) {
      // Test if this memory type has the required properties.
      const VkFlags propertyFlags = memProps.memoryTypes[type].propertyFlags;
      if ((propertyFlags & requiredProperties) == requiredProperties) {
        return type;
      }
    }
  }
  if ((requiredProperties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
    // there's no DEVICE_LOCAL memory heap here - look again
    requiredProperties &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (uint32_t type = 0; type < memProps.memoryTypeCount; type++) {
      if ((typeBits & (1 << type)) != 0) {
        // Test if this memory type has the required properties.
        const VkFlags propertyFlags = memProps.memoryTypes[type].propertyFlags;
        if ((propertyFlags & requiredProperties) == requiredProperties) {
          return type;
        }
      }
    }
  }
  IGL_LOG_ERROR("Memory type %d with properties %d not found.", typeBits, requiredProperties);
  return 0;
}

// Returns true if one of the memory types allowed by `typeBits` has all of `requiredProperties`.
bool hasMemoryType(const VkPhysicalDeviceMemoryProperties& memProps,
                   const uint32_t typeBits,
                   const VkMemoryPropertyFlags requiredProperties) {
  for (uint32_t type = 0; type < memProps.memoryTypeCount; type++) {
    if ((typeBits & (1u << type)) != 0 &&
        (memProps.memoryTypes[type].propertyFlags & requiredProperties) == requiredProperties) {
      return true;
    }
  }
  return false;
}

void logLazilyAllocatedFallback(VkFormat format, VkSampleCountFlagBits samples) {
  IGL_LOG_INFO_ONCE(
      "Lazily allocated memory is not available for a Memoryless image (format %d, %u samples). "
      "Using device-local memory instead.\n",
      static_cast<int>(format),
      static_cast<uint32_t>(samples));
}

// VkImage export and import is only implemented on Windows, Linux and Android platforms.
#if IGL_PLATFORM_WINDOWS
constexpr auto kHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#elif IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID
constexpr auto kHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

} // namespace

namespace igl::vulkan {

static_assert(std::is_trivially_copyable_v<VulkanImageCreateInfo>);

namespace {

/// @brief Attempts VMA image allocation, retrying once after draining deferred tasks on failure.
/// Returns true if the allocation succeeded.
bool tryAllocateImageVma(const VulkanContext& ctx,
                         const VkImageCreateInfo& ci,
                         const VmaAllocationCreateInfo& ciAlloc,
                         VkImage* outImage,
                         VmaAllocation* outAllocation,
                         VkFormat imageFormat,
                         VkMemoryPropertyFlags memFlags) {
  const VkResult result = vmaCreateImage(static_cast<VmaAllocator>(ctx.getVmaAllocator()),
                                         &ci,
                                         &ciAlloc,
                                         outImage,
                                         outAllocation,
                                         nullptr);

  if (result == VK_SUCCESS && *outAllocation != nullptr) {
    return true;
  }

  IGL_LOG_INFO(
      "vmaCreateImage failed: error result: %d, memflags: %d, imageformat: %d. Retrying after "
      "draining deferred tasks.\n",
      result,
      memFlags,
      imageFormat);
  // Allocation failed - possibly due to memory pressure from deferred tasks not being
  // processed. Drain deferred tasks queue to free memory and retry.
  const_cast<VulkanContext&>(ctx).waitDeferredTasks();

  const VkResult retryResult = vmaCreateImage(static_cast<VmaAllocator>(ctx.getVmaAllocator()),
                                              &ci,
                                              &ciAlloc,
                                              outImage,
                                              outAllocation,
                                              nullptr);

  if (retryResult != VK_SUCCESS || *outAllocation == nullptr) {
    IGL_LOG_INFO(
        "vmaCreateImage retry failed: error result: %d, memflags: %d, imageformat: %d. "
        "Continuing with null allocation.\n",
        retryResult,
        memFlags,
        imageFormat);
    return false;
  }

  IGL_LOG_INFO(
      "vmaCreateImage retry succeeded: memflags: %d, imageformat: %d after draining "
      "deferred tasks.\n",
      memFlags,
      imageFormat);
  return true;
}

VmaAllocationCreateInfo makeImageAllocationInfo(VkMemoryPropertyFlags memFlags) {
  const bool isHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  const bool isLazilyAllocated = (memFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0;
  return {
      .flags = 0,
      .usage = isHostVisible ? VMA_MEMORY_USAGE_CPU_TO_GPU : VMA_MEMORY_USAGE_AUTO,
      // For lazily allocated (memoryless) images, require the flag so VMA picks GMEM-backed
      // memory on TBDR GPUs. For other non-host-visible cases, pass memFlags as a preference
      // to guide VMA without causing hard allocation failures.
      .requiredFlags = isLazilyAllocated ? memFlags : static_cast<VkMemoryPropertyFlags>(0),
      .preferredFlags =
          (!isHostVisible && !isLazilyAllocated) ? memFlags : static_cast<VkMemoryPropertyFlags>(0),
  };
}

} // namespace

VulkanImage::VulkanImage(const VulkanContext& ctx,
                         VkImage image,
                         const char* debugName,
                         VkImageUsageFlags usageFlags,
                         bool isExternallyManaged,
                         VkExtent3D extent,
                         VkImageType type,
                         VkFormat imageFormat,
                         uint32_t mipLevels,
                         uint32_t arrayLayers,
                         VkSampleCountFlagBits samples,
                         bool isImported,
                         bool isSrgbMutableFormat) :
  ctx_(&ctx),
  physicalDevice_(ctx.getVkPhysicalDevice()),
  device_(ctx.getVkDevice()),
  vkImage_(image),
  usageFlags_(usageFlags),
  isExternallyManaged_(isExternallyManaged),
  extent_(extent),
  type_(type),
  imageFormat_(imageFormat),
  mipLevels_(mipLevels),
  arrayLayers_(arrayLayers),
  samples_(samples),
  isDepthFormat_(hasDepth(imageFormat)), // NOLINT(readability-identifier-naming)
  isStencilFormat_(hasStencil(imageFormat)), // NOLINT(readability-identifier-naming)
  isDepthOrStencilFormat_(isDepthFormat_ || isStencilFormat_),
  isImported_(isImported),
  isSrgbMutableFormat_(isSrgbMutableFormat) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  setName(debugName);
  VK_ASSERT(ivkSetDebugObjectName(
      &ctx_->vf_, device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)vkImage_, debugName));
  ctx_->vf_.vkGetPhysicalDeviceFormatProperties(physicalDevice_, imageFormat_, &formatProperties_);
}
VulkanImage::VulkanImage(const VulkanContext& ctx,
                         VkImage image,
                         const VulkanImageCreateInfo& createInfo,
                         const char* debugName) :
  VulkanImage(ctx,
              image,
              debugName,
              createInfo.usageFlags,
              createInfo.isExternallyManaged,
              createInfo.extent,
              createInfo.type,
              createInfo.imageFormat,
              createInfo.mipLevels,
              createInfo.arrayLayers,
              createInfo.samples,
              createInfo.isImported,
              createInfo.isSrgbMutableFormat) {}

VulkanImage::VulkanImage(const VulkanContext& ctx,
                         VkExtent3D extent,
                         VkImageType type,
                         VkFormat format,
                         uint32_t mipLevels,
                         uint32_t arrayLayers,
                         VkImageTiling tiling,
                         VkImageUsageFlags usageFlags,
                         VkMemoryPropertyFlags memFlags,
                         VkImageCreateFlags createFlags,
                         VkSampleCountFlagBits samples,
                         const char* debugName,
                         bool isSrgbMutableFormat) :
  ctx_(&ctx),
  physicalDevice_(ctx.getVkPhysicalDevice()),
  device_(ctx.getVkDevice()),
  usageFlags_(usageFlags),
  extent_(extent),
  type_(type),
  imageFormat_(format),
  mipLevels_(mipLevels),
  arrayLayers_(arrayLayers),
  samples_(samples),
  isDepthFormat_(hasDepth(format)), // NOLINT(readability-identifier-naming)
  isStencilFormat_(hasStencil(format)), // NOLINT(readability-identifier-naming)
  isDepthOrStencilFormat_(isDepthFormat_ || isStencilFormat_),
  isCubemap_((createFlags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0),
  tiling_(tiling) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_DEBUG_ASSERT(mipLevels_ > 0, "The image must contain at least one mip level");
  IGL_DEBUG_ASSERT(arrayLayers_ > 0, "The image must contain at least one layer");
  IGL_DEBUG_ASSERT(imageFormat_ != VK_FORMAT_UNDEFINED, "Invalid VkFormat value");
  IGL_DEBUG_ASSERT(samples_ > 0, "The image must contain at least one sample");

  setName(debugName);

  const bool isDisjoint = (createFlags & VK_IMAGE_CREATE_DISJOINT_BIT) != 0;

  // When the caller marks a mutable UNORM image as sRGB-backed, chain a VkImageFormatListCreateInfo
  // so the driver knows only two view formats will be used. This avoids the performance
  // pessimization some drivers apply for unconstrained mutable images.
  //
  // VkImageFormatListCreateInfo is core only in Vulkan 1.2+ (below that it needs
  // VK_KHR_image_format_list, which IGL never enables), so also gate on the device API version.
  // On an older device the struct would be an unrecognized pNext entry that validation layers flag
  // and that silently drops the view-format constraint, so we neither chain it nor set the flag.
  const bool isMutableFormat = (createFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0;
  const bool supportsFormatList =
      ctx.getVkPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_2;
  const VkFormat srgbCounterpart = isSrgbMutableFormat ? unormToSrgb(format) : format;
  if (isSrgbMutableFormat && isMutableFormat && srgbCounterpart != format && !supportsFormatList) {
    IGL_LOG_ERROR_ONCE(
        "VulkanImage requested sRGB mutable format list, but VkImageFormatListCreateInfo "
        "requires Vulkan 1.2 or VK_KHR_image_format_list.\n");
  }
  const bool needsFormatList = isSrgbMutableFormat && isMutableFormat && supportsFormatList &&
                               srgbCounterpart != format;
  isSrgbMutableFormat_ = needsFormatList;
  const VkFormat viewFormats[2] = {format, srgbCounterpart};
  const VkImageFormatListCreateInfo formatListCI = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
      .viewFormatCount = needsFormatList ? 2u : 0u,
      .pViewFormats = needsFormatList ? viewFormats : nullptr,
  };

  const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = needsFormatList ? &formatListCI : nullptr,
      .flags = createFlags,
      .imageType = type,
      .format = imageFormat_,
      .extent = extent_,
      .mipLevels = mipLevels_,
      .arrayLayers = arrayLayers_,
      .samples = samples,
      .tiling = tiling,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  if (IGL_VULKAN_USE_VMA && !isDisjoint) {
    bool isAllocated = false;
    VkMemoryPropertyFlags allocFlags = memFlags;
    if ((allocFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0) {
      // A single attempt, because draining deferred tasks does not help when no lazily allocated
      // memory type can back this image.
      const VmaAllocationCreateInfo ciAlloc = makeImageAllocationInfo(allocFlags);
      isAllocated = vmaCreateImage(static_cast<VmaAllocator>(ctx_->getVmaAllocator()),
                                   &ci,
                                   &ciAlloc,
                                   &vkImage_,
                                   &vmaAllocation_,
                                   nullptr) == VK_SUCCESS &&
                    vmaAllocation_ != nullptr;
      if (!isAllocated) {
        logLazilyAllocatedFallback(imageFormat_, samples);
        allocFlags &= ~VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
      }
    }
    if (!isAllocated) {
      const VmaAllocationCreateInfo ciAlloc = makeImageAllocationInfo(allocFlags);
      tryAllocateImageVma(*ctx_, ci, ciAlloc, &vkImage_, &vmaAllocation_, imageFormat_, allocFlags);
    }

    if (vmaAllocation_) {
      VkMemoryRequirements memRequirements{};
      ctx_->vf_.vkGetImageMemoryRequirements(device_, vkImage_, &memRequirements);

      // handle memory-mapped buffers
      if ((memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        vmaMapMemory(
            static_cast<VmaAllocator>(ctx_->getVmaAllocator()), vmaAllocation_, &mappedPtr_);
        if ((memRequirements.memoryTypeBits & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
          isCoherentMemory_ = true;
        }
      }

      vmaSetAllocationName(static_cast<VmaAllocator>(ctx_->getVmaAllocator()),
                           vmaAllocation_,
                           IGL_FORMAT("VMA Allocation: {}", debugName).c_str());

      VmaAllocationInfo allocationInfo;
      vmaGetAllocationInfo(
          static_cast<VmaAllocator>(ctx_->getVmaAllocator()), vmaAllocation_, &allocationInfo);
      allocatedSize = allocationInfo.size;
    }
  } else {
    // create a disjoint image - TODO: merge it with the VMA code path above
    VK_ASSERT(ctx_->vf_.vkCreateImage(device_, &ci, nullptr, &vkImage_));

    VkMemoryRequirements2 memRequirements[3] = {
        {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2},
        {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2},
        {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2},
    };

    // back the image with some memory
    {
      const uint32_t numPlanes = igl::vulkan::getNumImagePlanes(format);
      IGL_DEBUG_ASSERT(numPlanes > 0 && numPlanes <= kMaxImagePlanes);
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const VkImagePlaneMemoryRequirementsInfo planes[kMaxImagePlanes] = {
          {.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
           .planeAspect = VK_IMAGE_ASPECT_PLANE_0_BIT},
          {.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
           .planeAspect = VK_IMAGE_ASPECT_PLANE_1_BIT},
          {.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
           .planeAspect = VK_IMAGE_ASPECT_PLANE_2_BIT},
      };
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const VkImageMemoryRequirementsInfo2 imgRequirements[kMaxImagePlanes] = {
          {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
           .pNext = isDisjoint && numPlanes > 0 ? &planes[0] : nullptr,
           .image = vkImage_},
          {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
           .pNext = isDisjoint && numPlanes > 1 ? &planes[1] : nullptr,
           .image = vkImage_},
          {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
           .pNext = isDisjoint && numPlanes > 2 ? &planes[2] : nullptr,
           .image = vkImage_},
      };
      for (uint32_t p = 0; p != numPlanes; p++) {
        ctx_->vf_.vkGetImageMemoryRequirements2(device_, &imgRequirements[p], &memRequirements[p]);
        VkMemoryPropertyFlags planeMemFlags = memFlags;
        if ((planeMemFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0 &&
            !hasMemoryType(ctx_->memoryProperties,
                           memRequirements[p].memoryRequirements.memoryTypeBits,
                           planeMemFlags)) {
          logLazilyAllocatedFallback(imageFormat_, samples);
          planeMemFlags &= ~VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
        }
        VK_ASSERT(ivkAllocateMemory2(&ctx_->vf_,
                                     physicalDevice_,
                                     device_,
                                     &memRequirements[p],
                                     planeMemFlags,
                                     false,
                                     &vkMemory_[p]));

        VK_ASSERT(ivkSetDebugObjectName(&ctx_->vf_,
                                        device_,
                                        VK_OBJECT_TYPE_DEVICE_MEMORY,
                                        (uint64_t)vkMemory_[p],
                                        IGL_FORMAT("Memory [{}]: {}", p, debugName).c_str()));
      }
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const VkBindImagePlaneMemoryInfo bindImagePlaneMemoryInfo[kMaxImagePlanes] = {
          {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
           .planeAspect = VK_IMAGE_ASPECT_PLANE_0_BIT},
          {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
           .planeAspect = VK_IMAGE_ASPECT_PLANE_1_BIT},
          {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
           .planeAspect = VK_IMAGE_ASPECT_PLANE_2_BIT},
      };
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const VkBindImageMemoryInfo bindInfo[kMaxImagePlanes] = {
          {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
           .pNext = isDisjoint ? &bindImagePlaneMemoryInfo[0] : nullptr,
           .image = vkImage_,
           .memory = vkMemory_[0]},
          {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
           .pNext = &bindImagePlaneMemoryInfo[1],
           .image = vkImage_,
           .memory = vkMemory_[1]},
          {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
           .pNext = &bindImagePlaneMemoryInfo[2],
           .image = vkImage_,
           .memory = vkMemory_[2]},
      };
      VK_ASSERT(ctx_->vf_.vkBindImageMemory2(device_, numPlanes, bindInfo));

      allocatedSize = memRequirements[0].memoryRequirements.size +
                      memRequirements[1].memoryRequirements.size +
                      memRequirements[2].memoryRequirements.size;
    }

    // handle memory-mapped images
    if ((memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      // map only the first image plane
      VK_ASSERT(ctx_->vf_.vkMapMemory(device_, vkMemory_[0], 0, VK_WHOLE_SIZE, 0, &mappedPtr_));
      const uint32_t memoryTypeBits = memRequirements[0].memoryRequirements.memoryTypeBits;
      if ((memoryTypeBits & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        isCoherentMemory_ = true;
      }
    }
  }

  VK_ASSERT(ivkSetDebugObjectName(
      &ctx_->vf_, device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)vkImage_, debugName));

  // Get physical device's properties for the image's format
  ctx_->vf_.vkGetPhysicalDeviceFormatProperties(physicalDevice_, imageFormat_, &formatProperties_);
}

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
VulkanImage::VulkanImage(const VulkanContext& ctx,
                         AHardwareBuffer* hwBuffer,
                         uint64_t memoryAllocationSize,
                         VkExtent3D extent,
                         VkImageType type,
                         VkFormat format,
                         uint32_t mipLevels,
                         uint32_t arrayLayers,
                         VkImageTiling tiling,
                         VkImageUsageFlags usageFlags,
                         VkImageCreateFlags createFlags,
                         VkSampleCountFlagBits samples,
                         const char* debugName) :
  ctx_(&ctx),
  physicalDevice_(ctx.getVkPhysicalDevice()),
  device_(ctx.getVkDevice()),
  usageFlags_(usageFlags),
  extent_(extent),
  type_(type),
  imageFormat_(format),
  mipLevels_(mipLevels),
  arrayLayers_(arrayLayers),
  samples_(samples),
  isDepthFormat_(hasDepth(format)),
  isStencilFormat_(hasStencil(format)),
  isDepthOrStencilFormat_(isDepthFormat_ || isStencilFormat_),
  isImported_(true),
  tiling_(tiling) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_DEBUG_ASSERT(hwBuffer);
  IGL_DEBUG_ASSERT(mipLevels_ > 0, "The image must contain at least one mip level");
  IGL_DEBUG_ASSERT(arrayLayers_ > 0, "The image must contain at least one layer");
  IGL_DEBUG_ASSERT(imageFormat_ != VK_FORMAT_UNDEFINED, "Invalid VkFormat value");
  IGL_DEBUG_ASSERT(samples_ > 0, "The image must contain at least one sample");

  setName(debugName);

  const VkExternalMemoryImageCreateInfo extImgMem = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};

  const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &extImgMem,
      .flags = createFlags,
      .imageType = type,
      .format = imageFormat_,
      .extent = extent_,
      .mipLevels = mipLevels_,
      .arrayLayers = arrayLayers_,
      .samples = samples,
      .tiling = tiling,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  VK_ASSERT(ctx_->vf_.vkCreateImage(device_, &ci, nullptr, &vkImage_));
  VK_ASSERT(ivkSetDebugObjectName(
      &ctx_->vf_, device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)vkImage_, debugName));

  // NOTE: Importing a hardware buffer causes Vulkan to acquire a reference to
  // the hardware buffer, which it releases when the allocated memory is freed.

  const VkImportAndroidHardwareBufferInfoANDROID hwBufferInfo = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
      .buffer = hwBuffer,
  };

  const VkMemoryDedicatedAllocateInfo dedicatedAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .pNext = &hwBufferInfo,
      .image = vkImage_,
      .buffer = VK_NULL_HANDLE,
  };

  VkAndroidHardwareBufferPropertiesANDROID hwBufferProperties{
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
  };
  VK_ASSERT(ctx_->vf_.vkGetAndroidHardwareBufferPropertiesANDROID(
      device_, hwBuffer, &hwBufferProperties));

  const VkMemoryAllocateInfo memoryAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &dedicatedAllocateInfo,
      .allocationSize = hwBufferProperties.allocationSize,
      .memoryTypeIndex = ivkGetMemoryTypeIndex(ctx_->memoryProperties,
                                               hwBufferProperties.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

  IGL_LOG_INFO("Imported texture has requirements %d, ends up index %d",
               hwBufferProperties.memoryTypeBits,
               memoryAllocateInfo.memoryTypeIndex);

  VK_ASSERT(ctx_->vf_.vkAllocateMemory(device_, &memoryAllocateInfo, nullptr, &vkMemory_[0]));
  VK_ASSERT(ctx_->vf_.vkBindImageMemory(device_, vkImage_, vkMemory_[0], 0));

  ctx_->vf_.vkGetPhysicalDeviceFormatProperties(physicalDevice_, imageFormat_, &formatProperties_);
}
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

VulkanImage::VulkanImage(const VulkanContext& ctx,
                         int32_t undupedFileDescriptor,
                         uint64_t memoryAllocationSize,
                         VkExtent3D extent,
                         VkImageType type,
                         VkFormat format,
                         uint32_t mipLevels,
                         uint32_t arrayLayers,
                         VkImageTiling tiling,
                         VkImageUsageFlags usageFlags,
                         VkImageCreateFlags createFlags,
                         VkSampleCountFlagBits samples,
                         const char* debugName) :
  ctx_(&ctx),
  physicalDevice_(ctx.getVkPhysicalDevice()),
  device_(ctx.getVkDevice()),
  usageFlags_(usageFlags),
  extent_(extent),
  type_(type),
  imageFormat_(format),
  mipLevels_(mipLevels),
  arrayLayers_(arrayLayers),
  samples_(samples),
  isDepthFormat_(hasDepth(format)), // NOLINT(readability-identifier-naming)
  isStencilFormat_(hasStencil(format)), // NOLINT(readability-identifier-naming)
  isDepthOrStencilFormat_(isDepthFormat_ || isStencilFormat_),
  isImported_(true),
  tiling_(tiling) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_DEBUG_ASSERT(mipLevels_ > 0, "The image must contain at least one mip level");
  IGL_DEBUG_ASSERT(arrayLayers_ > 0, "The image must contain at least one layer");
  IGL_DEBUG_ASSERT(imageFormat_ != VK_FORMAT_UNDEFINED, "Invalid VkFormat value");
  IGL_DEBUG_ASSERT(samples_ > 0, "The image must contain at least one sample");

#ifdef VK_USE_PLATFORM_WIN32_KHR
  IGL_DEBUG_ABORT("You can only import a VulkanImage on non-windows environments");
#endif

  setName(debugName);

  const VkExternalMemoryImageCreateInfo extImgMem = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
  };

  const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &extImgMem,
      .flags = createFlags,
      .imageType = type,
      .format = imageFormat_,
      .extent = extent_,
      .mipLevels = mipLevels_,
      .arrayLayers = arrayLayers_,
      .samples = samples,
      .tiling = tiling,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  // create image.. importing external memory cannot use VMA
  VK_ASSERT(ctx_->vf_.vkCreateImage(device_, &ci, nullptr, &vkImage_));
  VK_ASSERT(ivkSetDebugObjectName(
      &ctx_->vf_, device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)vkImage_, debugName));

  int importedFd = -1;
#ifndef VK_USE_PLATFORM_WIN32_KHR
  importedFd = dup(undupedFileDescriptor);
#endif
  IGL_DEBUG_ASSERT(importedFd >= 0);

  // NOTE: Importing memory from a file descriptor transfers ownership of the fd from the
  // app to the Vk implementation. The app must not perform any operations on the fd after
  // a successful import.
  // Vulkan implementation is responsible for closing the fds
  //
  // Apps can import the same underlying memory into:
  //  - multiple instances of vk
  //  - same instance from which it was exported
  //  - multiple times into a given vk instance.
  // in all cases, each import operation must create a distinct VkDeviceMemory object

  const VkImageMemoryRequirementsInfo2 memoryRequirementInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
      .image = vkImage_,
  };
  VkMemoryRequirements2 memoryRequirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
  };
  ctx_->vf_.vkGetImageMemoryRequirements2(device_, &memoryRequirementInfo, &memoryRequirements);

  // TODO_VULKAN: Verify the following from the spec:
  // the memory from which fd was exported must have been created on the same physical device
  // as device.
  const VkImportMemoryFdInfoKHR fdInfo = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
      .fd = importedFd,
  };

  const VkMemoryAllocateInfo memoryAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &fdInfo,
      .allocationSize = memoryAllocationSize,
      .memoryTypeIndex = ivkGetMemoryTypeIndex(ctx_->memoryProperties,
                                               memoryRequirements.memoryRequirements.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
  };

  IGL_LOG_INFO("Imported texture has requirements %d, ends up index %d",
               memoryRequirements.memoryRequirements.memoryTypeBits,
               memoryAllocateInfo.memoryTypeIndex);

  // @fb-only
  // @fb-only
// @fb-only
  // @fb-only
      // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
// @fb-only
  VK_ASSERT(ctx_->vf_.vkAllocateMemory(device_, &memoryAllocateInfo, nullptr, &vkMemory_[0]));
  VK_ASSERT(ctx_->vf_.vkBindImageMemory(device_, vkImage_, vkMemory_[0], 0));
// @fb-only

  VK_ASSERT(ivkSetDebugObjectName(&ctx_->vf_,
                                  device_,
                                  VK_OBJECT_TYPE_DEVICE_MEMORY,
                                  (uint64_t)vkMemory_[0],
                                  IGL_FORMAT("Memory: {}", debugName).c_str()));

  ctx_->vf_.vkGetPhysicalDeviceFormatProperties(physicalDevice_, imageFormat_, &formatProperties_);
}

#if IGL_PLATFORM_WINDOWS
VulkanImage::VulkanImage(const VulkanContext& ctx,
                         void* windowsHandle,
                         VkExtent3D extent,
                         VkImageType type,
                         VkFormat format,
                         uint32_t mipLevels,
                         uint32_t arrayLayers,
                         VkImageTiling tiling,
                         VkImageUsageFlags usageFlags,
                         VkImageCreateFlags createFlags,
                         VkSampleCountFlagBits samples,
                         const char* debugName) :
  ctx_(&ctx),
  physicalDevice_(ctx.getVkPhysicalDevice()),
  device_(ctx.getVkDevice()),
  usageFlags_(usageFlags),
  extent_(extent),
  type_(type),
  imageFormat_(format),
  mipLevels_(mipLevels),
  arrayLayers_(arrayLayers),
  samples_(samples),
  isDepthFormat_(hasDepth(format)),
  isStencilFormat_(hasStencil(format)),
  isDepthOrStencilFormat_(isDepthFormat_ || isStencilFormat_),
  isImported_(true),
  tiling_(tiling) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_DEBUG_ASSERT(mipLevels_ > 0, "The image must contain at least one mip level");
  IGL_DEBUG_ASSERT(arrayLayers_ > 0, "The image must contain at least one layer");
  IGL_DEBUG_ASSERT(imageFormat_ != VK_FORMAT_UNDEFINED, "Invalid VkFormat value");
  IGL_DEBUG_ASSERT(samples_ > 0, "The image must contain at least one sample");

  const VkExternalMemoryImageCreateInfo extImgMem = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
  };

  const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &extImgMem,
      .flags = createFlags,
      .imageType = type,
      .format = imageFormat_,
      .extent = extent_,
      .mipLevels = mipLevels_,
      .arrayLayers = arrayLayers_,
      .samples = samples,
      .tiling = tiling,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  // create image. importing external memory cannot use VMA
  VK_ASSERT(ctx_->vf_.vkCreateImage(device_, &ci, nullptr, &vkImage_));
  VK_ASSERT(ivkSetDebugObjectName(
      &ctx_->vf_, device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)vkImage_, debugName));

  const VkImageMemoryRequirementsInfo2 memoryRequirementInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
      .image = vkImage_,
  };

  VkMemoryRequirements2 memoryRequirements = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
  ctx_->vf_.vkGetImageMemoryRequirements2(device_, &memoryRequirementInfo, &memoryRequirements);

  const VkImportMemoryWin32HandleInfoKHR handleInfo = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
      .handle = windowsHandle,
  };

  const VkMemoryAllocateInfo memoryAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &handleInfo,
      .allocationSize = memoryRequirements.memoryRequirements.size,
      .memoryTypeIndex = ivkGetMemoryTypeIndex(ctx_->memoryProperties,
                                               memoryRequirements.memoryRequirements.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
  };

  IGL_LOG_INFO("Imported texture has memoryAllocationSize %" PRIu64
               ", requirements 0x%08X, ends up index 0x%08X",
               memoryRequirements.memoryRequirements.size,
               memoryRequirements.memoryRequirements.memoryTypeBits,
               memoryAllocateInfo.memoryTypeIndex);

  VK_ASSERT(ctx_->vf_.vkAllocateMemory(device_, &memoryAllocateInfo, nullptr, &vkMemory_[0]));
  VK_ASSERT(ctx_->vf_.vkBindImageMemory(device_, vkImage_, vkMemory_[0], 0));

  VK_ASSERT(ivkSetDebugObjectName(&ctx_->vf_,
                                  device_,
                                  VK_OBJECT_TYPE_DEVICE_MEMORY,
                                  (uint64_t)vkMemory_[0],
                                  IGL_FORMAT("Memory: {}", debugName).c_str()));

  ctx_->vf_.vkGetPhysicalDeviceFormatProperties(physicalDevice_, imageFormat_, &formatProperties_);
}
#endif // IGL_PLATFORM_WINDOWS

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID
VulkanImage VulkanImage::createWithExportMemory(const VulkanContext& ctx,
                                                VkExtent3D extent,
                                                VkImageType type,
                                                VkFormat format,
                                                uint32_t mipLevels,
                                                uint32_t arrayLayers,
                                                VkImageTiling tiling,
                                                VkImageUsageFlags usageFlags,
                                                VkImageCreateFlags createFlags,
                                                VkSampleCountFlagBits samples,
                                                const char* debugName) {
  const VkPhysicalDeviceExternalImageFormatInfo externaInfo = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
      .handleType = kHandleType,
  };
  const VkPhysicalDeviceImageFormatInfo2 formatInfo2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .pNext = &externaInfo,
      .format = format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = tiling,
      .usage = usageFlags,
      .flags = createFlags,
  };

  VkExternalImageFormatProperties externalImageFormatProperties = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
  };
  VkImageFormatProperties2 imageFormatProperties2 = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &externalImageFormatProperties,
  };
  const auto result = ctx.vf_.vkGetPhysicalDeviceImageFormatProperties2(
      ctx.getVkPhysicalDevice(), &formatInfo2, &imageFormatProperties2);
  if (result != VK_SUCCESS) {
    IGL_LOG_ERROR(
        "External memory is not supported. format: %d image_tiling: %d usage: %d flags: %d",
        format,
        tiling,
        usageFlags,
        createFlags);
    return VulkanImage();
  }
  const auto& externalFormatProperties = externalImageFormatProperties.externalMemoryProperties;
  if ((externalFormatProperties.externalMemoryFeatures &
       VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0) {
    IGL_LOG_ERROR(
        "External memory cannot be exported. format: %d image_tiling: %d usage: %d flags: %d",
        format,
        tiling,
        usageFlags,
        createFlags);
    return VulkanImage();
  }

  IGL_DEBUG_ASSERT((externalFormatProperties.compatibleHandleTypes & kHandleType) != 0);
  const VkExternalMemoryHandleTypeFlags compatibleHandleTypes = kHandleType;

  return {ctx,
          extent,
          type,
          format,
          mipLevels,
          arrayLayers,
          tiling,
          usageFlags,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          createFlags,
          samples,
          compatibleHandleTypes,
          debugName};
}

VulkanImage::VulkanImage(const VulkanContext& ctx,
                         VkExtent3D extent,
                         VkImageType type,
                         VkFormat format,
                         uint32_t mipLevels,
                         uint32_t arrayLayers,
                         VkImageTiling tiling,
                         VkImageUsageFlags usageFlags,
                         VkMemoryPropertyFlags memFlags,
                         VkImageCreateFlags createFlags,
                         VkSampleCountFlagBits samples,
                         VkExternalMemoryHandleTypeFlags compatibleHandleTypes,
                         const char* debugName) :
  ctx_(&ctx),
  physicalDevice_(ctx.getVkPhysicalDevice()),
  device_(ctx.getVkDevice()),
  usageFlags_(usageFlags),
  extent_(extent),
  type_(type),
  imageFormat_(format),
  mipLevels_(mipLevels),
  arrayLayers_(arrayLayers),
  samples_(samples),
  isDepthFormat_(hasDepth(format)), // NOLINT(readability-identifier-naming)
  isStencilFormat_(hasStencil(format)), // NOLINT(readability-identifier-naming)
  isDepthOrStencilFormat_(isDepthFormat_ || isStencilFormat_),
  isExported_(true),
  tiling_(tiling) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_DEBUG_ASSERT(mipLevels_ > 0, "The image must contain at least one mip level");
  IGL_DEBUG_ASSERT(arrayLayers_ > 0, "The image must contain at least one layer");
  IGL_DEBUG_ASSERT(imageFormat_ != VK_FORMAT_UNDEFINED, "Invalid VkFormat value");
  IGL_DEBUG_ASSERT(samples_ > 0, "The image must contain at least one sample");

  const VkExternalMemoryImageCreateInfo externalImageCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = compatibleHandleTypes,
  };

  const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &externalImageCreateInfo,
      .flags = createFlags,
      .imageType = type,
      .format = imageFormat_,
      .extent = extent_,
      .mipLevels = mipLevels_,
      .arrayLayers = arrayLayers_,
      .samples = samples,
      .tiling = tiling,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  // create VkImage importing external memory cannot use VMA
  VK_ASSERT(ctx_->vf_.vkCreateImage(device_, &ci, nullptr, &vkImage_));
  VK_ASSERT(ivkSetDebugObjectName(
      &ctx_->vf_, device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)vkImage_, debugName));

  // For Android we need a dedicated allocation for exporting the image, otherwise
  // the exported handle is not generated properly.
#if IGL_PLATFORM_ANDROID
  const VkMemoryDedicatedAllocateInfo dedicatedAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .pNext = nullptr,
      .image = vkImage_,
      .buffer = VK_NULL_HANDLE,
  };
#endif // IGL_PLATFORM_ANDROID

  const VkExportMemoryAllocateInfo externalMemoryAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
#if IGL_PLATFORM_ANDROID
      .pNext = &dedicatedAllocateInfo,
#endif // IGL_PLATFORM_ANDROID
      .handleTypes = compatibleHandleTypes,
  };

  std::array<VkBindImagePlaneMemoryInfo, kMaxImagePlanes> bindImagePlaneMemoryInfo{};
  std::array<VkBindImageMemoryInfo, kMaxImagePlanes> bindInfo{};
  const uint32_t numPlanes = igl::vulkan::getNumImagePlanes(format);
  IGL_DEBUG_ASSERT(numPlanes > 0 && numPlanes <= kMaxImagePlanes);
  for (uint32_t p = 0; p != numPlanes; p++) {
    const auto imagePlaneMemoryRequirementsInfo = VkImagePlaneMemoryRequirementsInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
        .planeAspect = static_cast<VkImageAspectFlagBits>(VK_IMAGE_ASPECT_PLANE_0_BIT << p),
    };

    const VkImageMemoryRequirementsInfo2 imageMemoryRequirementInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
        .pNext = numPlanes > 1 ? &imagePlaneMemoryRequirementsInfo : nullptr,
        .image = vkImage_,
    };

    VkMemoryRequirements2 memoryRequirements = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    ctx_->vf_.vkGetImageMemoryRequirements2(
        device_, &imageMemoryRequirementInfo, &memoryRequirements);

    const VkMemoryAllocateInfo memoryAllocateInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &externalMemoryAllocateInfo,
        .allocationSize = memoryRequirements.memoryRequirements.size,
        .memoryTypeIndex = ivkGetMemoryTypeIndex(
            ctx_->memoryProperties, memoryRequirements.memoryRequirements.memoryTypeBits, memFlags),
    };

    IGL_LOG_INFO("Creating image to be exported with memoryAllocationSize %" PRIu64
                 ", requirements 0x%08X, ends up index 0x%08X",
                 memoryRequirements.memoryRequirements.size,
                 memoryRequirements.memoryRequirements.memoryTypeBits,
                 memoryAllocateInfo.memoryTypeIndex);

    VK_ASSERT(ctx_->vf_.vkAllocateMemory(device_, &memoryAllocateInfo, nullptr, &vkMemory_[p]));
    // Record allocation size so consumers (e.g. test framework swapchains)
    // can advertise it to memory importers.
    allocatedSize += memoryRequirements.memoryRequirements.size;

    VK_ASSERT(ivkSetDebugObjectName(&ctx_->vf_,
                                    device_,
                                    VK_OBJECT_TYPE_DEVICE_MEMORY,
                                    (uint64_t)vkMemory_[p],
                                    IGL_FORMAT("Memory [{}]: {}", p, debugName).c_str()));

    bindImagePlaneMemoryInfo[p] = VkBindImagePlaneMemoryInfo{
        .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
        .planeAspect = static_cast<VkImageAspectFlagBits>(VK_IMAGE_ASPECT_PLANE_0_BIT << p),
    };
    bindInfo[p] = VkBindImageMemoryInfo{
        .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
        .pNext = numPlanes > 1 ? &bindImagePlaneMemoryInfo[p] : nullptr,
        .image = vkImage_,
        .memory = vkMemory_[p],
    };
  }
  VK_ASSERT(ctx_->vf_.vkBindImageMemory2(device_, numPlanes, bindInfo.data()));

#if IGL_PLATFORM_WINDOWS
  const VkMemoryGetWin32HandleInfoKHR getHandleInfo{
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
      .memory = vkMemory_[0],
      .handleType = kHandleType,
  };
  VK_ASSERT(ctx_->vf_.vkGetMemoryWin32HandleKHR(device_, &getHandleInfo, &exportedMemoryHandle_));
#else

  const VkMemoryGetFdInfoKHR getFdInfo{
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = vkMemory_[0],
      .handleType = kHandleType,
  };
  VK_ASSERT(ctx_->vf_.vkGetMemoryFdKHR(device_, &getFdInfo, &exportedFd_));

#endif
}
#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID

VulkanImage::~VulkanImage() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  destroy();
}

void VulkanImage::destroy() {
  if (!valid()) {
    return;
  }

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  if (!isExternallyManaged_) {
    if (vkMemory_[1] == VK_NULL_HANDLE) {
      if (vmaAllocation_) {
        if (mappedPtr_) {
          vmaUnmapMemory(static_cast<VmaAllocator>(ctx_->getVmaAllocator()), vmaAllocation_);
        }
        ctx_->deferredTask(std::packaged_task<void()>(
            [vma = ctx_->getVmaAllocator(), image = vkImage_, allocation = vmaAllocation_]() {
              vmaDestroyImage(static_cast<VmaAllocator>(vma), image, allocation);
            }));
      } else {
        if (mappedPtr_) {
          ctx_->vf_.vkUnmapMemory(device_, vkMemory_[0]);
        }
        ctx_->deferredTask(std::packaged_task<void()>(
            [vf = &ctx_->vf_, device = device_, image = vkImage_, memory = vkMemory_[0]]() {
              vf->vkDestroyImage(device, image, nullptr);
              if (memory != VK_NULL_HANDLE) {
                vf->vkFreeMemory(device, memory, nullptr);
              }
            }));
      }
    } else {
      // this never uses VMA
      if (mappedPtr_) {
        ctx_->vf_.vkUnmapMemory(device_, vkMemory_[0]);
      }
      ctx_->deferredTask(std::packaged_task<void()>([vf = &ctx_->vf_,
                                                     device = device_,
                                                     image = vkImage_,
                                                     memory0 = vkMemory_[0],
                                                     memory1 = vkMemory_[1],
                                                     memory2 = vkMemory_[2]]() {
        vf->vkDestroyImage(device, image, nullptr);
        vf->vkFreeMemory(device, memory0, nullptr);
        vf->vkFreeMemory(device, memory1, nullptr);
        vf->vkFreeMemory(device, memory2, nullptr);
      }));
    }
  }

  // ycbcrConversion_ is context-owned; VulkanImage only clears its non-owning reference.
  ycbcrConversion_ = VK_NULL_HANDLE;

  ctx_ = nullptr;
  vkImage_ = VK_NULL_HANDLE;
}

VulkanImageView VulkanImage::createImageView(VkImageViewType type,
                                             VkFormat format,
                                             VkImageAspectFlags aspectMask,
                                             uint32_t baseLevel,
                                             uint32_t numLevels,
                                             uint32_t baseLayer,
                                             uint32_t numLayers,
                                             const char* debugName) const {
  const VulkanImageViewCreateInfo ci = {
      .image = vkImage_,
      .viewType = type,
      .format = format,
      .subresourceRange = VkImageSubresourceRange{.aspectMask = aspectMask,
                                                  .baseMipLevel = baseLevel,
                                                  .levelCount = numLevels,
                                                  .baseArrayLayer = baseLayer,
                                                  .layerCount = numLayers},
  };

  return VulkanImageView{*ctx_, ci, debugName};
}

VulkanImageView VulkanImage::createImageView(VulkanImageViewCreateInfo createInfo,
                                             const char* debugName) const {
  createInfo.image = vkImage_;
  return VulkanImageView{*ctx_, createInfo, debugName};
}

void VulkanImage::transitionLayout(VkCommandBuffer cmdBuf,
                                   VkImageLayout newImageLayout,
                                   VkPipelineStageFlags srcStageMask,
                                   VkPipelineStageFlags dstStageMask,
                                   const VkImageSubresourceRange& subresourceRange) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_TRANSITION);

  VkAccessFlags srcAccessMask = 0;
  VkAccessFlags dstAccessMask = 0;

  if (imageLayout_ == VK_IMAGE_LAYOUT_UNDEFINED) {
    // we do not need to wait for any previous operations in this case
    srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  }

  const VkPipelineStageFlags doNotRequireAccessMask =
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT |
      VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkPipelineStageFlags srcRemainingMask = srcStageMask & ~doNotRequireAccessMask;
  VkPipelineStageFlags dstRemainingMask = dstStageMask & ~doNotRequireAccessMask;

  if ((srcStageMask & VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT) != 0) {
    srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    srcRemainingMask &= ~VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  }
  if ((srcStageMask & VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT) != 0) {
    srcAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    srcRemainingMask &= ~VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  }
  if ((srcStageMask & VK_PIPELINE_STAGE_TRANSFER_BIT) != 0) {
    srcAccessMask |= VK_ACCESS_TRANSFER_WRITE_BIT;
    srcRemainingMask &= ~VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  if ((srcStageMask & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0) {
    srcAccessMask |= VK_ACCESS_SHADER_READ_BIT;
    srcAccessMask |= VK_ACCESS_SHADER_WRITE_BIT;
    srcRemainingMask &= ~VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }
  if ((srcStageMask & VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT) != 0) {
    srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    srcRemainingMask &= ~VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  }
  if ((srcStageMask & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) != 0) {
    srcAccessMask |= VK_ACCESS_SHADER_READ_BIT;
    srcRemainingMask &= ~VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }

  (void)srcRemainingMask;
  IGL_DEBUG_ASSERT(
      srcRemainingMask == 0,
      "Automatic access mask deduction is not implemented (yet) for this srcStageMask = %u",
      srcRemainingMask);

  if ((dstStageMask & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_SHADER_READ_BIT;
    dstAccessMask |= VK_ACCESS_SHADER_WRITE_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }
  if ((dstStageMask & VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  }
  if ((dstStageMask & VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  }
  if ((dstStageMask & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_SHADER_READ_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  if ((dstStageMask & VK_PIPELINE_STAGE_VERTEX_SHADER_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_SHADER_READ_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
  }
  if ((dstStageMask & VK_PIPELINE_STAGE_VERTEX_INPUT_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_INDEX_READ_BIT;
    dstAccessMask |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
  }
  if ((dstStageMask & VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
  }
  if ((dstStageMask & VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    dstAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  }
  if ((dstStageMask & VK_PIPELINE_STAGE_TRANSFER_BIT) != 0) {
    dstAccessMask |= VK_ACCESS_TRANSFER_READ_BIT;
    dstAccessMask |= VK_ACCESS_TRANSFER_WRITE_BIT;
    dstRemainingMask &= ~VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  (void)dstRemainingMask;
  IGL_DEBUG_ASSERT(
      dstRemainingMask == 0,
      "Automatic access mask deduction is not implemented (yet) for this dstStageMask = %u",
      dstRemainingMask);

#if IGL_DEBUG_ENFORCE_FULL_IMAGE_BARRIER
  // full image barrier
  srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

  srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
#endif // IGL_DEBUG_ENFORCE_FULL_IMAGE_BARRIER

  ivkImageMemoryBarrier(&ctx_->vf_,
                        cmdBuf,
                        vkImage_,
                        srcAccessMask,
                        dstAccessMask,
                        imageLayout_,
                        newImageLayout,
                        srcStageMask,
                        dstStageMask,
                        subresourceRange);

  ctx_->journalImageLayout(cmdBuf, *this);
  imageLayout_ = newImageLayout;
}

void VulkanImage::clearColorImage(VkCommandBuffer commandBuffer,
                                  const Color& rgba,
                                  const VkImageSubresourceRange* subresourceRange) const {
  IGL_DEBUG_ASSERT((usageFlags_ & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0);
  IGL_DEBUG_ASSERT(samples_ == VK_SAMPLE_COUNT_1_BIT);
  IGL_DEBUG_ASSERT(!isDepthOrStencilFormat_);

  const VkImageLayout oldLayout = imageLayout_;

  VkClearColorValue value{};
  value.float32[0] = rgba.r;
  value.float32[1] = rgba.g;
  value.float32[2] = rgba.b;
  value.float32[3] = rgba.a;

  const VkImageSubresourceRange defaultRange{
      .aspectMask = getImageAspectFlags(),
      .baseMipLevel = 0,
      .levelCount = VK_REMAINING_MIP_LEVELS,
      .baseArrayLayer = 0,
      .layerCount = VK_REMAINING_ARRAY_LAYERS,
  };

  transitionLayout(commandBuffer,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   subresourceRange ? *subresourceRange : defaultRange);

  ctx_->vf_.vkCmdClearColorImage(commandBuffer,
                                 getVkImage(),
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &value,
                                 1,
                                 subresourceRange ? subresourceRange : &defaultRange);

  const VkImageLayout newLayout = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? ((usageFlags_ & VK_IMAGE_USAGE_SAMPLED_BIT) != 0
                                             ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                             : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                                      : oldLayout;

  transitionLayout(commandBuffer,
                   newLayout,
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   subresourceRange ? *subresourceRange : defaultRange);
}

VkImageAspectFlags VulkanImage::getImageAspectFlags() const {
  VkImageAspectFlags flags = 0;

  if (isDepthFormat_) {
    flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  if (isStencilFormat_) {
    flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  if (!isDepthOrStencilFormat_) {
    flags |= VK_IMAGE_ASPECT_COLOR_BIT;
  }

  return flags;
}

Result VulkanImage::generateMipmap(VkCommandBuffer commandBuffer,
                                   const TextureRangeDesc& range) const {
  IGL_PROFILER_FUNCTION();

  if (isSrgbMutableFormat_) {
    // The VkImage is UNORM (see Texture::create()); route mip generation through a native-sRGB
    // scratch so downsampling filters in sRGB space instead of averaging encoded texels as linear.
    return generateMipmapSrgb(commandBuffer, range);
  }

  return generateMipmapBlit(commandBuffer, range);
}

Result VulkanImage::generateMipmapBlit(VkCommandBuffer commandBuffer,
                                       const TextureRangeDesc& range) const {
  IGL_PROFILER_FUNCTION();

  // Check if device supports downscaling for color or depth/stencil buffer based on image format
  {
    const uint32_t formatFeatureMask =
        (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT);

    const bool hardwareDownscalingSupported =
        ((formatProperties_.optimalTilingFeatures & formatFeatureMask) == formatFeatureMask);

    if (!hardwareDownscalingSupported) {
      // Not all drivers can blit-downscale every format. In particular, KosmicKrisp (the Vulkan-to-
      // Metal driver) cannot blit into depth images, so depth formats such as VK_FORMAT_D16_UNORM
      // report BLIT_SRC but not BLIT_DST. Mipmap generation is implemented via vkCmdBlitImage(),
      // so
      // there is nothing we can do here other than skip it; aborting would take down any otherwise
      // healthy application (e.g. one rendering a mipmapped depth shadow map). Report and no-op.
      return Result(Result::Code::Unsupported,
                    IGL_FORMAT("generateMipmap(): image format {} does not support hardware blit "
                               "downscaling (optimalTilingFeatures missing BLIT_SRC/BLIT_DST)",
                               static_cast<uint32_t>(imageFormat_)));
    }
  }

  // Choose linear filter for color formats if supported by the device, else use nearest filter
  // Choose nearest filter by default for depth/stencil formats
  const VkFilter blitFilter =
      [](bool isDepthOrStencilFormat, bool imageFilterLinear) {
        if (isDepthOrStencilFormat) {
          return VK_FILTER_NEAREST;
        }
        if (imageFilterLinear) {
          return VK_FILTER_LINEAR;
        }
        return VK_FILTER_NEAREST;
      }(isDepthOrStencilFormat_,
        (formatProperties_.optimalTilingFeatures &
         VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0u);

  const VkImageAspectFlags imageAspectFlags = getImageAspectFlags();

  ivkCmdBeginDebugUtilsLabel(
      &ctx_->vf_, commandBuffer, "Generate mipmaps", K_COLOR_GENERATE_MIPMAPS.toFloatPtr());

  IGL_SCOPE_EXIT {
    ivkCmdEndDebugUtilsLabel(&ctx_->vf_, commandBuffer);
  };

  const VkImageLayout originalImageLayout = imageLayout_;

  IGL_DEBUG_ASSERT(originalImageLayout != VK_IMAGE_LAYOUT_UNDEFINED);

  IGL_DEBUG_ASSERT(!isCubemap_ || arrayLayers_ % 6u == 0,
                   "Cubemaps must have a multiple of 6 array layers!");
  const uint32_t multiplier = isCubemap_ ? arrayLayers_ / 6u : 1u;

  // 0: Transition the first mip-level - all layers - to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
  transitionLayout(commandBuffer,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VkImageSubresourceRange{.aspectMask = imageAspectFlags,
                                           .baseMipLevel = range.mipLevel,
                                           .levelCount = range.numMipLevels,
                                           .baseArrayLayer = 0,
                                           .layerCount = VK_REMAINING_ARRAY_LAYERS});

  for (uint32_t arrayLayer = range.layer; arrayLayer < (range.layer + range.numLayers);
       ++arrayLayer) {
    for (uint32_t face = range.face; face < (range.face + range.numFaces); ++face) {
      const uint32_t layer = arrayLayer * multiplier + face;
      int32_t mipWidth = extent_.width > 1 ? static_cast<int32_t>(extent_.width) >> (range.mipLevel)
                                           : 1;
      int32_t mipHeight =
          extent_.height > 1 ? static_cast<int32_t>(extent_.height) >> (range.mipLevel) : 1;

      for (uint32_t i = (range.mipLevel + 1); i < (range.mipLevel + range.numMipLevels); ++i) {
        // 1: Transition the i-th level to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        //    It will be copied into from the (i-1)-th layer
        ivkImageMemoryBarrier(&ctx_->vf_,
                              commandBuffer,
                              vkImage_,
                              0, /* srcAccessMask */
                              VK_ACCESS_TRANSFER_WRITE_BIT, /* dstAccessMask */
                              VK_IMAGE_LAYOUT_UNDEFINED, /* oldImageLayout */
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, /* newImageLayout */
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, /* srcStageMask */
                              VK_PIPELINE_STAGE_TRANSFER_BIT, /* dstStageMask */
                              VkImageSubresourceRange{.aspectMask = imageAspectFlags,
                                                      .baseMipLevel = i,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = layer,
                                                      .layerCount = 1});

        const int32_t nextLevelWidth = mipWidth > 1 ? mipWidth / 2 : 1;
        const int32_t nextLevelHeight = mipHeight > 1 ? mipHeight / 2 : 1;

        const std::array<VkOffset3D, 2> srcOffsets = {
            VkOffset3D{.x = 0, .y = 0, .z = 0},
            VkOffset3D{.x = mipWidth, .y = mipHeight, .z = 1},
        };
        const std::array<VkOffset3D, 2> dstOffsets = {
            VkOffset3D{.x = 0, .y = 0, .z = 0},
            VkOffset3D{.x = nextLevelWidth, .y = nextLevelHeight, .z = 1},
        };

        // 2: Blit the image from the prev mip-level (i-1) (VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        // to the current mip level (i) (VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
#if IGL_VULKAN_PRINT_COMMANDS
        IGL_LOG_INFO("%p vkCmdBlitImage()\n", commandBuffer);
#endif // IGL_VULKAN_PRINT_COMMANDS
        ivkCmdBlitImage(&ctx_->vf_,
                        commandBuffer,
                        vkImage_,
                        vkImage_,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        srcOffsets.data(),
                        dstOffsets.data(),
                        VkImageSubresourceLayers{.aspectMask = imageAspectFlags,
                                                 .mipLevel = i - 1,
                                                 .baseArrayLayer = layer,
                                                 .layerCount = 1},
                        VkImageSubresourceLayers{.aspectMask = imageAspectFlags,
                                                 .mipLevel = i,
                                                 .baseArrayLayer = layer,
                                                 .layerCount = 1},
                        blitFilter);

        // 3: Transition i-th level to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL as it will be read
        // from in the next iteration
        ivkImageMemoryBarrier(&ctx_->vf_,
                              commandBuffer,
                              vkImage_,
                              VK_ACCESS_TRANSFER_WRITE_BIT, /* srcAccessMask */
                              VK_ACCESS_TRANSFER_READ_BIT, /* dstAccessMask */
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, /* oldImageLayout */
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, /* newImageLayout */
                              VK_PIPELINE_STAGE_TRANSFER_BIT, /* srcStageMask */
                              VK_PIPELINE_STAGE_TRANSFER_BIT /* dstStageMask */,
                              VkImageSubresourceRange{.aspectMask = imageAspectFlags,
                                                      .baseMipLevel = i,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = layer,
                                                      .layerCount = 1});

        // Compute the size of the next mip-level
        mipWidth = nextLevelWidth;
        mipHeight = nextLevelHeight;
      }
    }
  }

  // 4: Transition all levels and layers/faces to their final layout
  ivkImageMemoryBarrier(&ctx_->vf_,
                        commandBuffer,
                        vkImage_,
                        VK_ACCESS_TRANSFER_WRITE_BIT, // srcAccessMask
                        0, // dstAccessMask
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, // oldImageLayout
                        originalImageLayout, // newImageLayout
                        VK_PIPELINE_STAGE_TRANSFER_BIT, // srcStageMask
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, // dstStageMask
                        VkImageSubresourceRange{.aspectMask = imageAspectFlags,
                                                .baseMipLevel = range.mipLevel,
                                                .levelCount = range.numMipLevels,
                                                .baseArrayLayer = 0,
                                                .layerCount = VK_REMAINING_ARRAY_LAYERS});

  imageLayout_ = originalImageLayout;
  return Result();
}

Result VulkanImage::generateMipmapSrgb(VkCommandBuffer commandBuffer,
                                       const TextureRangeDesc& range) const {
  IGL_PROFILER_FUNCTION();

  // vkCmdBlitImage() filters in the VkImage's creation-format numeric space and cannot be pointed
  // at a VkImageView, so on this UNORM base image it would average sRGB-encoded texels as if they
  // were linear (wrong gamma). Round-trip through a transient *native sRGB* scratch: copy the
  // source mip across (raw bytes; UNORM and sRGB share a format-compatibility class, so
  // vkCmdCopyImage() is exact), let the normal blit path downsample there (gamma-correct because
  // the scratch is sRGB), then copy the generated mips back.
  // The scratch lives only as long as this call, while a recording of a reusable command buffer
  // runs later and again.
  if (ctx_->isRecordingSlot(commandBuffer)) {
    return Result(Result::Code::Unsupported,
                  "generateMipmap(): the sRGB scratch path cannot record into a reusable command "
                  "buffer");
  }
  IGL_DEBUG_ASSERT(!isDepthOrStencilFormat_, "sRGB scratch mipmap path is color-only");
  const uint32_t expectedNumLayers = isCubemap_ ? arrayLayers_ / 6u : arrayLayers_;
  const uint32_t expectedNumFaces = isCubemap_ ? 6u : 1u;
  const bool isFullLayerFaceRange = range.layer == 0 && range.face == 0 &&
                                    range.numLayers == expectedNumLayers &&
                                    range.numFaces == expectedNumFaces;
  if (!IGL_DEBUG_VERIFY(isFullLayerFaceRange,
                        "sRGB scratch mipmap path only supports the full layer/face range")) {
    // Only full layer/face range is supported here; the copy-back in step 5 uses
    // arrayLayers_ unconditionally while generateMipmapBlit() iterates only over
    // range.numLayers/numFaces, so a partial range would copy uninitialized
    // scratch mips back into this image.
    return Result(Result::Code::Unsupported,
                  "generateMipmap(): the sRGB scratch path only supports the full layer/face "
                  "range");
  }

  const uint32_t baseMip = range.mipLevel;
  const uint32_t numMips = range.numMipLevels;
  if (numMips <= 1) {
    return Result();
  }

  const VkFormat srgbFormat = unormToSrgb(imageFormat_);
  if (!IGL_DEBUG_VERIFY(srgbFormat != imageFormat_)) {
    // isSrgbMutableFormat_ implies an sRGB counterpart; nothing sensible to do otherwise
    return Result(
        Result::Code::InvalidOperation,
        IGL_FORMAT("generateMipmap(): no sRGB counterpart for the backing UNORM format {}",
                   static_cast<uint32_t>(imageFormat_)));
  }

  const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  const VkImageLayout originalImageLayout = imageLayout_;
  IGL_DEBUG_ASSERT(originalImageLayout != VK_IMAGE_LAYOUT_UNDEFINED);

  // A plain local: ~VulkanImage() already routes vmaDestroyImage() through a deferredTask() bound
  // to the submission being recorded here, so the scratch outlives the GPU work. Keeping it alive
  // with a second deferred task instead would defer its destruction twice, and the inner task
  // would be enqueued while ~VulkanContext() is clearing the queue -- too late to ever run.
  VulkanImage scratch(
      *ctx_,
      extent_,
      type_,
      srgbFormat,
      mipLevels_,
      arrayLayers_,
      tiling_,
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      isCubemap_ ? static_cast<VkImageCreateFlags>(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) : 0u,
      samples_,
      "Image: sRGB mipmap scratch");
  if (!IGL_DEBUG_VERIFY(scratch.valid())) {
    return Result(Result::Code::RuntimeError,
                  "generateMipmap(): failed to create the sRGB scratch image");
  }

  ivkCmdBeginDebugUtilsLabel(&ctx_->vf_,
                             commandBuffer,
                             "Generate mipmaps (sRGB scratch)",
                             K_COLOR_GENERATE_MIPMAPS.toFloatPtr());
  IGL_SCOPE_EXIT {
    ivkCmdEndDebugUtilsLabel(&ctx_->vf_, commandBuffer);
  };

  const auto mipExtent = [this](uint32_t level) -> VkExtent3D {
    const uint32_t w = extent_.width >> level;
    const uint32_t h = extent_.height >> level;
    const uint32_t d = extent_.depth >> level;
    return VkExtent3D{
        .width = w ? w : 1u,
        .height = h ? h : 1u,
        .depth = (type_ == VK_IMAGE_TYPE_3D) ? (d ? d : 1u) : 1u,
    };
  };

  // 1) This image's source mip -> TRANSFER_SRC; whole scratch -> TRANSFER_DST.
  transitionLayout(commandBuffer,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VkImageSubresourceRange{.aspectMask = aspect,
                                           .baseMipLevel = baseMip,
                                           .levelCount = 1,
                                           .baseArrayLayer = 0,
                                           .layerCount = VK_REMAINING_ARRAY_LAYERS});
  scratch.transitionLayout(commandBuffer,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VkImageSubresourceRange{.aspectMask = aspect,
                                                   .baseMipLevel = 0,
                                                   .levelCount = VK_REMAINING_MIP_LEVELS,
                                                   .baseArrayLayer = 0,
                                                   .layerCount = VK_REMAINING_ARRAY_LAYERS});

  // 2) Copy the source mip (all layers/faces) into the scratch as raw bytes.
  const VkImageCopy copyIn = {
      .srcSubresource = {.aspectMask = aspect,
                         .mipLevel = baseMip,
                         .baseArrayLayer = 0,
                         .layerCount = arrayLayers_},
      .srcOffset = {.x = 0, .y = 0, .z = 0},
      .dstSubresource = {.aspectMask = aspect,
                         .mipLevel = baseMip,
                         .baseArrayLayer = 0,
                         .layerCount = arrayLayers_},
      .dstOffset = {.x = 0, .y = 0, .z = 0},
      .extent = mipExtent(baseMip),
  };
  ctx_->vf_.vkCmdCopyImage(commandBuffer,
                           vkImage_,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           scratch.vkImage_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copyIn);

  // 3) Downsample on the scratch via the plain blit path. The scratch is native sRGB, so
  //    vkCmdBlitImage() filters in sRGB space. Its tracked imageLayout_ is TRANSFER_DST (step 1),
  //    matching the layout the source mip holds after the copy, so generateMipmapBlit()'s initial
  //    transition preserves it.
  if (const Result result = scratch.generateMipmapBlit(commandBuffer, range); !result.isOk()) {
    if (originalImageLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
      ivkImageMemoryBarrier(&ctx_->vf_,
                            commandBuffer,
                            vkImage_,
                            VK_ACCESS_TRANSFER_READ_BIT,
                            0,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            originalImageLayout,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            VkImageSubresourceRange{.aspectMask = aspect,
                                                    .baseMipLevel = baseMip,
                                                    .levelCount = 1,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = VK_REMAINING_ARRAY_LAYERS});
    }
    imageLayout_ = originalImageLayout;
    return result;
  }

  const VkImageSubresourceRange generatedRange{.aspectMask = aspect,
                                               .baseMipLevel = baseMip + 1,
                                               .levelCount = numMips - 1,
                                               .baseArrayLayer = 0,
                                               .layerCount = VK_REMAINING_ARRAY_LAYERS};

  // 4) Generated scratch mips -> TRANSFER_SRC; matching mips on this image -> TRANSFER_DST.
  // The scratch image is uniformly in TRANSFER_DST after generateMipmapBlit(), so its
  // transition from DST -> SRC with generatedRange is valid.
  scratch.transitionLayout(commandBuffer,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           generatedRange);
  // This image, however, is in mixed real layouts after step 1: baseMip is TRANSFER_SRC,
  // while generatedRange is still in originalImageLayout. Using transitionLayout() here
  // would emit oldLayout = imageLayout_ = TRANSFER_SRC for a subresource whose real
  // layout is originalImageLayout, violating VUID-VkImageMemoryBarrier-oldLayout-01197.
  // Issue the barrier with the true per-level oldLayout.
  ivkImageMemoryBarrier(&ctx_->vf_,
                        commandBuffer,
                        vkImage_,
                        0,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        originalImageLayout,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        generatedRange);

  // 5) Copy the generated mips back into this image, raw bytes, one region per level.
  for (uint32_t i = baseMip + 1; i < baseMip + numMips; ++i) {
    const VkImageCopy copyBack = {
        .srcSubresource = {.aspectMask = aspect,
                           .mipLevel = i,
                           .baseArrayLayer = 0,
                           .layerCount = arrayLayers_},
        .srcOffset = {.x = 0, .y = 0, .z = 0},
        .dstSubresource = {.aspectMask = aspect,
                           .mipLevel = i,
                           .baseArrayLayer = 0,
                           .layerCount = arrayLayers_},
        .dstOffset = {.x = 0, .y = 0, .z = 0},
        .extent = mipExtent(i),
    };
    ctx_->vf_.vkCmdCopyImage(commandBuffer,
                             scratch.vkImage_,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             vkImage_,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             1,
                             &copyBack);
  }

  // 6) Restore this image's touched levels to their original layout. After step 4 the
  // image is in mixed layouts: baseMip = TRANSFER_SRC, generatedRange = TRANSFER_DST.
  // A single barrier with oldLayout = TRANSFER_DST (or TRANSFER_SRC) would mismatch the
  // other subrange and could let drivers discard base-mip metadata (contents must be
  // preserved). Restore with two barriers using correct per-subrange oldLayouts, then
  // sync the single-value tracker.
  ivkImageMemoryBarrier(&ctx_->vf_,
                        commandBuffer,
                        vkImage_,
                        VK_ACCESS_TRANSFER_READ_BIT,
                        0,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        originalImageLayout,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VkImageSubresourceRange{.aspectMask = aspect,
                                                .baseMipLevel = baseMip,
                                                .levelCount = 1,
                                                .baseArrayLayer = 0,
                                                .layerCount = VK_REMAINING_ARRAY_LAYERS});
  ivkImageMemoryBarrier(&ctx_->vf_,
                        commandBuffer,
                        vkImage_,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        0,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        originalImageLayout,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        generatedRange);

  imageLayout_ = originalImageLayout;

  return Result();
}

void VulkanImage::setName(std::string name) noexcept { // NOLINT(bugprone-exception-escape)
#if IGL_DEBUG
  name_ = std::move(name);
#else
  (void)name;
#endif
}

bool VulkanImage::valid() const {
  return ctx_ != nullptr;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
  destroy();
  ctx_ = other.ctx_;
  physicalDevice_ = other.physicalDevice_;
  device_ = other.device_;
  vkImage_ = other.vkImage_;
  usageFlags_ = other.usageFlags_;
  vmaAllocation_ = other.vmaAllocation_;
  formatProperties_ = other.formatProperties_;
  mappedPtr_ = other.mappedPtr_;
  isExternallyManaged_ = other.isExternallyManaged_;
  extent_ = other.extent_;
  type_ = other.type_;
  imageFormat_ = other.imageFormat_;
  isSrgbMutableFormat_ = other.isSrgbMutableFormat_;
  mipLevels_ = other.mipLevels_;
  arrayLayers_ = other.arrayLayers_;
  samples_ = other.samples_;
  isDepthFormat_ = other.isDepthFormat_;
  isStencilFormat_ = other.isStencilFormat_;
  isDepthOrStencilFormat_ = other.isDepthOrStencilFormat_;
  allocatedSize = other.allocatedSize;
  imageLayout_ = other.imageLayout_;
  isImported_ = other.isImported_;
  isCubemap_ = other.isCubemap_;
  isExported_ = other.isExported_;
  exportedMemoryHandle_ = other.exportedMemoryHandle_;
  exportedFd_ = other.exportedFd_;
#if IGL_DEBUG
  name_ = std::move(other.name_);
#endif
  tiling_ = other.tiling_;
  isCoherentMemory_ = other.isCoherentMemory_;
  extendedFormat_ = other.extendedFormat_;
  samplerYcbcrConversionCreateInfo_ = other.samplerYcbcrConversionCreateInfo_;
  ycbcrConversion_ = other.ycbcrConversion_;
  for (size_t i = 0; i != IGL_ARRAY_NUM_ELEMENTS(vkMemory_); i++) {
    vkMemory_[i] = other.vkMemory_[i];
  }

  other.ctx_ = nullptr;
  other.vkImage_ = VK_NULL_HANDLE;
  other.ycbcrConversion_ = VK_NULL_HANDLE;

  return *this;
}

void VulkanImage::flushMappedMemory() const {
  if (!isMappedPtrAccessible() || isCoherentMemory()) {
    return;
  }

  if (vmaAllocation_) {
    vmaFlushAllocation(
        static_cast<VmaAllocator>(ctx_->getVmaAllocator()), vmaAllocation_, 0, VK_WHOLE_SIZE);
  } else {
    const VkMappedMemoryRange memoryRange = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = vkMemory_[0],
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    ctx_->vf_.vkFlushMappedMemoryRanges(device_, 1, &memoryRange);
  }
}

} // namespace igl::vulkan
