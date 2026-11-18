/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only
// @fb-only

#include "NativeHWBuffer.h"

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

#include <android/hardware_buffer.h>
#include <vulkan/vulkan_android.h>
#include <igl/Macros.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanEnumToString.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanImmediateCommands.h>
#include <igl/vulkan/VulkanTexture.h>
#include <igl/vulkan/YcbcrFormat.h>
#include <igl/vulkan/util/TextureFormat.h>

#if IGL_LOGGING_ENABLED
#include <mutex>
#endif

namespace igl::vulkan::android {

namespace {
uint32_t ivkGetMemoryTypeIndex(const VkPhysicalDeviceMemoryProperties& memProps,
                               const uint32_t typeBits,
                               const VkMemoryPropertyFlags requiredProperties) {
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
  return 0;
}

#if IGL_LOGGING_ENABLED
// Emits a one-time debug snapshot of the first imported AHB's Vulkan format and driver
// properties. The std::call_once guard is owned here rather than on the texture-creation path,
// so callers add a single line and the diagnostics never re-run per import.
void logFirstAhbImportDiagnostics(const VkAndroidHardwareBufferFormatPropertiesANDROID& formatProps,
                                  const VkPhysicalDeviceDriverPropertiesKHR& driverProps) {
  static std::once_flag once;
  std::call_once(once, [&formatProps, &driverProps]() {
    const VkComponentMapping& components = formatProps.samplerYcbcrConversionComponents;
    IGL_LOG_DEBUG(
        "[vulkan][ahb][first_frame] Vulkan AHB import\n"
        "  format=%s (%d) externalFormat=0x%llx formatFeatures=0x%x chromaFilter=%s\n"
        "  components r=%s g=%s b=%s a=%s\n"
        "  ycbcrModel=%s ycbcrRange=%s\n"
        "  driverID=%s (%d) conformance=%u.%u.%u.%u\n",
        vkFormatToString(formatProps.format),
        static_cast<int>(formatProps.format),
        static_cast<unsigned long long>(formatProps.externalFormat),
        static_cast<unsigned>(formatProps.formatFeatures),
        formatFeaturesAllowLinearChromaFilter(formatProps.formatFeatures) ? "LINEAR" : "NEAREST",
        vkComponentSwizzleToString(components.r),
        vkComponentSwizzleToString(components.g),
        vkComponentSwizzleToString(components.b),
        vkComponentSwizzleToString(components.a),
        vkSamplerYcbcrModelConversionToString(formatProps.suggestedYcbcrModel),
        vkSamplerYcbcrRangeToString(formatProps.suggestedYcbcrRange),
        vkDriverIdToString(driverProps.driverID),
        static_cast<int>(driverProps.driverID),
        static_cast<unsigned>(driverProps.conformanceVersion.major),
        static_cast<unsigned>(driverProps.conformanceVersion.minor),
        static_cast<unsigned>(driverProps.conformanceVersion.subminor),
        static_cast<unsigned>(driverProps.conformanceVersion.patch));
  });
}
#endif // IGL_LOGGING_ENABLED
} // namespace

NativeHWTextureBuffer::NativeHWTextureBuffer(Device& device, TextureFormat format) :
  Super(device, format) {}

NativeHWTextureBuffer::~NativeHWTextureBuffer() = default;

NativeHWTextureBuffer::AHBImportUsage NativeHWTextureBuffer::resolveAHBImportUsage(
    uint64_t ahbUsage,
    VkImageAspectFlags aspectMask,
    uint64_t externalFormat) noexcept {
  AHBImportUsage out;
  out.textureUsage = igl::android::getIglBufferUsage(ahbUsage);

  if ((ahbUsage & AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) != 0) {
    out.imageUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (((ahbUsage & AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT) != 0) &&
      aspectMask == VK_IMAGE_ASPECT_COLOR_BIT) {
    out.imageUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }
  // Depth/stencil buffers are not required to advertise GPU_COLOR_OUTPUT, so the attachment
  // usage is derived from the format alone.
  if ((aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0) {
    out.imageUsage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  }
  if ((ahbUsage & AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER) != 0) {
    out.imageUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }

  if (externalFormat != 0) {
    // VUID-VkImageCreateInfo-pNext-09457: with a nonzero externalFormat and without the
    // externalFormatResolve feature (which IGL never enables), usage must not include
    // COLOR_ATTACHMENT_BIT. The attachment bit above comes from the producer's AHB usage
    // flags, but IGL only samples imported external images, so drop it from both contracts.
    out.imageUsage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    out.textureUsage = static_cast<TextureDesc::TextureUsage>(
        out.textureUsage & ~TextureDesc::TextureUsageBits::Attachment);
  }
  return out;
}

VkSamplerYcbcrConversion NativeHWTextureBuffer::getVkSamplerYcbcrConversion() const noexcept {
  // Null texture_ and null ycbcrConversion_ are both valid "not available" states.
  if (!texture_) {
    return VK_NULL_HANDLE;
  }
  return texture_->image.ycbcrConversion_;
}

Result NativeHWTextureBuffer::create(const TextureDesc& desc) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  return createHWBuffer(desc, false, false);
}

Result NativeHWTextureBuffer::createTextureInternal(AHardwareBuffer* hwBuffer) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (!hwBuffer) {
    return Result(Result::Code::RuntimeError, "null buffer passed to create texture");
  }
  AHardwareBuffer_Desc hwbDesc{};
  AHardwareBuffer_describe(hwBuffer, &hwbDesc);

  VkImageAspectFlags aspectMask = 0;
  switch (hwbDesc.format) {
  case AHARDWAREBUFFER_FORMAT_D16_UNORM:
  case AHARDWAREBUFFER_FORMAT_D24_UNORM:
  case AHARDWAREBUFFER_FORMAT_D32_FLOAT:
    aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    break;

  case AHARDWAREBUFFER_FORMAT_S8_UINT:
    aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    break;

  case AHARDWAREBUFFER_FORMAT_D24_UNORM_S8_UINT:
  case AHARDWAREBUFFER_FORMAT_D32_FLOAT_S8_UINT:
    aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    break;

  default:
    aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    break;
  }

  auto& ctx = device_.getVulkanContext();
  auto* device = device_.getVulkanContext().getVkDevice();
  VkImageCreateFlags createFlags = 0;
  if (hwbDesc.usage & AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT) {
    createFlags |= VK_IMAGE_CREATE_PROTECTED_BIT;
  }

  VkAndroidHardwareBufferFormatPropertiesANDROID ahbFormatProps = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
      .pNext = nullptr,
  };
  VkAndroidHardwareBufferPropertiesANDROID ahbProps = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
      .pNext = &ahbFormatProps,
  };

  VK_ASSERT(ctx.vf_.vkGetAndroidHardwareBufferPropertiesANDROID(device, hwBuffer, &ahbProps));

#if IGL_LOGGING_ENABLED
  logFirstAhbImportDiagnostics(ahbFormatProps, ctx.getVkPhysicalDeviceDriverProperties());
#endif

  // Some drivers map a camera buffer to a concrete multi-planar VkFormat (for example
  // G8_B8R8_2PLANE_420 for Y8Cb8Cr8_420) and also report an external format. An image view of such
  // a format needs a sampler YCbCr conversion, which only the external-format path below creates,
  // so that path is used for it too. Without an external format there is no conversion to build.
  if (vkFormatRequiresYcbcrConversion(ahbFormatProps.format)) {
    if (ahbFormatProps.externalFormat == 0) {
      IGL_LOG_ERROR(
          "AHB import: format %d needs a YCbCr conversion but the driver gave no external format",
          static_cast<int>(ahbFormatProps.format));
      return Result(Result::Code::RuntimeError,
                    "AHB format needs a YCbCr conversion and has no external format");
    }
    ahbFormatProps.format = VK_FORMAT_UNDEFINED;
  }

  VkExternalFormatANDROID externalFormat = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID,
  };

  if (ahbFormatProps.format == VK_FORMAT_UNDEFINED) {
    externalFormat.externalFormat = ahbFormatProps.externalFormat;
  }

  const AHBImportUsage importUsage =
      resolveAHBImportUsage(hwbDesc.usage, aspectMask, externalFormat.externalFormat);

  VkExternalMemoryImageCreateInfo externalMemoryImageInfo = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .pNext = &externalFormat,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
  };

  // A driver format with no IGL equivalent (R8G8B8_UNORM, for one) stays Invalid, as a release
  // build always did; a debug build used to abort here.
  TextureFormat iglFormat = TextureFormat::Invalid;
  if (!util::tryVkTextureFormatToTextureFormat(static_cast<int32_t>(ahbFormatProps.format),
                                               iglFormat)) {
    IGL_LOG_DEBUG("AHB import: format %d has no IGL texture format\n",
                  static_cast<int>(ahbFormatProps.format));
  }

  auto desc = TextureDesc::newNativeHWBufferImage(
      iglFormat, importUsage.textureUsage, hwbDesc.width, hwbDesc.height);

  VkImage vkImage = VK_NULL_HANDLE;

  VkImageCreateInfo vkImageInfo = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                   .pNext = &externalMemoryImageInfo,
                                   .flags = createFlags,
                                   .imageType = VK_IMAGE_TYPE_2D,
                                   .format = ahbFormatProps.format,
                                   .extent = {.width = static_cast<uint32_t>(desc.width),
                                              .height = static_cast<uint32_t>(desc.height),
                                              .depth = 1},
                                   .mipLevels = 1,
                                   .arrayLayers = 1,
                                   .samples = VK_SAMPLE_COUNT_1_BIT,
                                   .tiling = VK_IMAGE_TILING_OPTIMAL,
                                   .usage = importUsage.imageUsage,
                                   .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                   .queueFamilyIndexCount = 0,
                                   .pQueueFamilyIndices = nullptr,
                                   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  // Create Vk Image.
  VK_ASSERT(ctx.vf_.vkCreateImage(device, &vkImageInfo, nullptr, &vkImage));

  if (vkImage == VK_NULL_HANDLE) {
    IGL_LOG_ERROR("failed to create image view format is %d and external format is %d",
                  vkImageInfo.format,
                  externalFormat.externalFormat);
    return Result(Result::Code::RuntimeError, "Failed to create vulkan image");
  }

  VK_ASSERT(ivkSetDebugObjectName(&ctx.vf_,
                                  device,
                                  VK_OBJECT_TYPE_IMAGE,
                                  (uint64_t)vkImage,
                                  "Image: AHB NativeHWTextureBuffer"));

  // To import memory created outside of the current Vulkan instance from an
  // Android hardware buffer, add a VkImportAndroidHardwareBufferInfoANDROID
  // structure to the pNext chain of the VkMemoryAllocateInfo structure.
  VkImportAndroidHardwareBufferInfoANDROID ahbImportInfo = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
      .pNext = nullptr,
      .buffer = hwBuffer};

  // If the VkMemoryAllocateInfo pNext chain includes a
  // VkMemoryDedicatedAllocateInfo structure, then that structure includes a
  // handle of the sole buffer or image resource that the memory can be bound
  // to.
  VkMemoryDedicatedAllocateInfo dedicatedAllocInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .pNext = &ahbImportInfo,
      .image = vkImage,
      .buffer = VK_NULL_HANDLE};

  // Find the memory type that supports the required properties.
  uint32_t memoryTypeBits = ahbProps.memoryTypeBits;

  uint32_t typeIndex = ivkGetMemoryTypeIndex(
      ctx.memoryProperties, memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // An instance of the VkMemoryAllocateInfo structure defines a memory import
  // operation.
  VkMemoryAllocateInfo memAllocInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &dedicatedAllocInfo,
      // If the parameters define an import operation and the external handle type
      // is VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
      // allocationSize must be the size returned by
      // vkGetAndroidHardwareBufferPropertiesANDROID() for the Android hardware
      // buffer.
      .allocationSize = ahbProps.allocationSize,
      .memoryTypeIndex = typeIndex};

  // A Vulkan device operates on data in device memory via memory objects that
  // are represented in the API by a VkDeviceMemory handle.
  // Allocate memory.
  VkDeviceMemory vkDeviceMemory = VK_NULL_HANDLE;
  VK_ASSERT(ctx.vf_.vkAllocateMemory(device, &memAllocInfo, nullptr, &vkDeviceMemory));

  VK_ASSERT(ivkSetDebugObjectName(&ctx.vf_,
                                  device,
                                  VK_OBJECT_TYPE_DEVICE_MEMORY,
                                  (uint64_t)vkDeviceMemory,
                                  "Memory: AHB NativeHWTextureBuffer"));

  // Attach memory to the image object.
  VK_ASSERT(ctx.vf_.vkBindImageMemory(device, vkImage, vkDeviceMemory, 0));

  auto vulkanImage = VulkanImage(ctx,
                                 vkImage,
                                 "Image: videoTexture",
                                 importUsage.imageUsage,
                                 false,
                                 vkImageInfo.extent,
                                 vkImageInfo.imageType,
                                 vkImageInfo.format,
                                 vkImageInfo.mipLevels,
                                 vkImageInfo.arrayLayers,
                                 VK_SAMPLE_COUNT_1_BIT,
                                 true);
  vulkanImage.vkMemory_[0] = vkDeviceMemory;
  vulkanImage.extendedFormat_ = externalFormat.externalFormat;

  VkImageViewCreateInfo viewInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = vkImage,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = vkImageInfo.format,
      .components =
          {
              .r = VK_COMPONENT_SWIZZLE_IDENTITY,
              .g = VK_COMPONENT_SWIZZLE_IDENTITY,
              .b = VK_COMPONENT_SWIZZLE_IDENTITY,
              .a = VK_COMPONENT_SWIZZLE_IDENTITY,
          },
      .subresourceRange = {.aspectMask = aspectMask,
                           .baseMipLevel = 0,
                           .levelCount = vkImageInfo.mipLevels,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
  };

  viewInfo.pNext = nullptr;
  VkSamplerYcbcrConversionInfo conversionInfo = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
      .pNext = nullptr,
      .conversion = VK_NULL_HANDLE,
  };

  if (ahbFormatProps.format == VK_FORMAT_UNDEFINED && externalFormat.externalFormat) {
    viewInfo.pNext = &conversionInfo;
    // Use the driver-reported raw-plane swizzle, with a legacy Qualcomm-Adreno correction for
    // Camera2 YUV_420_888 streams whose reported IDENTITY mapping still swaps Cb/Cr.
    // VK_KHR_driver_properties gives a narrow gate; zeroed pre-1.2 properties skip the workaround.
    VkComponentMapping components = ahbFormatProps.samplerYcbcrConversionComponents;
    const VkPhysicalDeviceDriverPropertiesKHR& driverProps =
        ctx.getVkPhysicalDeviceDriverProperties();
    const bool isQualcommDriver = driverProps.driverID == VK_DRIVER_ID_QUALCOMM_PROPRIETARY;
    const bool isLegacyConformance =
        driverProps.conformanceVersion.major < 1 ||
        (driverProps.conformanceVersion.major == 1 && driverProps.conformanceVersion.minor < 3);
    if (isQualcommDriver && isLegacyConformance) {
      components.r = VK_COMPONENT_SWIZZLE_B;
      components.b = VK_COMPONENT_SWIZZLE_R;
    }
    // A driver whose external format lacks the linear-filter feature (PowerVR, for one) accepts
    // only a nearest chroma filter in the conversion; the context's cache keys on the filter too.
    const VkFilter chromaFilter =
        formatFeaturesAllowLinearChromaFilter(ahbFormatProps.formatFeatures) ? VK_FILTER_LINEAR
                                                                             : VK_FILTER_NEAREST;
    vulkanImage.samplerYcbcrConversionCreateInfo_ = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .pNext = &externalFormat,
        .format = ahbFormatProps.format,
        .ycbcrModel = ahbFormatProps.suggestedYcbcrModel,
        .ycbcrRange = ahbFormatProps.suggestedYcbcrRange,
        .components = components,
        .xChromaOffset = ahbFormatProps.suggestedXChromaOffset,
        .yChromaOffset = ahbFormatProps.suggestedYChromaOffset,
        .chromaFilter = chromaFilter,
        .forceExplicitReconstruction = VK_FALSE};

    // Reuse the context-owned conversion so AImageReader pool slots share one handle.
    // This avoids rebuilding immutable-sampler pipelines when the slot rotates.
    conversionInfo.conversion =
        ctx.getOrCreateExternalYcbcrConversion(vulkanImage.samplerYcbcrConversionCreateInfo_);
    if (conversionInfo.conversion == VK_NULL_HANDLE) {
      return Result(Result::Code::RuntimeError,
                    "Failed to create external-format YCbCr conversion for AHB import");
    }
    // Expose the non-owning handle to consumers that need a matching VkSampler.
    vulkanImage.ycbcrConversion_ = conversionInfo.conversion;
  }

  VulkanImageView vulkanImageView(ctx, viewInfo, "Image View: videoTexture");

  auto vkTexture = device_.getVulkanContext().createTexture(
      std::move(vulkanImage), std::move(vulkanImageView), "SurfaceTexture");

  if (!vkTexture) {
    return Result(Result::Code::RuntimeError, "Failed to create vulkan texture");
  }

  desc_ = desc; // Field within the Texture class
  textureDesc_ = desc; // Field within the NativeHWTextureBuffer class
  texture_ = std::move(vkTexture);

  return Result{Result::Code::Ok};
}

Result NativeHWTextureBuffer::uploadInternal(TextureType /*type*/,
                                             const TextureRangeDesc& range,
                                             const void* IGL_NULLABLE data,
                                             size_t bytesPerRow,
                                             const uint32_t* IGL_NULLABLE /*mipLevelBytes*/) const {
  IGL_PROFILER_FUNCTION();
  auto result = uploadToHWBuffer(getProperties(), range, data, bytesPerRow);
  if (!result.isOk()) {
    IGL_DEBUG_ABORT("Cannot upload buffer for HW texture (vulkan NativeHWTextureBuffer).");
    return result;
  }

  // The underlying VkImage was created with VK_IMAGE_LAYOUT_UNDEFINED and the pixel
  // upload above happens through the AHardwareBuffer CPU lock path (no GPU commands).
  // Without an explicit layout transition, the image stays in UNDEFINED, which is
  // not legal to sample from in shaders. Transition to SHADER_READ_ONLY_OPTIMAL via
  // a one-shot immediate command buffer so subsequent draws can sample the texture.
  if (texture_) {
    const VkImageSubresourceRange subresourceRange = {
        .aspectMask = [numPlanes = igl::vulkan::getNumImagePlanes(
                           texture_->image.imageFormat_)]() -> VkImageAspectFlags {
          if (numPlanes < 2) {
            return VK_IMAGE_ASPECT_COLOR_BIT;
          }
          VkImageAspectFlags aspectMask = 0;
          for (uint32_t p = 0; p < numPlanes; p++) {
            aspectMask |= static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_PLANE_0_BIT << p);
          }
          return aspectMask;
        }(),
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    auto& ctx = device_.getVulkanContext();
    const auto& wrapper = ctx.immediate_->acquire();
    texture_->image.transitionLayout(wrapper.cmdBuf,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     subresourceRange);
    ctx.immediate_->submit(wrapper);
  }

  return result;
}

} // namespace igl::vulkan::android

#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
