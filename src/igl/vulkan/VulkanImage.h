/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanHelpers.h>
#include <igl/vulkan/VulkanImageView.h>

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
struct AHardwareBuffer;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

namespace igl::vulkan {

class VulkanContext;

struct VulkanImageCreateInfo {
  VkImageUsageFlags usageFlags = 0;
  bool isExternallyManaged = true;
  VkExtent3D extent = VkExtent3D{.width = 0, .height = 0, .depth = 0};
  VkImageType type = VK_IMAGE_TYPE_MAX_ENUM;
  VkFormat imageFormat = VK_FORMAT_UNDEFINED;
  uint32_t mipLevels = 1;
  uint32_t arrayLayers = 1;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  bool isImported = false;
  bool isSrgbMutableFormat = false;
};

/**
 * @brief Encapsulates a Vulkan Image object (`VkImage`) along with some of its properties
 */
class VulkanImage final {
 public:
  explicit VulkanImage() = default;
  /**
   * @brief Constructs a `VulkanImage` object from a `VkImage` object. If a debug name is provided,
   * the constructor will assign it to the `VkImage` object. No other Vulkan functions are called
   */
  VulkanImage(const VulkanContext& ctx,
              VkImage image,
              const char* debugName = nullptr,
              VkImageUsageFlags usageFlags = 0,
              bool isExternallyManaged = true,
              VkExtent3D extent = VkExtent3D{.width = 0, .height = 0, .depth = 0},
              VkImageType type = VK_IMAGE_TYPE_MAX_ENUM,
              VkFormat imageFormat = VK_FORMAT_UNDEFINED,
              uint32_t mipLevels = 1,
              uint32_t arrayLayers = 1,
              VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
              bool isImported = false,
              bool isSrgbMutableFormat = false);

  /**
   * @brief Constructs a `VulkanImage` object from a `VkImage` object. If a debug name is provided,
   * the constructor will assign it to the `VkImage` object. No other Vulkan functions are called
   */
  VulkanImage(const VulkanContext& ctx,
              VkImage image,
              const VulkanImageCreateInfo& createInfo,
              const char* debugName = nullptr);

  /**
   * @brief Constructs a `VulkanImage` object and a `VkImage` object. Except for the debug name, all
   * other parameters are required. The debug name, if provided, is associated with the newly
   * created `VkImage` object.
   *
   * The image must contain at least one mip level, one array layer and one sample
   * (`VK_SAMPLE_COUNT_1_BIT`). The format cannot be undefined (`VK_FORMAT_UNDEFINED`).
   *
   * If the image is host-visible (`memFlags` contains `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`), then
   * it is memory mapped until the object's destruction.
   *
   * If `memFlags` contains `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT` and no lazily allocated memory
   * can back the image, the image is allocated without that flag instead.
   */
  VulkanImage(const VulkanContext& ctx,
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
              const char* debugName = nullptr,
              bool isSrgbMutableFormat = false);

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
  /**
   * @brief Constructs a `VulkanImage` object and a `VkImage` object from an `AHardwareBuffer`. The
   * `VkImage` object is backed by external memory.
   *
   * This constructor does not support VMA.
   *
   * Except for the debug name, all other parameters are required. The debug name, if provided, is
   * associated with the newly created `VkImage` object.
   *
   * The image must contain at least one mip-level, one array layer and one sample
   * (`VK_SAMPLE_COUNT_1_BIT`). The format cannot be undefined (`VK_FORMAT_UNDEFINED`).
   *
   * If the image is host-visible (`memFlags` contains `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`), then
   * it is memory mapped until the object's destruction.
   *
   * This constructor is only supported on Android environments.
   *
   * NOTE: Importing a hardware buffer causes Vulkan to acquire a reference to the hardware buffer,
   * which it releases when the allocated memory is freed. Destroying a VulkanImage does not
   * necessarily guarantee that the underlying hardware buffer is also freed, as the application may
   * retain additional references to the hardware buffer.
   */
  VulkanImage(const VulkanContext& ctx,
              AHardwareBuffer* ahb,
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
              const char* debugName = nullptr);
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
  /**
   * @brief Constructs a `VulkanImage` object and a `VkImage` object from a file descriptor. The
   * `VkImage` object is backed by external memory. The handle type of the external memory used is
   * `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR`.
   *
   * This constructor does not support VMA.
   *
   * Except for the debug name, all other parameters are required. The debug name, if provided, is
   * associated with the newly created VkImage object.
   *
   * The image must contain at least one mip level, one array layer and one sample
   * (`VK_SAMPLE_COUNT_1_BIT`). The format cannot be undefined (`VK_FORMAT_UNDEFINED`).
   *
   * If the image is host-visible (`memFlags` contains `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`), then
   * it is memory mapped until the object's destruction.
   *
   * This constructor is only supported on non-Windows environments.
   *
   * NOTE: Importing memory from a file descriptor transfers ownership of the descriptor from the
   * application to the Vulkan implementation. The application must not perform any operations on
   * the file descriptor after a successful import. The file descriptors are closed on object's
   * destruction automatically.
   */
  VulkanImage(const VulkanContext& ctx,
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
              const char* debugName = nullptr);

#if IGL_PLATFORM_WINDOWS
  /**
   * @brief Creates a `VulkanImage` with memory imported from a Windows handle.
   * NOTE:
   * 1. This should only be called on Windows and will crash on other platforms.
   * 2. The implementation currently only supports handleType
   * `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT`, which means the handle must be created
   * by Vulkan API, not other graphics API.
   */
  VulkanImage(const VulkanContext& ctx,
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
              const char* debugName = nullptr);
#endif // IGL_PLATFORM_WINDOWS

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID
  /**
   * @brief Creates a `VulkanImage` object whose memory can be exported externally.
   * On Windows, the exported `HANDLE` will be stored in `exportedMemoryHandle_`.
   * On Linux/Android, the exported file descriptor will be stored in `exportedFd_`.
   */
  static VulkanImage createWithExportMemory(const VulkanContext& ctx,
                                            VkExtent3D extent,
                                            VkImageType type,
                                            VkFormat format,
                                            uint32_t mipLevels,
                                            uint32_t arrayLayers,
                                            VkImageTiling tiling,
                                            VkImageUsageFlags usageFlags,
                                            VkImageCreateFlags createFlags,
                                            VkSampleCountFlagBits samples,
                                            const char* debugName = nullptr);
#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID

  ~VulkanImage();

  VulkanImage(const VulkanImage&) = default;
  VulkanImage& operator=(const VulkanImage&) = delete;

  VulkanImage(VulkanImage&& other) noexcept {
    *this = std::move(other);
  }
  VulkanImage& operator=(VulkanImage&& other) noexcept;

  [[nodiscard]] VkImage getVkImage() const {
    return vkImage_;
  }

  /**
   * @brief Returns true if the object is valid: constructed, and its construction succeeded.
   */
  [[nodiscard]] bool valid() const;

  /**
   * @brief The result of the first construction step that failed (`VK_SUCCESS` for a valid image).
   * An image whose construction failed owns no Vulkan object.
   */
  [[nodiscard]] VkResult initResult() const noexcept {
    return initResult_;
  }

  [[nodiscard]] VkImageUsageFlags getVkImageUsageFlags() const {
    return usageFlags_;
  }

  [[nodiscard]] bool isSampledImage() const {
    return (usageFlags_ & VK_IMAGE_USAGE_SAMPLED_BIT) > 0;
  }

  [[nodiscard]] bool isStorageImage() const {
    return (usageFlags_ & VK_IMAGE_USAGE_STORAGE_BIT) > 0;
  }

  /**
   * @brief Creates a `VkImageView` object from the `VkImage` stored in the object.
   *
   * Setting `numLevels` to a non-zero value will override `mipLevels_` value from the original
   * vulkan image, and can be used to create image views with different number of levels
   */
  VulkanImageView createImageView(VkImageViewType type,
                                  VkFormat format,
                                  VkImageAspectFlags aspectMask,
                                  uint32_t baseLevel,
                                  uint32_t numLevels = VK_REMAINING_MIP_LEVELS,
                                  uint32_t baseLayer = 0,
                                  uint32_t numLayers = 1,
                                  const char* debugName = nullptr) const;
  VulkanImageView createImageView(VulkanImageViewCreateInfo createInfo,
                                  const char* debugName = nullptr) const;
  /**
   * @brief Destroys the image and its memory at once instead of through deferred tasks and leaves
   * the object invalid. Only for an image that no GPU work can reference because it was refused
   * right after it was created. No-op for an invalid image or one whose VkImage is not owned.
   */
  void discardNeverSubmitted();
  [[nodiscard]] Result generateMipmap(VkCommandBuffer commandBuffer,
                                      const TextureRangeDesc& range) const;

  /**
   * @brief Transitions the `VkImage`'s layout from the current layout (stored in the object) to the
   * `newImageLayout` by recording an Image Memory Barrier into the commandBuffer.
   *
   * The source and destination access masks for the transition are automatically deduced based on
   * the `srcStageMask` and the `dstStageMask` parameters. Not not all `VkPipelineStageFlags` are
   * supported.
   */
  void transitionLayout(VkCommandBuffer cmdBuf,
                        VkImageLayout newImageLayout,
                        VkPipelineStageFlags srcStageMask,
                        VkPipelineStageFlags dstStageMask,
                        const VkImageSubresourceRange& subresourceRange) const;
  void clearColorImage(VkCommandBuffer commandBuffer,
                       const Color& rgba,
                       const VkImageSubresourceRange* subresourceRange = nullptr) const;

  [[nodiscard]] VkImageAspectFlags getImageAspectFlags() const;

  [[nodiscard]] bool isMappedPtrAccessible() const {
    return (mappedPtr_ != nullptr) && ((tiling_ & VK_IMAGE_TILING_LINEAR) != 0);
  }

  [[nodiscard]] bool isCoherentMemory() const {
    return isCoherentMemory_;
  }

  void flushMappedMemory() const;

 public:
  // Vulkan as for v1.3.210 supports max 3 planes for multi-plane images.
  static constexpr uint8_t kMaxImagePlanes = 3;

  const VulkanContext* ctx_ = nullptr;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkImage vkImage_ = VK_NULL_HANDLE;
  VkImageUsageFlags usageFlags_ = 0;
  // Separate VkDeviceMemory objects to support disjoint multiplanar images
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDeviceMemory vkMemory_[kMaxImagePlanes] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
  VmaAllocation vmaAllocation_ = VK_NULL_HANDLE;
  VkFormatProperties formatProperties_{};
  void* mappedPtr_ = nullptr;
  bool isExternallyManaged_ = false;
  VkExtent3D extent_ = {.width = 0, .height = 0, .depth = 0};
  VkImageType type_ = VK_IMAGE_TYPE_MAX_ENUM;
  VkFormat imageFormat_ = VK_FORMAT_UNDEFINED;
  uint32_t mipLevels_ = 1;
  uint32_t arrayLayers_ = 1;
  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
  bool isDepthFormat_ = false;
  bool isStencilFormat_ = false;
  // NOLINTNEXTLINE(readability-identifier-naming)
  bool isDepthOrStencilFormat_ = false;
  VkDeviceSize allocatedSize = 0;
  mutable VkImageLayout imageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED; // current image layout
  bool isImported_ = false;
  bool isExported_ = false;
  bool isCubemap_ = false;
  // True when the VkImage was created UNORM with VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT and a
  // requested VkImageFormatListCreateInfo constraining views to exactly {UNORM, sRGB}, or when a
  // wrapped VkImage was explicitly marked as having the same externally-enforced shape.
  // NOLINTNEXTLINE(readability-identifier-naming)
  bool isSrgbMutableFormat_ = false;
  void* exportedMemoryHandle_ = nullptr; // windows handle
  int exportedFd_ = -1; // linux fd
  uint32_t extendedFormat_ = 0; // defined by VkAndroidHardwareBufferFormatPropertiesANDROID
  // NOLINTNEXTLINE(readability-identifier-naming)
  VkSamplerYcbcrConversionCreateInfo samplerYcbcrConversionCreateInfo_ = {};
  // Non-owning reference to a context-cached conversion for Android external-format AHB images.
  // Consumers that create a VkSampler need the same conversion handle used by the image view.
  // VK_NULL_HANDLE for non-AHB images and defined-multi-plane AHB imports.
  // NOLINTNEXTLINE(readability-identifier-naming)
  VkSamplerYcbcrConversion ycbcrConversion_ = VK_NULL_HANDLE;
#if IGL_DEBUG
  std::string name_;
#endif

 private:
  /// Destroys what a failing constructor created, clears the handles and records `result`.
  void failInit(VkResult result);

  VkImageTiling tiling_ = VK_IMAGE_TILING_OPTIMAL;
  bool isCoherentMemory_ = false;
  VkResult initResult_ = VK_SUCCESS;

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID
  /**
   * @brief Constructs a `VulkanImage` object and a `VkImage` object. Except for the debug name, all
   * other parameters are required. The debug name, if provided, is associated with the newly
   * created `VkImage` object.
   *
   * This version is only supported on Windows, Linux and Android environments and accepts both
   * `VkExternalMemoryImageCreateInfoKHR` and `VkExportMemoryAllocateInfoKHR` pre-filled structures.
   *
   * On Windows, the external memory handle is stored in `exportedMemoryHandle_`. On Linux and
   * Android the external memory handle is stored in `exportedFd_`.
   *
   * The image must contain at least one mip level, one array layer and one sample
   * (`VK_SAMPLE_COUNT_1_BIT`). The format cannot be undefined (`VK_FORMAT_UNDEFINED`).
   *
   * If the image is host-visible (`memFlags` contains `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`), then
   * it is memory mapped until the object's destruction.
   */
  VulkanImage(const VulkanContext& ctx,
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
              const char* debugName);
#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_LINUX || IGL_PLATFORM_ANDROID

  // Records the vkCmdBlitImage() mip-generation chain directly on this image. Correct only when the
  // VkImage's format matches the texture's color space (native UNORM or native sRGB); see
  // generateMipmapSrgb() for the UNORM-base sRGB case.
  [[nodiscard]] Result generateMipmapBlit(VkCommandBuffer commandBuffer,
                                          const TextureRangeDesc& range) const;

  // Gamma-correct mipmap generation for sRGB textures whose backing VkImage is UNORM
  // (isSrgbMutableFormat_). vkCmdBlitImage() filters in the image's creation-format space and
  // cannot take a view, so it would average sRGB-encoded texels as if linear. Round-trips the
  // source mip through a transient native-sRGB scratch image where the normal blit path downsamples
  // correctly. Color and full layer/face range only.
  [[nodiscard]] Result generateMipmapSrgb(VkCommandBuffer commandBuffer,
                                          const TextureRangeDesc& range) const;

  // No-op in all builds except DEBUG
  void setName(std::string name) noexcept;

  void destroy();
};

} // namespace igl::vulkan
