/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This is a very low-level C wrapper of the Vulkan API to make it slightly less painful to use.

#pragma once

#include <glslang/Include/glslang_c_interface.h>
#include <igl/Macros.h>
#include <igl/vulkan/VulkanFunctionTable.h>
#include <igl/vulkan/VulkanVma.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Adds a node to the linked list of next nodes
void ivkAddNext(void* node, const void* next);

const char* ivkGetVulkanResultString(VkResult result);

/// @brief Creates a Debug Utils Messenger if the VK_EXT_debug_utils extension is available and the
/// platform is not Android or Mac Catalyst. Otherwise the function is defined as a no-op that
/// always returns VK_SUCCESS
VkResult ivkCreateDebugUtilsMessenger(const struct VulkanFunctionTable* vt,
                                      VkInstance instance,
                                      PFN_vkDebugUtilsMessengerCallbackEXT callback,
                                      void* logUserData,
                                      VkDebugUtilsMessengerEXT* outMessenger);

// This function uses VK_EXT_debug_report extension, which is deprecated by VK_EXT_debug_utils.
// However, it is available on some Android devices where VK_EXT_debug_utils is not available.
VkResult ivkCreateDebugReportMessenger(const struct VulkanFunctionTable* vt,
                                       VkInstance instance,
                                       PFN_vkDebugReportCallbackEXT callback,
                                       void* logUserData,
                                       VkDebugReportCallbackEXT* outMessenger);

/** @brief Creates a platform specific VkSurfaceKHR object. The surface creation functions
 * conditionally-compiled and guarded by their respective platform specific extension macros defined
 * by the Vulkan API. The current supported platforms, and their macros, are:
 * - Windows: VK_USE_PLATFORM_WIN32_KHR
 * - Linux: VK_USE_PLATFORM_XLIB_KHR
 * - MacOS: VK_USE_PLATFORM_METAL_EXT
 * - Android: VK_USE_PLATFORM_ANDROID_KHR
 */
VkResult ivkCreateSurface(const struct VulkanFunctionTable* vt,
                          VkInstance instance,
                          void* window,
                          void* display,
                          void* layer,
                          VkSurfaceKHR* outSurface);

/** @brief Creates a Vulkan device from the given physical device.
 * Enables the core features in `enabledFeatures->features` and its `pNext` feature chain exactly as
 * supplied. The caller selects supported features to enable, as
 * `VulkanFeatures::enableRequestedFeatures()` does.
 */
VkResult ivkCreateDevice(const struct VulkanFunctionTable* vt,
                         VkPhysicalDevice physicalDevice,
                         size_t numQueueCreateInfos,
                         const VkDeviceQueueCreateInfo* queueCreateInfos,
                         size_t numDeviceExtensions,
                         const char** deviceExtensions,
                         const VkPhysicalDeviceFeatures2* enabledFeatures,
                         VkDevice* outDevice);

VkResult ivkCreateSwapchain(const struct VulkanFunctionTable* vt,
                            VkDevice device,
                            VkSurfaceKHR surface,
                            uint32_t minImageCount,
                            VkSurfaceFormatKHR surfaceFormat,
                            VkPresentModeKHR presentMode,
                            const VkSurfaceCapabilitiesKHR* caps,
                            VkImageUsageFlags imageUsage,
                            uint32_t queueFamilyIndex,
                            uint32_t width,
                            uint32_t height,
                            VkSwapchainKHR* outSwapchain);

VkResult ivkCreateCommandPool(const struct VulkanFunctionTable* vt,
                              VkDevice device,
                              VkCommandPoolCreateFlags flags,
                              uint32_t queueFamilyIndex,
                              VkCommandPool* outCommandPool);

VkResult ivkAllocateCommandBuffer(const struct VulkanFunctionTable* vt,
                                  VkDevice device,
                                  VkCommandPool commandPool,
                                  VkCommandBuffer* outCommandBuffer);

// `outMemoryPropertyFlags` may be NULL; when non-NULL it receives the property flags of the memory
// type used for the allocation, allowing callers to inspect the properties of the memory that was
// actually allocated without re-querying the physical device.
VkResult ivkAllocateMemory(const struct VulkanFunctionTable* vt,
                           VkPhysicalDevice physDev,
                           VkDevice device,
                           const VkMemoryRequirements* memRequirements,
                           VkMemoryPropertyFlags props,
                           bool enableBufferDeviceAddress,
                           VkDeviceMemory* outMemory,
                           VkMemoryPropertyFlags* outMemoryPropertyFlags);

VkResult ivkAllocateMemory2(const struct VulkanFunctionTable* vt,
                            VkPhysicalDevice physDev,
                            VkDevice device,
                            const VkMemoryRequirements2* memRequirements,
                            VkMemoryPropertyFlags props,
                            bool enableBufferDeviceAddress,
                            VkDeviceMemory* outMemory);

bool ivkIsHostVisibleSingleHeapMemory(const VkPhysicalDeviceMemoryProperties* memProps);

uint32_t ivkFindMemoryType(const VkPhysicalDeviceMemoryProperties* memProps,
                           uint32_t memoryTypeBits,
                           VkMemoryPropertyFlags flags);

VkResult ivkCreateGraphicsPipeline(const struct VulkanFunctionTable* vt,
                                   VkDevice device,
                                   VkPipelineCache pipelineCache,
                                   VkPipelineCreateFlags flags,
                                   uint32_t numShaderStages,
                                   const VkPipelineShaderStageCreateInfo* shaderStages,
                                   const VkPipelineVertexInputStateCreateInfo* vertexInputState,
                                   const VkPipelineInputAssemblyStateCreateInfo* inputAssemblyState,
                                   const VkPipelineTessellationStateCreateInfo* tessellationState,
                                   const VkPipelineViewportStateCreateInfo* viewportState,
                                   const VkPipelineRasterizationStateCreateInfo* rasterizationState,
                                   const VkPipelineMultisampleStateCreateInfo* multisampleState,
                                   const VkPipelineDepthStencilStateCreateInfo* depthStencilState,
                                   const VkPipelineColorBlendStateCreateInfo* colorBlendState,
                                   const VkPipelineDynamicStateCreateInfo* dynamicState,
                                   VkPipelineLayout pipelineLayout,
                                   VkRenderPass renderPass,
                                   uint32_t subpass,
                                   VkPipeline* outPipeline);

VkResult ivkCreateDescriptorSetLayout(const struct VulkanFunctionTable* vt,
                                      VkDevice device,
                                      VkDescriptorSetLayoutCreateFlags flags,
                                      uint32_t numBindings,
                                      const VkDescriptorSetLayoutBinding* bindings,
                                      const VkDescriptorBindingFlags* bindingFlags,
                                      VkDescriptorSetLayout* outLayout);

VkResult ivkAllocateDescriptorSet(const struct VulkanFunctionTable* vt,
                                  VkDevice device,
                                  VkDescriptorPool pool,
                                  VkDescriptorSetLayout layout,
                                  VkDescriptorSet* outDescriptorSet);

VkResult ivkCreateDescriptorPool(const struct VulkanFunctionTable* vt,
                                 VkDevice device,
                                 VkDescriptorPoolCreateFlags flags,
                                 uint32_t maxDescriptorSets,
                                 uint32_t numPoolSizes,
                                 const VkDescriptorPoolSize* poolSizes,
                                 VkDescriptorPool* outDescriptorPool);

VkWriteDescriptorSet ivkGetWriteDescriptorSetImageInfo(VkDescriptorSet dstSet,
                                                       uint32_t dstBinding,
                                                       VkDescriptorType descriptorType,
                                                       uint32_t numDescriptors,
                                                       const VkDescriptorImageInfo* pImageInfo);

VkWriteDescriptorSet ivkGetWriteDescriptorSetBufferInfo(VkDescriptorSet dstSet,
                                                        uint32_t dstBinding,
                                                        VkDescriptorType descriptorType,
                                                        uint32_t numDescriptors,
                                                        const VkDescriptorBufferInfo* pBufferInfo);

void ivkImageMemoryBarrier(const struct VulkanFunctionTable* vt,
                           VkCommandBuffer buffer,
                           VkImage image,
                           VkAccessFlags srcAccessMask,
                           VkAccessFlags dstAccessMask,
                           VkImageLayout oldImageLayout,
                           VkImageLayout newImageLayout,
                           VkPipelineStageFlags srcStageMask,
                           VkPipelineStageFlags dstStageMask,
                           VkImageSubresourceRange subresourceRange);

void ivkBufferBarrier(const struct VulkanFunctionTable* vt,
                      VkCommandBuffer cmdBuffer,
                      VkBuffer buffer,
                      VkBufferUsageFlags usageFlags,
                      VkPipelineStageFlags srcStageMask,
                      VkPipelineStageFlags dstStageMask);

void ivkCmdBlitImage(const struct VulkanFunctionTable* vt,
                     VkCommandBuffer buffer,
                     VkImage srcImage,
                     VkImage dstImage,
                     VkImageLayout srcImageLayout,
                     VkImageLayout dstImageLayout,
                     const VkOffset3D* srcOffsets,
                     const VkOffset3D* dstOffsets,
                     VkImageSubresourceLayers srcSubresourceRange,
                     VkImageSubresourceLayers dstSubresourceRange,
                     VkFilter filter);

/// @brief Adds a name for the Vulkan object with handle equals to `handle` and type equals to
/// `type`. This function is a no-op if `VK_EXT_DEBUG_UTILS_SUPPORTED` is not defined
VkResult ivkSetDebugObjectName(const struct VulkanFunctionTable* vt,
                               VkDevice device,
                               VkObjectType type,
                               uint64_t handle,
                               const char* name);

/// @brief Opens a command buffer debug region. This function is a no-op if
/// `VK_EXT_DEBUG_UTILS_SUPPORTED` is not defined
void ivkCmdBeginDebugUtilsLabel(const struct VulkanFunctionTable* vt,
                                VkCommandBuffer buffer,
                                const char* name,
                                const float colorRGBA[4]);

/// @brief Inserts a debug label into a command buffer. This function is a no-op if
/// `VK_EXT_DEBUG_UTILS_SUPPORTED` is not defined
void ivkCmdInsertDebugUtilsLabel(const struct VulkanFunctionTable* vt,
                                 VkCommandBuffer buffer,
                                 const char* name,
                                 const float colorRGBA[4]);

/// @brief Closes a command buffer debug region. This function is a no-op if
/// `VK_EXT_DEBUG_UTILS_SUPPORTED` is not defined
void ivkCmdEndDebugUtilsLabel(const struct VulkanFunctionTable* vt, VkCommandBuffer buffer);

VkVertexInputBindingDescription ivkGetVertexInputBindingDescription(uint32_t binding,
                                                                    uint32_t stride,
                                                                    VkVertexInputRate inputRate);
VkVertexInputAttributeDescription ivkGetVertexInputAttributeDescription(uint32_t location,
                                                                        uint32_t binding,
                                                                        VkFormat format,
                                                                        uint32_t offset);

VkResult ivkVmaCreateAllocator(const struct VulkanFunctionTable* vt,
                               VkPhysicalDevice physDev,
                               VkDevice device,
                               VkInstance instance,
                               uint32_t apiVersion,
                               bool enableBufferDeviceAddress,
                               VkDeviceSize preferredLargeHeapBlockSize,
                               VmaAllocator* outVma);

void ivkUpdateGlslangResource(glslang_resource_t* res,
                              const VkPhysicalDeviceProperties* props,
                              const VkPhysicalDeviceMeshShaderPropertiesEXT* meshShaderProps);

#ifdef __cplusplus
}
#endif
