/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(VK_USE_PLATFORM_WIN32_KHR)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(VK_USE_PLATFORM_XLIB_KHR)
#include <X11/Xlib.h>
#endif

#include "VulkanHelpers.h"

#include <assert.h>

const char* ivkGetVulkanResultString(VkResult result) {
#define RESULT_CASE(res) \
  case res:              \
    return #res
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (result) {
    RESULT_CASE(VK_SUCCESS);
    RESULT_CASE(VK_NOT_READY);
    RESULT_CASE(VK_TIMEOUT);
    RESULT_CASE(VK_EVENT_SET);
    RESULT_CASE(VK_EVENT_RESET);
    RESULT_CASE(VK_INCOMPLETE);
    RESULT_CASE(VK_ERROR_OUT_OF_HOST_MEMORY);
    RESULT_CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    RESULT_CASE(VK_ERROR_INITIALIZATION_FAILED);
    RESULT_CASE(VK_ERROR_DEVICE_LOST);
    RESULT_CASE(VK_ERROR_MEMORY_MAP_FAILED);
    RESULT_CASE(VK_ERROR_LAYER_NOT_PRESENT);
    RESULT_CASE(VK_ERROR_EXTENSION_NOT_PRESENT);
    RESULT_CASE(VK_ERROR_FEATURE_NOT_PRESENT);
    RESULT_CASE(VK_ERROR_INCOMPATIBLE_DRIVER);
    RESULT_CASE(VK_ERROR_TOO_MANY_OBJECTS);
    RESULT_CASE(VK_ERROR_FORMAT_NOT_SUPPORTED);
    RESULT_CASE(VK_ERROR_SURFACE_LOST_KHR);
    RESULT_CASE(VK_ERROR_OUT_OF_DATE_KHR);
    RESULT_CASE(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR);
    RESULT_CASE(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
    RESULT_CASE(VK_ERROR_VALIDATION_FAILED_EXT);
    RESULT_CASE(VK_ERROR_FRAGMENTED_POOL);
    RESULT_CASE(VK_ERROR_UNKNOWN);
    // Provided by VK_VERSION_1_1
    RESULT_CASE(VK_ERROR_OUT_OF_POOL_MEMORY);
    // Provided by VK_VERSION_1_1
    RESULT_CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE);
    // Provided by VK_VERSION_1_2
    RESULT_CASE(VK_ERROR_FRAGMENTATION);
    // Provided by VK_VERSION_1_2
    RESULT_CASE(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
    // Provided by VK_KHR_swapchain
    RESULT_CASE(VK_SUBOPTIMAL_KHR);
    // Provided by VK_NV_glsl_shader
    RESULT_CASE(VK_ERROR_INVALID_SHADER_NV);
#ifdef VK_ENABLE_BETA_EXTENSIONS
    // Provided by VK_KHR_video_queue
    RESULT_CASE(VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR);
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
    // Provided by VK_KHR_video_queue
    RESULT_CASE(VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR);
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
    // Provided by VK_KHR_video_queue
    RESULT_CASE(VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR);
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
    // Provided by VK_KHR_video_queue
    RESULT_CASE(VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR);
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
    // Provided by VK_KHR_video_queue
    RESULT_CASE(VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR);
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
    // Provided by VK_KHR_video_queue
    RESULT_CASE(VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR);
#endif
    // Provided by VK_EXT_image_drm_format_modifier
    RESULT_CASE(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
    // Provided by VK_KHR_global_priority
    RESULT_CASE(VK_ERROR_NOT_PERMITTED_KHR);
    // Provided by VK_EXT_full_screen_exclusive
    RESULT_CASE(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
    // Provided by VK_KHR_deferred_host_operations
    RESULT_CASE(VK_THREAD_IDLE_KHR);
    // Provided by VK_KHR_deferred_host_operations
    RESULT_CASE(VK_THREAD_DONE_KHR);
    // Provided by VK_KHR_deferred_host_operations
    RESULT_CASE(VK_OPERATION_DEFERRED_KHR);
    // Provided by VK_KHR_deferred_host_operations
    RESULT_CASE(VK_OPERATION_NOT_DEFERRED_KHR);
  default:
    return "Unknown VkResult Value";
  }
#undef RESULT_CASE
}

VkResult ivkCreateCommandPool(const struct VulkanFunctionTable* vt,
                              VkDevice device,
                              VkCommandPoolCreateFlags flags,
                              uint32_t queueFamilyIndex,
                              VkCommandPool* outCommandPool) {
  const VkCommandPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = NULL,
      .flags = flags,
      .queueFamilyIndex = queueFamilyIndex,
  };

  return vt->vkCreateCommandPool(device, &ci, NULL, outCommandPool);
}

VkResult ivkAllocateCommandBuffer(const struct VulkanFunctionTable* vt,
                                  VkDevice device,
                                  VkCommandPool commandPool,
                                  VkCommandBuffer* outCommandBuffer) {
  const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = NULL,
      .commandPool = commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  return vt->vkAllocateCommandBuffers(device, &ai, outCommandBuffer);
}

VkResult ivkAllocateMemory(const struct VulkanFunctionTable* vt,
                           VkPhysicalDevice physDev,
                           VkDevice device,
                           const VkMemoryRequirements* memRequirements,
                           VkMemoryPropertyFlags props,
                           bool enableBufferDeviceAddress,
                           VkDeviceMemory* outMemory,
                           VkMemoryPropertyFlags* outMemoryPropertyFlags) {
  assert(memRequirements);

  const VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = enableBufferDeviceAddress ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR : 0,
  };

  VkPhysicalDeviceMemoryProperties memProperties;
  vt->vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);

  const uint32_t memoryTypeIndex =
      ivkFindMemoryType(&memProperties, memRequirements->memoryTypeBits, props);
  if (outMemoryPropertyFlags) {
    *outMemoryPropertyFlags = memProperties.memoryTypes[memoryTypeIndex].propertyFlags;
  }

  const VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &memoryAllocateFlagsInfo,
      .allocationSize = memRequirements->size,
      .memoryTypeIndex = memoryTypeIndex,
  };

  return vt->vkAllocateMemory(device, &ai, NULL, outMemory);
}

VkResult ivkAllocateMemory2(const struct VulkanFunctionTable* vt,
                            VkPhysicalDevice physDev,
                            VkDevice device,
                            const VkMemoryRequirements2* memRequirements,
                            VkMemoryPropertyFlags props,
                            bool enableBufferDeviceAddress,
                            VkDeviceMemory* outMemory) {
  assert(memRequirements);

  const VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = enableBufferDeviceAddress ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR : 0,
  };

  VkPhysicalDeviceMemoryProperties memProperties;
  vt->vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);

  const VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &memoryAllocateFlagsInfo,
      .allocationSize = memRequirements->memoryRequirements.size,
      .memoryTypeIndex = ivkFindMemoryType(
          &memProperties, memRequirements->memoryRequirements.memoryTypeBits, props),
  };

  return vt->vkAllocateMemory(device, &ai, NULL, outMemory);
}

bool ivkIsHostVisibleSingleHeapMemory(const VkPhysicalDeviceMemoryProperties* memProps) {
  if (memProps->memoryHeapCount != 1) {
    return false;
  }

  const uint32_t flag = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  for (uint32_t i = 0; i < memProps->memoryTypeCount; i++) {
    if ((memProps->memoryTypes[i].propertyFlags & flag) == flag) {
      return true;
    }
  }

  return false;
}

uint32_t ivkFindMemoryType(const VkPhysicalDeviceMemoryProperties* memProps,
                           uint32_t memoryTypeBits,
                           VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < memProps->memoryTypeCount; i++) {
    const bool hasProperties = (memProps->memoryTypes[i].propertyFlags & flags) == flags;
    if ((memoryTypeBits & (1 << i)) && hasProperties) {
      return i;
    }
  }

  assert(false);

  return 0;
}

void ivkAddNext(void* node, const void* next) {
  if (!node || !next) {
    return;
  }

  VkBaseInStructure* cur = (VkBaseInStructure*)node;

  while (cur->pNext) {
    cur = (VkBaseInStructure*)cur->pNext;
  }

  cur->pNext = next;
}

VkResult ivkCreateDevice(const struct VulkanFunctionTable* vt,
                         VkPhysicalDevice physicalDevice,
                         size_t numQueueCreateInfos,
                         const VkDeviceQueueCreateInfo* queueCreateInfos,
                         size_t numDeviceExtensions,
                         const char** deviceExtensions,
                         const VkPhysicalDeviceFeatures2* enabledFeatures,
                         VkDevice* outDevice) {
  assert(numQueueCreateInfos >= 1);
  VkDeviceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = (uint32_t)numQueueCreateInfos,
      .pQueueCreateInfos = queueCreateInfos,
      .enabledExtensionCount = (uint32_t)numDeviceExtensions,
      .ppEnabledExtensionNames = deviceExtensions,
      .pEnabledFeatures = &enabledFeatures->features,
  };

  // Append all feature structs being requested for this device
  ci.pNext = enabledFeatures->pNext;

  return vt->vkCreateDevice(physicalDevice, &ci, NULL, outDevice);
}

#if !IGL_PLATFORM_MACCATALYST
#define VK_EXT_DEBUG_UTILS_SUPPORTED 1
#else
#define VK_EXT_DEBUG_UTILS_SUPPORTED 0
#endif

#if VK_EXT_DEBUG_UTILS_SUPPORTED
VkResult ivkCreateDebugUtilsMessenger(const struct VulkanFunctionTable* vt,
                                      VkInstance instance,
                                      PFN_vkDebugUtilsMessengerCallbackEXT callback,
                                      void* logUserData,
                                      VkDebugUtilsMessengerEXT* outMessenger) {
  // Some Android devices don't have the VK_EXT_debug_utils functions available, even though the
  // extension is supported and has been enabled
  if (vt->vkCreateDebugUtilsMessengerEXT == NULL) {
    return VK_SUCCESS;
  }

  const VkDebugUtilsMessengerCreateInfoEXT ci = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = callback,
      .pUserData = logUserData,
  };

  return vt->vkCreateDebugUtilsMessengerEXT(instance, &ci, NULL, outMessenger);
}
#else // VK_EXT_DEBUG_UTILS_SUPPORTED
// Stub version
VkResult ivkCreateDebugUtilsMessenger(const struct VulkanFunctionTable* vt,
                                      VkInstance instance,
                                      PFN_vkDebugUtilsMessengerCallbackEXT callback,
                                      void* logUserData,
                                      VkDebugUtilsMessengerEXT* outMessenger) {
  return VK_SUCCESS;
}
#endif // VK_EXT_DEBUG_UTILS_SUPPORTED

#if defined(VK_EXT_debug_report)
VkResult ivkCreateDebugReportMessenger(const struct VulkanFunctionTable* vt,
                                       VkInstance instance,
                                       PFN_vkDebugReportCallbackEXT callback,
                                       void* logUserData,
                                       VkDebugReportCallbackEXT* outMessenger) {
  VkDebugReportCallbackCreateInfoEXT ci = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
      .flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT |
               VK_DEBUG_REPORT_DEBUG_BIT_EXT,
      .pfnCallback = callback,
      .pUserData = logUserData,
  };

  return vt->vkCreateDebugReportCallbackEXT(instance, &ci, NULL, outMessenger);
}
#else // defined(VK_EXT_debug_report)
// Stub version
VkResult ivkCreateDebugReportMessenger(VkInstance instance,
                                       PFN_vkDebugReportCallbackEXT callback,
                                       void* logUserData,
                                       VkDebugReportCallbackEXT* outMessenger) {
  return VK_SUCCESS;
}
#endif // defined(VK_EXT_debug_report)

VkResult ivkCreateSurface(const struct VulkanFunctionTable* vt,
                          VkInstance instance,
                          void* window,
                          void* display,
                          void* layer,
                          VkSurfaceKHR* outSurface) {
#if !defined(VK_USE_PLATFORM_METAL_EXT)
  (void)layer;
#endif
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  const VkWin32SurfaceCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
      .hinstance = GetModuleHandle(NULL),
      .hwnd = (HWND)window,
  };
  return vt->vkCreateWin32SurfaceKHR(instance, &ci, NULL, outSurface);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
  const VkAndroidSurfaceCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
      .pNext = NULL,
      .flags = 0,
      .window = window,
  };
  return vt->vkCreateAndroidSurfaceKHR(instance, &ci, NULL, outSurface);
#elif defined(VK_USE_PLATFORM_METAL_EXT)
  const VkMetalSurfaceCreateInfoEXT ci = {
      .sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
      .pNext = NULL,
      .flags = 0,
      .pLayer = layer,
  };
  return vt->vkCreateMetalSurfaceEXT(instance, &ci, NULL, outSurface);
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
  const VkXlibSurfaceCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
      .pNext = NULL,
      .flags = 0,
      .dpy = (Display*)display,
      .window = (Window)window,
  };
  return vt->vkCreateXlibSurfaceKHR(instance, &ci, NULL, outSurface);
#else
  (void)instance;
  (void)window;
  (void)outSurface;
  // TODO: implement for other platforms
  return VK_NOT_READY;
#endif
}

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
                            VkSwapchainKHR* outSwapchain) {
  assert(caps);
  const bool isCompositeAlphaOpaqueSupported =
      (caps->supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0;
  const VkSwapchainCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = surface,
      .minImageCount = minImageCount,
      .imageFormat = surfaceFormat.format,
      .imageColorSpace = surfaceFormat.colorSpace,
      .imageExtent = {.width = width, .height = height},
      .imageArrayLayers = 1,
      .imageUsage = imageUsage,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &queueFamilyIndex,
      .preTransform = caps->currentTransform,
      .compositeAlpha = isCompositeAlphaOpaqueSupported ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                                        : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      .presentMode = presentMode,
      .clipped = VK_TRUE,
      .oldSwapchain = VK_NULL_HANDLE,
  };
  return vt->vkCreateSwapchainKHR(device, &ci, NULL, outSwapchain);
}

VkResult ivkCreateDescriptorSetLayout(const struct VulkanFunctionTable* vt,
                                      VkDevice device,
                                      VkDescriptorSetLayoutCreateFlags flags,
                                      uint32_t numBindings,
                                      const VkDescriptorSetLayoutBinding* bindings,
                                      const VkDescriptorBindingFlags* bindingFlags,
                                      VkDescriptorSetLayout* outLayout) {
  // Do not enable VK_EXT_descriptor_indexing for Android until
  // we fix the extension enumeration crash when the validation layer is enabled
#if !IGL_PLATFORM_ANDROID
  const VkDescriptorSetLayoutBindingFlagsCreateInfo setLayoutBindingFlagsCI = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT,
      .bindingCount = numBindings,
      .pBindingFlags = bindingFlags,
  };
#endif // !IGL_PLATFORM_ANDROID

  const VkDescriptorSetLayoutCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
#if !IGL_PLATFORM_ANDROID
      .pNext = bindingFlags ? &setLayoutBindingFlagsCI : NULL,
#endif
      .flags = flags,
      .bindingCount = numBindings,
      .pBindings = bindings,
  };
  return vt->vkCreateDescriptorSetLayout(device, &ci, NULL, outLayout);
}

VkResult ivkAllocateDescriptorSet(const struct VulkanFunctionTable* vt,
                                  VkDevice device,
                                  VkDescriptorPool pool,
                                  VkDescriptorSetLayout layout,
                                  VkDescriptorSet* outDescriptorSet) {
  const VkDescriptorSetAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &layout,
  };
  return vt->vkAllocateDescriptorSets(device, &ai, outDescriptorSet);
}

VkResult ivkCreateDescriptorPool(const struct VulkanFunctionTable* vt,
                                 VkDevice device,
                                 VkDescriptorPoolCreateFlags flags,
                                 uint32_t maxDescriptorSets,
                                 uint32_t numPoolSizes,
                                 const VkDescriptorPoolSize* poolSizes,
                                 VkDescriptorPool* outDescriptorPool) {
  const VkDescriptorPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = flags,
      .maxSets = maxDescriptorSets,
      .poolSizeCount = numPoolSizes,
      .pPoolSizes = poolSizes,
  };
  return vt->vkCreateDescriptorPool(device, &ci, NULL, outDescriptorPool);
}

VkWriteDescriptorSet ivkGetWriteDescriptorSetImageInfo(VkDescriptorSet dstSet,
                                                       uint32_t dstBinding,
                                                       VkDescriptorType descriptorType,
                                                       uint32_t numDescriptors,
                                                       const VkDescriptorImageInfo* pImageInfo) {
  const VkWriteDescriptorSet set = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = NULL,
      .dstSet = dstSet,
      .dstBinding = dstBinding,
      .dstArrayElement = 0,
      .descriptorCount = numDescriptors,
      .descriptorType = descriptorType,
      .pImageInfo = pImageInfo,
      .pBufferInfo = NULL,
      .pTexelBufferView = NULL,
  };
  return set;
}

VkWriteDescriptorSet ivkGetWriteDescriptorSetBufferInfo(VkDescriptorSet dstSet,
                                                        uint32_t dstBinding,
                                                        VkDescriptorType descriptorType,
                                                        uint32_t numDescriptors,
                                                        const VkDescriptorBufferInfo* pBufferInfo) {
  const VkWriteDescriptorSet set = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = NULL,
      .dstSet = dstSet,
      .dstBinding = dstBinding,
      .dstArrayElement = 0,
      .descriptorCount = numDescriptors,
      .descriptorType = descriptorType,
      .pImageInfo = NULL,
      .pBufferInfo = pBufferInfo,
      .pTexelBufferView = NULL,
  };
  return set;
}

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
                                   VkPipeline* outPipeline) {
  const VkGraphicsPipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = NULL,
      .flags = flags,
      .stageCount = numShaderStages,
      .pStages = shaderStages,
      .pVertexInputState = vertexInputState,
      .pInputAssemblyState = inputAssemblyState,
      .pTessellationState = tessellationState,
      .pViewportState = viewportState,
      .pRasterizationState = rasterizationState,
      .pMultisampleState = multisampleState,
      .pDepthStencilState = depthStencilState,
      .pColorBlendState = colorBlendState,
      .pDynamicState = dynamicState,
      .layout = pipelineLayout,
      .renderPass = renderPass,
      .subpass = subpass,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1,
  };
  return vt->vkCreateGraphicsPipelines(device, pipelineCache, 1, &ci, NULL, outPipeline);
}

void ivkImageMemoryBarrier(const struct VulkanFunctionTable* vt,
                           VkCommandBuffer buffer,
                           VkImage image,
                           VkAccessFlags srcAccessMask,
                           VkAccessFlags dstAccessMask,
                           VkImageLayout oldImageLayout,
                           VkImageLayout newImageLayout,
                           VkPipelineStageFlags srcStageMask,
                           VkPipelineStageFlags dstStageMask,
                           VkImageSubresourceRange subresourceRange) {
  const VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = srcAccessMask,
      .dstAccessMask = dstAccessMask,
      .oldLayout = oldImageLayout,
      .newLayout = newImageLayout,
      .image = image,
      .subresourceRange = subresourceRange,
  };
  vt->vkCmdPipelineBarrier(buffer, srcStageMask, dstStageMask, 0, 0, NULL, 0, NULL, 1, &barrier);
}

void ivkBufferBarrier(const struct VulkanFunctionTable* vt,
                      VkCommandBuffer cmdBuffer,
                      VkBuffer buffer,
                      VkBufferUsageFlags usageFlags,
                      VkPipelineStageFlags srcStageMask,
                      VkPipelineStageFlags dstStageMask) {
  VkBufferMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = 0,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  };

  // Each access bit must be supported by at least one stage in its
  // corresponding stage mask, otherwise validation flags
  // VUID-vkCmdPipelineBarrier-{p,Buffer,Image}MemoryBarriers-02818. In
  // particular, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT and VK_ACCESS_INDEX_READ_BIT
  // require VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, NOT just any stage that happens
  // to read the buffer.
  const VkPipelineStageFlags kShaderStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  if (srcStageMask & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) {
    barrier.srcAccessMask |= VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  }
  if (srcStageMask & VK_PIPELINE_STAGE_TRANSFER_BIT) {
    barrier.srcAccessMask |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  }
  if (srcStageMask & VK_PIPELINE_STAGE_HOST_BIT) {
    barrier.srcAccessMask |= VK_ACCESS_HOST_WRITE_BIT;
  }
  if (srcStageMask & kShaderStages) {
    barrier.srcAccessMask |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  }

  if (dstStageMask & VK_PIPELINE_STAGE_TRANSFER_BIT) {
    barrier.dstAccessMask |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  }
  if (dstStageMask & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) {
    barrier.dstAccessMask |= VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  }
  if (dstStageMask & VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT) {
    barrier.dstAccessMask |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  }
  if (dstStageMask & VK_PIPELINE_STAGE_VERTEX_INPUT_BIT) {
    if (usageFlags & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) {
      barrier.dstAccessMask |= VK_ACCESS_INDEX_READ_BIT;
    }
    if (usageFlags & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
      barrier.dstAccessMask |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    }
  }
  if (dstStageMask & kShaderStages) {
    barrier.dstAccessMask |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    if (usageFlags & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
      barrier.dstAccessMask |= VK_ACCESS_UNIFORM_READ_BIT;
    }
  }

  vt->vkCmdPipelineBarrier(cmdBuffer, srcStageMask, dstStageMask, 0, 0, NULL, 1, &barrier, 0, NULL);
}

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
                     VkFilter filter) {
  const VkImageBlit blit = {
      .srcSubresource = srcSubresourceRange,
      .srcOffsets = {srcOffsets[0], srcOffsets[1]},
      .dstSubresource = dstSubresourceRange,
      .dstOffsets = {dstOffsets[0], dstOffsets[1]},
  };
  vt->vkCmdBlitImage(buffer, srcImage, srcImageLayout, dstImage, dstImageLayout, 1, &blit, filter);
}

VkResult ivkSetDebugObjectName(const struct VulkanFunctionTable* vt,
                               VkDevice device,
                               VkObjectType type,
                               uint64_t handle,
                               const char* name) {
  // Some Android devices don't have the VK_EXT_debug_utils functions available, even though the
  // extension is supported and has been enabled
  if (!name || !*name || vt->vkSetDebugUtilsObjectNameEXT == NULL) {
    return VK_SUCCESS;
  }

#if VK_EXT_DEBUG_UTILS_SUPPORTED
  const VkDebugUtilsObjectNameInfoEXT ni = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .objectType = type,
      .objectHandle = handle,
      .pObjectName = name,
  };

  return vt->vkSetDebugUtilsObjectNameEXT(device, &ni);
#else
  return VK_SUCCESS;
#endif // VK_EXT_DEBUG_UTILS_SUPPORTED
}

void ivkCmdBeginDebugUtilsLabel(const struct VulkanFunctionTable* vt,
                                VkCommandBuffer buffer,
                                const char* name,
                                const float colorRGBA[4]) {
#if VK_EXT_DEBUG_UTILS_SUPPORTED
  // Some Android devices don't have the VK_EXT_debug_utils functions available, even though the
  // extension is supported and has been enabled
  if (!name || !*name || vt->vkCmdBeginDebugUtilsLabelEXT == NULL) {
    return;
  }

  const VkDebugUtilsLabelEXT label = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
      .pNext = NULL,
      .pLabelName = name,
      .color = {colorRGBA[0], colorRGBA[1], colorRGBA[2], colorRGBA[3]},
  };
  vt->vkCmdBeginDebugUtilsLabelEXT(buffer, &label);
#endif // VK_EXT_DEBUG_UTILS_SUPPORTED
}

void ivkCmdInsertDebugUtilsLabel(const struct VulkanFunctionTable* vt,
                                 VkCommandBuffer buffer,
                                 const char* name,
                                 const float colorRGBA[4]) {
#if VK_EXT_DEBUG_UTILS_SUPPORTED
  // Some Android devices don't have the VK_EXT_debug_utils functions available, even though the
  // extension is supported and has been enabled
  if (!name || !*name || vt->vkCmdInsertDebugUtilsLabelEXT == NULL) {
    return;
  }

  const VkDebugUtilsLabelEXT label = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
      .pNext = NULL,
      .pLabelName = name,
      .color = {colorRGBA[0], colorRGBA[1], colorRGBA[2], colorRGBA[3]},
  };
  vt->vkCmdInsertDebugUtilsLabelEXT(buffer, &label);
#endif // VK_EXT_DEBUG_UTILS_SUPPORTED
}

void ivkCmdEndDebugUtilsLabel(const struct VulkanFunctionTable* vt, VkCommandBuffer buffer) {
#if VK_EXT_DEBUG_UTILS_SUPPORTED
  // Some Android devices don't have the VK_EXT_debug_utils functions available, even though the
  // extension is supported and has been enabled
  if (vt->vkCmdEndDebugUtilsLabelEXT != NULL) {
    vt->vkCmdEndDebugUtilsLabelEXT(buffer);
  }
#endif // VK_EXT_DEBUG_UTILS_SUPPORTED
}

VkVertexInputBindingDescription ivkGetVertexInputBindingDescription(uint32_t binding,
                                                                    uint32_t stride,
                                                                    VkVertexInputRate inputRate) {
  const VkVertexInputBindingDescription desc = {
      .binding = binding,
      .stride = stride,
      .inputRate = inputRate,
  };
  return desc;
}

VkVertexInputAttributeDescription ivkGetVertexInputAttributeDescription(uint32_t location,
                                                                        uint32_t binding,
                                                                        VkFormat format,
                                                                        uint32_t offset) {
  const VkVertexInputAttributeDescription desc = {
      .location = location,
      .binding = binding,
      .format = format,
      .offset = offset,
  };
  return desc;
}

VkResult ivkVmaCreateAllocator(const struct VulkanFunctionTable* vt,
                               VkPhysicalDevice physDev,
                               VkDevice device,
                               VkInstance instance,
                               uint32_t apiVersion,
                               bool enableBufferDeviceAddress,
                               VkDeviceSize preferredLargeHeapBlockSize,
                               VmaAllocator* outVma) {
  const VmaVulkanFunctions funcs = {
      .vkGetInstanceProcAddr = vt->vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr = vt->vkGetDeviceProcAddr,
      .vkGetPhysicalDeviceProperties = vt->vkGetPhysicalDeviceProperties,
      .vkGetPhysicalDeviceMemoryProperties = vt->vkGetPhysicalDeviceMemoryProperties,
      .vkAllocateMemory = vt->vkAllocateMemory,
      .vkFreeMemory = vt->vkFreeMemory,
      .vkMapMemory = vt->vkMapMemory,
      .vkUnmapMemory = vt->vkUnmapMemory,
      .vkFlushMappedMemoryRanges = vt->vkFlushMappedMemoryRanges,
      .vkInvalidateMappedMemoryRanges = vt->vkInvalidateMappedMemoryRanges,
      .vkBindBufferMemory = vt->vkBindBufferMemory,
      .vkBindImageMemory = vt->vkBindImageMemory,
      .vkGetBufferMemoryRequirements = vt->vkGetBufferMemoryRequirements,
      .vkGetImageMemoryRequirements = vt->vkGetImageMemoryRequirements,
      .vkCreateBuffer = vt->vkCreateBuffer,
      .vkDestroyBuffer = vt->vkDestroyBuffer,
      .vkCreateImage = vt->vkCreateImage,
      .vkDestroyImage = vt->vkDestroyImage,
      .vkCmdCopyBuffer = vt->vkCmdCopyBuffer,

#if VMA_VULKAN_VERSION >= 1001000
      .vkGetBufferMemoryRequirements2KHR = vt->vkGetBufferMemoryRequirements2,
      .vkGetImageMemoryRequirements2KHR = vt->vkGetImageMemoryRequirements2,
      .vkBindBufferMemory2KHR = vt->vkBindBufferMemory2,
      .vkBindImageMemory2KHR = vt->vkBindImageMemory2,
      .vkGetPhysicalDeviceMemoryProperties2KHR = vt->vkGetPhysicalDeviceMemoryProperties2,
#endif

#if VMA_VULKAN_VERSION >= 1003000
      .vkGetDeviceBufferMemoryRequirements = vt->vkGetDeviceBufferMemoryRequirements,
      .vkGetDeviceImageMemoryRequirements = vt->vkGetDeviceImageMemoryRequirements,
#endif
  };

  const VmaAllocatorCreateInfo ci = {
      .flags = enableBufferDeviceAddress ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0,
      .physicalDevice = physDev,
      .device = device,
      .preferredLargeHeapBlockSize = preferredLargeHeapBlockSize,
      .pAllocationCallbacks = NULL,
      .pDeviceMemoryCallbacks = NULL,
      .pHeapSizeLimit = NULL,
      .pVulkanFunctions = &funcs,
      .instance = instance,
      .vulkanApiVersion = apiVersion,
  };
  return vmaCreateAllocator(&ci, outVma);
}

void ivkUpdateGlslangResource(glslang_resource_t* res,
                              const VkPhysicalDeviceProperties* props,
                              const VkPhysicalDeviceMeshShaderPropertiesEXT* meshShaderProps) {
  const VkPhysicalDeviceLimits* limits = props ? &props->limits : NULL;
  if (!limits || !res) {
    return;
  }

  res->max_vertex_attribs = (int)limits->maxVertexInputAttributes;
  res->max_clip_distances = (int)limits->maxClipDistances;
  res->max_compute_work_group_count_x = (int)limits->maxComputeWorkGroupCount[0];
  res->max_compute_work_group_count_y = (int)limits->maxComputeWorkGroupCount[1];
  res->max_compute_work_group_count_z = (int)limits->maxComputeWorkGroupCount[2];
  res->max_compute_work_group_size_x = (int)limits->maxComputeWorkGroupSize[0];
  res->max_compute_work_group_size_y = (int)limits->maxComputeWorkGroupSize[1];
  res->max_compute_work_group_size_z = (int)limits->maxComputeWorkGroupSize[2];
  res->max_vertex_output_components = (int)limits->maxVertexOutputComponents;
  res->max_geometry_input_components = (int)limits->maxGeometryInputComponents;
  res->max_geometry_output_components = (int)limits->maxGeometryOutputComponents;
  res->max_fragment_input_components = (int)limits->maxFragmentInputComponents;
  res->max_geometry_output_vertices = (int)limits->maxGeometryOutputVertices;
  res->max_geometry_total_output_components = (int)limits->maxGeometryTotalOutputComponents;
  res->max_tess_control_input_components =
      (int)limits->maxTessellationControlPerVertexInputComponents;
  res->max_tess_control_output_components =
      (int)limits->maxTessellationControlPerVertexOutputComponents;
  res->max_tess_evaluation_input_components = (int)limits->maxTessellationEvaluationInputComponents;
  res->max_tess_evaluation_output_components =
      (int)limits->maxTessellationEvaluationOutputComponents;
  res->max_viewports = (int)limits->maxViewports;
  res->max_cull_distances = (int)limits->maxCullDistances;
  res->max_combined_clip_and_cull_distances = (int)limits->maxCombinedClipAndCullDistances;

  if (meshShaderProps) {
    res->max_mesh_output_vertices_ext = meshShaderProps->maxMeshOutputVertices;
    res->max_mesh_output_primitives_ext = meshShaderProps->maxMeshOutputPrimitives;
    res->max_mesh_work_group_size_x_ext = meshShaderProps->maxMeshWorkGroupSize[0];
    res->max_mesh_work_group_size_y_ext = meshShaderProps->maxMeshWorkGroupSize[1];
    res->max_mesh_work_group_size_z_ext = meshShaderProps->maxMeshWorkGroupSize[2];
    res->max_task_work_group_size_x_ext = meshShaderProps->maxTaskWorkGroupSize[0];
    res->max_task_work_group_size_y_ext = meshShaderProps->maxTaskWorkGroupSize[1];
    res->max_task_work_group_size_z_ext = meshShaderProps->maxTaskWorkGroupSize[2];
    res->max_mesh_view_count_ext = meshShaderProps->maxMeshMultiviewViewCount;
  }
}
