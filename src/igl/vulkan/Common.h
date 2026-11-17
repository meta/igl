/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef IGL_VULKAN_COMMON_H
#define IGL_VULKAN_COMMON_H

#include <cassert> // IWYU pragma: export
#include <utility> // IWYU pragma: export

// set to 1 to see very verbose debug console logs with Vulkan commands
#define IGL_VULKAN_PRINT_COMMANDS 0

#include <igl/Macros.h> // IWYU pragma: export
#include <igl/Shader.h>
#include <igl/vulkan/VulkanFunctionTable.h> // IWYU pragma: export
#if IGL_PLATFORM_MACOSX
#include <vulkan/vulkan_metal.h> // IWYU pragma: export
#endif

#include <igl/ColorSpace.h> // IWYU pragma: export
#include <igl/Common.h> // IWYU pragma: export
#include <igl/DepthStencilState.h> // IWYU pragma: export
#include <igl/Format.h> // IWYU pragma: export
#include <igl/Texture.h> // IWYU pragma: export
#include <igl/VertexInputState.h> // IWYU pragma: export
#include <igl/vulkan/VulkanHelpers.h> // IWYU pragma: export

// Enable to use VulkanMemoryAllocator (VMA)
#define IGL_VULKAN_USE_VMA 1

// Macro that encapsulates a function call and check its return value against VK_SUCCESS. Prints
// location of failure when the result is not VK_SUCCESS, along with a stringified version of the
// result value. Asserts at the end of the code block. Expands to void function calls when build
// mode is not DEBUG
#define VK_ASSERT(func)                                            \
  {                                                                \
    const VkResult vk_assert_result = func;                        \
    if (vk_assert_result != VK_SUCCESS) {                          \
      IGL_DEBUG_ABORT("Vulkan API call failed: %s\n  %s\n",        \
                      #func,                                       \
                      ivkGetVulkanResultString(vk_assert_result)); \
    }                                                              \
  }

// Macro that encapsulates a function call and check its return value against VK_SUCCESS. Prints
// location of failure when the result is not VK_SUCCESS, along with a stringified version of the
// result value. Asserts at the end of the code block. The check remains even in build modes other
// than DEBUG
#if IGL_DEBUG_ABORT_ENABLED
// When IGL_DEBUG_ABORT_ENABLED is 1, VK_ASSERT may assert but will always log so use the existing
// macro.
#define VK_ASSERT_FORCE_LOG(func) VK_ASSERT(func)
#else
// When IGL_DEBUG_ABORT_ENABLED is 0, VK_ASSERT will neither log nor assert so need a separate
// definition that will explicitly log the result
#define VK_ASSERT_FORCE_LOG(func)                           \
  {                                                         \
    const VkResult vk_assert_result = func;                 \
    if (vk_assert_result != VK_SUCCESS) {                   \
      IGLLog(IGLLogError,                                   \
             "Vulkan API call failed: %s:%i\n  %s\n  %s\n", \
             __FILE__,                                      \
             __LINE__,                                      \
             #func,                                         \
             ivkGetVulkanResultString(vk_assert_result));   \
    }                                                       \
  }
#endif // IGL_DEBUG_ABORT_ENABLED

// Macro that encapsulates a function call and check its return value against VK_SUCCESS. Prints
// location of failure when the result is not VK_SUCCESS, along with a stringified version of the
// result value. Asserts at the end of the code block. The check remains even in build modes other
// than DEBUG. Returns the value passed as a parameter
#define VK_ASSERT_RETURN_VALUE(func, value)                        \
  {                                                                \
    const VkResult vk_assert_result = func;                        \
    if (vk_assert_result != VK_SUCCESS) {                          \
      IGL_DEBUG_ABORT("Vulkan API call failed: %s\n  %s\n",        \
                      #func,                                       \
                      ivkGetVulkanResultString(vk_assert_result)); \
      return value;                                                \
    }                                                              \
  }

// Calls the function provided as `func`, checks the return value from the function call against
// VK_SUCCESS and converts the return value from VkResult to an igl::Result and returns it
#define VK_ASSERT_RETURN(func) VK_ASSERT_RETURN_VALUE(func, getResultFromVkResult(vk_assert_result))

// Calls the function provided as `func`, checks the return value from the function call against
// VK_SUCCESS and returns VK_NULL_HANDLE
#define VK_ASSERT_RETURN_NULL_HANDLE(func) VK_ASSERT_RETURN_VALUE(func, VK_NULL_HANDLE)

#define IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx) (ctx)->ensureCurrentContextThread()

namespace igl::vulkan {

namespace util {
struct SpvModuleInfo;
} // namespace util

// The color definitions below are used by debugging utility functions, such as the ones provided by
// VK_EXT_debug_utils
#define K_COLOR_GENERATE_MIPMAPS igl::Color(1.f, 0.75f, 0.f)
#define K_COLOR_UPLOAD_IMAGE igl::Color(1.f, 0.2f, 0.78f)
#define K_COLOR_COMMAND_BUFFER_SUBMISSION_WITH_FENCE igl::Color(0.878f, 0.69f, 1.0f) // Mauve

// The VulkanContextConfig provides a way to override some of the default behaviors of the
// VulkanContext
struct VulkanContextConfig {
  bool terminateOnValidationError = false; // invoke std::terminate() on any validation error

  bool enableConcurrentVkDevicesSupport = false;

  bool enableValidation = true;
  bool enableGPUAssistedValidation = true;
  // Android validation remains disabled unless the caller explicitly opts in.
  bool enableValidationOnAndroid = false;
  bool enableExtraLogs = true;
  bool enableDescriptorIndexing = false;
  bool enableShaderInt16 = true;
  bool enableShaderDrawParameters = true;
  bool enableStorageBuffer16BitAccess = true;
  bool enableDualSrcBlend = true;
  bool enableGfxReconstruct = false;
  bool enableMultiviewPerViewViewports = false;
  bool enableDynamicVertexBufferStride = false;

  ColorSpace swapChainColorSpace = igl::ColorSpace::SRGBNonlinear;
  TextureFormat requestedSwapChainTextureFormat = igl::TextureFormat::RGBA_UNorm8;

  // the number of resources to support BufferAPIHintBits::Ring
  uint32_t maxResourceCount = 3u;

  // owned by the application - should be alive until initContext() returns
  const void* pipelineCacheData = nullptr;
  size_t pipelineCacheDataSize = 0;

  // Every submission on the IGL queues waits on a binary semaphore signaled by the previous
  // submission. When false, each command buffer begins with a full pipeline barrier instead, which
  // gives the same ordering on a single queue without the per-submission semaphores. Submissions
  // that carry injected wait semaphores keep chaining to the next submission either way.
  bool chainSubmitsWithSemaphores = true;

  // This enables fences generated at the end of submission to be exported to the client.
  // The client can then use the SubmitHandle to wait for the completion of the GPU work.
  bool exportableFences = false;

  // Use VK_EXT_headless_surface to create a headless swapchain
  bool headless = false;

  // Size for VulkanMemoryAllocator's default pool block size parameter.
  // Only relevant if VMA is used for memory allocation.
  // Passing 0 will prompt VMA to a large default value (currently 256 MB).
  // Using a smaller heap size would increase the chance of memory deallocation and result in less
  // memory wastage.
  size_t vmaPreferredLargeHeapBlockSize = 0;

  // Specifies a default fence timeout value.
  uint64_t fenceTimeoutNanoseconds = UINT64_MAX;

  size_t numExtraInstanceExtensions = 0;
  const char* IGL_NULLABLE* IGL_NULLABLE extraInstanceExtensions = nullptr;

  const char* engineName = "IGL/Vulkan";
  const char* applicationName = "IGL/Vulkan";
};

/**
 * @brief Encapsulates a handle to a VkSampler. The struct also stores the sampler id, which is used
 * for bindless rendering (see the ResourcesBinder and VulkanContext classes for more information)
 */
struct VulkanSampler final {
  VkSampler vkSampler = VK_NULL_HANDLE;
  /**
   * @brief The index into VulkanContext::samplers_. This index is intended to be used with bindless
   * rendering. Its value is set by the context when the resource is created and added to the vector
   * of samplers maintained by the VulkanContext.
   */
  uint32_t samplerId = 0;
};

static_assert(std::is_trivially_copyable_v<VulkanSampler>);

/// @brief The layout of the images that fragment shaders read as input attachments. It is the
/// layout of the input attachment references of a render pass and of the descriptors that read
/// them.
constexpr VkImageLayout kInputAttachmentImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

// The functions below are convenience functions used to convert to and from Vulkan values to IGL
// values

Result getResultFromVkResult(VkResult result);

inline Result checkVulkanApiVersion(uint32_t apiVersion) {
  if (apiVersion >= VK_API_VERSION_1_1) {
    return Result();
  }
  return Result(Result::Code::Unsupported,
                IGL_FORMAT("Vulkan 1.1 or newer is required (found {}.{}.{})",
                           VK_API_VERSION_MAJOR(apiVersion),
                           VK_API_VERSION_MINOR(apiVersion),
                           VK_API_VERSION_PATCH(apiVersion)));
}

void setResultFrom(Result* IGL_NULLABLE outResult, VkResult result);
VkFormat textureFormatToVkFormat(TextureFormat format);
TextureFormat vkFormatToTextureFormat(VkFormat format);
VkFormat invertRedAndBlue(VkFormat format);
bool isTextureFormatRGB(VkFormat format);
bool isTextureFormatBGR(VkFormat format);
bool isSrgbFormat(VkFormat format);
VkFormat srgbToUnorm(VkFormat format);
VkFormat unormToSrgb(VkFormat format);
bool hasDepth(VkFormat format);
bool hasStencil(VkFormat format);
uint32_t getNumImagePlanes(VkFormat format);
VkMemoryPropertyFlags resourceStorageToVkMemoryPropertyFlags(
    ResourceStorage resourceStorage,
    const VkPhysicalDeviceMemoryProperties* IGL_NULLABLE memProperties = nullptr);
VkCompareOp compareFunctionToVkCompareOp(CompareFunction func);
VkStencilOp stencilOperationToVkStencilOp(StencilOperation op);
VkSampleCountFlagBits getVulkanSampleCountFlags(size_t numSamples);
VkSurfaceFormatKHR colorSpaceToVkSurfaceFormat(ColorSpace colorSpace, bool isBGR);
uint32_t getVkLayer(TextureType type, uint32_t face, uint32_t layer);
TextureRangeDesc atVkLayer(TextureType type, const TextureRangeDesc& range, uint32_t vkLayer);
VkColorSpaceKHR colorSpaceToVkColorSpace(ColorSpace colorSpace);
ColorSpace vkColorSpaceToColorSpace(VkColorSpaceKHR colorSpace);
VkComponentMapping componentMappingToVkComponentMapping(const ComponentMapping& mapping);
/// @brief Build a `VkSpecializationInfo` from IGL function constants.
///
/// Fills `outEntries` with the Vulkan map entries derived from the IGL constants. The returned
/// `VkSpecializationInfo` has `pMapEntries` pointing into `outEntries` and `pData` pointing
/// directly into `constantValues.getData()` (zero-copy). `outEntries` and `constantValues`
/// must outlive any use of the returned info. When there are no constants the returned info
/// is zero-initialized (`mapEntryCount == 0`); pass `nullptr` to Vulkan in that case.
VkSpecializationInfo buildSpecializationInfo(const FunctionConstantValues& constantValues,
                                             std::vector<VkSpecializationMapEntry>& outEntries);

/// @brief Transition from the current layout to VK_IMAGE_LAYOUT_GENERAL
void transitionToGeneral(VkCommandBuffer cmdBuf, ITexture* texture);

/// @brief Transition from the current layout to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
void transitionToColorAttachment(VkCommandBuffer cmdBuf, ITexture* colorTex);

/// @brief Transition from the current layout to VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
void transitionToDepthStencilAttachment(VkCommandBuffer cmdBuf, ITexture* depthStencilTex);

/// @brief Transition from the current layout to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
void transitionToShaderReadOnly(VkCommandBuffer cmdBuf, ITexture* texture);

/// @brief Overrides the layout stored in the `texture` with the one in `layout`. This function does
/// not perform a transition, it only updates the texture's member variable that stores its current
/// layout, which a reusable command buffer whose recording fails then keeps
void overrideImageLayout(ITexture* texture, VkImageLayout layout);

/// @brief Checks that the input attachments and the storage images of a shader module follow the
/// conventions of IGL. A shader declares input attachment `i` once, as
/// `layout(input_attachment_index = i, set = kBindPoint_StorageImages, binding =
/// IGL_STORAGE_IMAGES_MAX + i) uniform subpassInput`, without arrays and multisampling. The storage
/// images in that set use the bindings below IGL_STORAGE_IMAGES_MAX. Returns
/// Result::Code::ArgumentInvalid with a message that names the offending variable and `debugName`
/// otherwise.
[[nodiscard]] Result validateShaderModuleBindings(const util::SpvModuleInfo& info,
                                                  const std::string& debugName);

/// @brief Ensures that all shader bindings are bound by checking the SPIR-V reflection. If the
/// function doesn't assert at some point, the shader bindings are correct. Only for debugging.
void ensureShaderModule(IShaderModule* sm);

/// @brief Implements the igl::IDepthStencilState interface
struct DepthStencilState final : public IDepthStencilState {
  explicit DepthStencilState(DepthStencilStateDesc desc) : desc(std::move(desc)) {}
  const DepthStencilStateDesc desc;
};

/// @brief Implements the igl::IVertexInputState interface
struct VertexInputState final : public IVertexInputState {
 public:
  explicit VertexInputState(VertexInputStateDesc desc) : desc(std::move(desc)) {}
  const VertexInputStateDesc desc;
};

} // namespace igl::vulkan

namespace igl::vulkan::functions {

void initialize(VulkanFunctionTable& table);
void loadInstanceFunctions(VulkanFunctionTable& table,
                           VkInstance instance,
                           bool enableExtDebugUtils);
void loadDeviceFunctions(VulkanFunctionTable& table, VkDevice device, uint32_t vulkanAPIVersion);

} // namespace igl::vulkan::functions

#endif // IGL_VULKAN_COMMON_H
