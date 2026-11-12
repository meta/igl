/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <deque>
#include <future>
#include <ldrutils/lutils/Pool.h>
#include <memory>
#include <unordered_map>
#include <igl/CommandEncoder.h>
#include <igl/HWDevice.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanFeatures.h>
#include <igl/vulkan/VulkanHelpers.h>
#include <igl/vulkan/VulkanImmediateCommands.h>
#include <igl/vulkan/VulkanQueuePool.h>
#include <igl/vulkan/VulkanRenderPassBuilder.h>
#include <igl/vulkan/VulkanStagingDevice.h>

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
struct AHardwareBuffer;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

#if defined(IGL_WITH_TRACY_GPU)
#include "tracy/TracyVulkan.hpp"
#endif

#define IGL_VULKAN_VALIDATION_LAYER_ERROR_SUMMARY 0

namespace igl {
class IRenderPipelineState;
}

namespace igl::vulkan {
namespace util {
struct SpvModuleInfo;
} // namespace util

class CommandQueue;
class ComputeCommandEncoder;
class RenderCommandEncoder;
class VulkanBuffer;
class VulkanDescriptorSetLayout;
class VulkanImage;
class VulkanImageView;
class VulkanSwapchain;
class VulkanTexture;

struct BindingsBuffers;
struct BindingsTextures;
struct BindingsStorageImages;
struct DescriptorPoolsArenas;
struct ImageLayoutJournal;
struct VulkanContextImpl;
struct VulkanImageCreateInfo;
struct VulkanImageViewCreateInfo;

/*
 * Descriptor sets:
 *  0 - combined image samplers
 *  1 - uniform/storage buffers
 *  2 - storage images and input attachments
 *  3 - bindless textures/samplers  <--  optional
 */
// NOLINTBEGIN(readability-identifier-naming)
enum {
  kBindPoint_CombinedImageSamplers = 0,
  kBindPoint_Buffers = 1,
  kBindPoint_StorageImages = 2,
  kBindPoint_Bindless = 3,
};
// NOLINTEND(readability-identifier-naming)

struct DeviceQueues {
  static constexpr uint32_t kInvalid = 0xFFFFFFFF;
  uint32_t graphicsQueueFamilyIndex = kInvalid;
  uint32_t computeQueueFamilyIndex = kInvalid;

  VkQueue IGL_NULLABLE graphicsQueue = VK_NULL_HANDLE;
  VkQueue IGL_NULLABLE computeQueue = VK_NULL_HANDLE;
};

struct DescriptorBuffer {
  std::shared_ptr<VulkanBuffer> buffer;
  size_t offset = 0;
  VulkanImmediateCommands::SubmitHandle handle = {};
  VkCommandBuffer bindCmdBuffer = VK_NULL_HANDLE;
};

class VulkanContext final {
 public:
  VulkanContext(VulkanContextConfig config,
                void* IGL_NULLABLE window,
                void* IGL_NULLABLE display = nullptr);
  ~VulkanContext();
  VulkanContext(const VulkanContext&) = delete;
  VulkanContext(VulkanContext&&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;
  VulkanContext& operator=(VulkanContext&&) = delete;

  Result queryDevices(const HWDeviceQueryDesc& desc, std::vector<HWDeviceDesc>& outDevices);
  Result initContext(const HWDeviceDesc& desc,
                     size_t numExtraDeviceExtensions = 0,
                     const char* IGL_NULLABLE* IGL_NULLABLE extraDeviceExtensions = nullptr,
                     const VulkanFeatures* IGL_NULLABLE requestedFeatures = nullptr,
                     const char* IGL_NULLABLE debugName = nullptr);

  Result initSwapchain(uint32_t width, uint32_t height);
  VkExtent2D getSwapchainExtent() const;

  VulkanImage createImage(VkImageType imageType,
                          VkExtent3D extent,
                          VkFormat format,
                          uint32_t mipLevels,
                          uint32_t arrayLayers,
                          VkImageTiling tiling,
                          VkImageUsageFlags usageFlags,
                          VkMemoryPropertyFlags memFlags,
                          VkImageCreateFlags flags,
                          VkSampleCountFlagBits samples,
                          Result* IGL_NULLABLE outResult,
                          const char* IGL_NULLABLE debugName = nullptr) const;

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
  std::unique_ptr<VulkanImage> createImageFromAndroidHardwareBuffer(
      AHardwareBuffer* androidHardwareBuffer,
      uint64_t memoryAllocationSize,
      VkImageType imageType,
      VkExtent3D extent,
      VkFormat format,
      uint32_t mipLevels,
      uint32_t arrayLayers,
      VkImageTiling tiling,
      VkImageUsageFlags usageFlags,
      VkImageCreateFlags flags,
      VkSampleCountFlagBits samples,
      igl::Result* IGL_NULLABLE outResult,
      const char* IGL_NULLABLE debugName = nullptr) const;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

  std::unique_ptr<VulkanImage> createImageFromFileDescriptor(
      int32_t fileDescriptor,
      uint64_t memoryAllocationSize,
      VkImageType imageType,
      VkExtent3D extent,
      VkFormat format,
      uint32_t mipLevels,
      uint32_t arrayLayers,
      VkImageTiling tiling,
      VkImageUsageFlags usageFlags,
      VkImageCreateFlags flags,
      VkSampleCountFlagBits samples,
      Result* IGL_NULLABLE outResult,
      const char* IGL_NULLABLE debugName = nullptr) const;
  std::unique_ptr<VulkanBuffer> createBuffer(VkDeviceSize bufferSize,
                                             VkBufferUsageFlags usageFlags,
                                             VkMemoryPropertyFlags memFlags,
                                             Result* IGL_NULLABLE outResult,
                                             const char* IGL_NULLABLE debugName = nullptr) const;
  std::shared_ptr<VulkanTexture> createTexture(VulkanImage&& image,
                                               VulkanImageView&& imageView,
                                               const char* IGL_NULLABLE debugName) const;
  std::shared_ptr<VulkanTexture> createTextureFromVkImage(
      VkImage vkImage,
      VulkanImageCreateInfo imageCreateInfo,
      VulkanImageViewCreateInfo imageViewCreateInfo,
      const char* IGL_NULLABLE debugName) const;

  SamplerHandle createSampler(const VkSamplerCreateInfo& ci,
                              VkFormat yuvVkFormat,
                              Result* IGL_NULLABLE outResult,
                              const char* IGL_NULLABLE debugName = nullptr) const;

  void createSurface(void* IGL_NULLABLE window, void* IGL_NULLABLE display);
  void createHeadlessSurface();

  bool hasSwapchain() const noexcept {
    return swapchain_ != nullptr;
  }

  Result waitIdle() const;

  /// @brief Returns the index of the current resource being used.
  ///        Its range is [0, config.maxResourceCount).
  [[nodiscard]] uint32_t currentSyncIndex() const noexcept {
    return syncCurrentIndex;
  }
  void syncAcquireNext() noexcept;
  /// @brief Called after every submit. Advances the sync index and waits for the submit that last
  /// used the new index, but only if a Ring buffer copy was read since the previous submit.
  void syncMarkSubmitted(VulkanImmediateCommands::SubmitHandle handle) noexcept;

  /// @brief Records that GPU work being encoded reads the current copy of a Ring buffer. Called by
  /// Buffer::getVkBuffer() and Buffer::gpuAddress(). The next submit is assumed to be the one that
  /// contains this work, i.e. one command buffer reading Ring buffers is encoded at a time.
  void markRingBufferRead() const noexcept {
    ringBufferReadSinceSubmit = true;
  }

  const VkPhysicalDeviceProperties& getVkPhysicalDeviceProperties() const {
    return vkPhysicalDeviceProperties2_.properties;
  }

  /// @brief Returns `VkQueueFamilyProperties::timestampValidBits` of the graphics queue family: the
  /// number of meaningful bits of the timestamps that its queues write, or 0 if they cannot write
  /// timestamps.
  [[nodiscard]] uint32_t getGraphicsQueueTimestampValidBits() const noexcept {
    return graphicsQueueTimestampValidBits_;
  }

  // Driver metadata used for vendor and driver-generation workaround gates.
  const VkPhysicalDeviceDriverPropertiesKHR& getVkPhysicalDeviceDriverProperties() const {
    return vkPhysicalDeviceDriverProperties_;
  }

  const VkPhysicalDeviceMeshShaderPropertiesEXT& getvkPhysicalDeviceMeshShaderPropertiesEXT()
      const {
    return vkPhysicalDeviceMeshShaderPropertiesEXT_;
  }

  VkFormat getClosestDepthStencilFormat(TextureFormat desiredFormat) const;

  struct RenderPassHandle {
    VkRenderPass pass = VK_NULL_HANDLE;
    uint8_t index = 0;
    // render passes that are compatible with each other (see
    // VulkanRenderPassBuilder::compatibilityKey()) share a compatIndex
    uint8_t compatIndex = 0;
    uint8_t numSubpasses = 1;
  };

  // render passes are owned and managed by the context. The returned handle has a null pass if the
  // render pass cannot be created, or if there are already 256 render passes or 256 classes of
  // compatible render passes.
  RenderPassHandle findRenderPass(const VulkanRenderPassBuilder& builder) const;
  RenderPassHandle getRenderPass(uint8_t index) const;

  // OpenXR needs Vulkan instance to find physical device
  VkInstance IGL_NULLABLE getVkInstance() const {
    return vkInstance_;
  }
  VkDevice IGL_NULLABLE getVkDevice() const {
    return vkDevice_;
  }
  VkPhysicalDevice IGL_NULLABLE getVkPhysicalDevice() const {
    return vkPhysicalDevice_;
  }
  VkDescriptorSetLayout getBindlessVkDescriptorSetLayout() const;
  VkDescriptorSet getBindlessVkDescriptorSet() const;

  std::vector<uint8_t> getPipelineCacheData() const;

  uint64_t getFrameNumber() const;

  using SubmitHandle = VulkanImmediateCommands::SubmitHandle;

  // execute a task some time in the future after the submit handle finished processing
  void deferredTask(std::packaged_task<void()>&& task, SubmitHandle handle = SubmitHandle()) const;

  /// @brief Creates descriptor pool arenas for a reusable command buffer. The descriptor sets that
  /// its recordings bind come from them instead of the context's arenas, which recycle the
  /// descriptor sets of every command buffer once it completes.
  [[nodiscard]] std::shared_ptr<DescriptorPoolsArenas> createDescriptorPoolsArenas() const;

  /// @brief Destroys `commandPool`, unless it is null, and the descriptor pools of `arenas`, which
  /// recordings of a reusable command buffer use, once `lastSubmission`, the last submission that
  /// runs one of them, completes: at once if it is empty or has completed.
  void destroyRecordings(VkCommandPool commandPool,
                         const std::vector<std::shared_ptr<DescriptorPoolsArenas>>& arenas,
                         SubmitHandle lastSubmission) const;

  /// @brief Keeps the layout of `image`, before a command recorded into `cmdBuf` changes it, in the
  /// ImageLayoutJournal of the reusable command buffer that records into `cmdBuf`, unless the
  /// journal holds a layout of `image` that no change elsewhere has replaced since, and marks it
  /// changed elsewhere in the other journals that hold it; a null `cmdBuf` stands for a change
  /// outside of IGL's command buffers
  void journalImageLayout(VkCommandBuffer cmdBuf, const VulkanImage& image) const;

  /// @brief Returns true if `cmdBuf` records a slot of a reusable command buffer, whose recording
  /// runs later, and as many times as the slot runs
  [[nodiscard]] bool isRecordingSlot(VkCommandBuffer cmdBuf) const;

  bool areValidationLayersEnabled() const;

  void* IGL_NULLABLE getVmaAllocator() const;

  VkSamplerYcbcrConversionInfo getOrCreateYcbcrConversionInfo(VkFormat format) const;

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
  // Get or create a context-owned conversion for external-format AHB imports.
  // The cache key includes externalFormat and every conversion-affecting field.
  // `info.format` must be VK_FORMAT_UNDEFINED and `info.pNext` must contain a non-zero
  // VkExternalFormatANDROID. Callers must serialize access and must not destroy the handle.
  VkSamplerYcbcrConversion getOrCreateExternalYcbcrConversion(
      const VkSamplerYcbcrConversionCreateInfo& info) const;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

  void freeResourcesForDescriptorSetLayout(VkDescriptorSetLayout dsl) const;

  // Returns a cached VkDescriptorSetLayout handle if an identical one has already been created
  // (identical bindings/flags/immutable samplers), otherwise creates a new one and caches it.
  // The returned handle is owned by the VulkanContext and destroyed when the context is destroyed.
  [[nodiscard]] VkDescriptorSetLayout getOrCreateVkDescriptorSetLayout(
      VkDescriptorSetLayoutCreateFlags flags,
      uint32_t numBindings,
      const VkDescriptorSetLayoutBinding* IGL_NULLABLE bindings,
      const VkDescriptorBindingFlags* IGL_NULLABLE bindingFlags,
      const char* IGL_NULLABLE debugName = nullptr) const;

  const VulkanFeatures& features() const noexcept;

  /// @brief Returns true if images can be backed by lazily allocated (tile) memory: the device has
  /// a lazily allocated memory type and VK_KHR_portability_subset is not enabled, because
  /// MoltenVK may split render passes, which defeats tile memory.
  [[nodiscard]] bool supportsLazilyAllocatedMemory() const;

  /// @brief Returns true if render passes can have several subpasses that read the color
  /// attachments of earlier subpasses as input attachments (DeviceFeatures::Subpasses): the device
  /// can create render passes with vkCreateRenderPass2(), binds the input attachments with
  /// descriptor sets rather than descriptor buffers, and VK_KHR_portability_subset is not enabled,
  /// because MoltenVK may split render passes, which defeats the purpose of subpasses.
  [[nodiscard]] bool supportsSubpasses() const;

  /// @brief Returns true if render passes with subpasses can render several views
  /// (DeviceFeatures::MultiviewSubpasses): the device supports subpasses and the multiview feature
  /// is enabled.
  [[nodiscard]] bool supportsMultiviewSubpasses() const;

  /// @brief Returns true if command buffers can be timed with a Timer (DeviceFeatures::Timers): the
  /// device can write timestamps with meaningful bits in the graphics queue and
  /// VK_KHR_portability_subset is not enabled, because MoltenVK resolves every timestamp of a
  /// command buffer to a single value taken when that command buffer completes, which always
  /// measures an elapsed time of zero.
  [[nodiscard]] bool supportsTimers() const;

  /// @brief Returns true if command buffers can be recorded once and replayed
  /// (PlatformDevice::supportsReusableCommandBuffers()): neither descriptor indexing nor
  /// VK_EXT_descriptor_buffer is enabled, because the context rewrites their descriptors while a
  /// recording could still bind them.
  [[nodiscard]] bool supportsReusableCommandBuffers() const;

  [[nodiscard]] const VkSurfaceCapabilitiesKHR& getSurfaceCapabilities() const noexcept {
    return deviceSurfaceCaps_;
  }

  void ensureCurrentContextThread() const;
  void setCurrentContextThread();

  /// @brief Returns true once a command submission or fence wait on one of this context's command
  /// queues returned VK_ERROR_DEVICE_LOST. Device loss is permanent. Safe to call from any thread.
  [[nodiscard]] bool isDeviceLost() const noexcept;

#if defined(IGL_WITH_TRACY_GPU)
  TracyVkCtx tracyCtx_ = nullptr;
  VkCommandPool profilingCommandPool_ = VK_NULL_HANDLE;
  VkCommandBuffer profilingCommandBuffer_ = VK_NULL_HANDLE;
#endif

  void waitDeferredTasks();

  /// @param handle The handle to the GPU Fence
  /// @return The Vulkan fence associated with the handle
  // @fb-only
  [[nodiscard]] VkFence getVkFenceFromSubmitHandle(igl::SubmitHandle handle) const noexcept;

  /// Android only for now - Creates the file descriptor for the underlying VkFence
  /// @param handle The handle to the GPU Fence
  /// @return The fd for the Vulkan Fence associated with the handleint
  // @fb-only
  [[nodiscard]] int getFenceFdFromSubmitHandle(igl::SubmitHandle handle) const noexcept;

  /// @brief Timeout to use for the descriptor-set fence wait in
  /// checkAndUpdateDescriptorSets().
  ///
  /// The wait is attempted once per pass, so re-arming the configured timeout every time
  /// would turn a bounded wait into an unbounded stall. The first attempt uses the configured
  /// value; once one has failed to succeed, later attempts poll instead of blocking, and still
  /// apply the pending update as soon as the fence signals.
  /// @param configuredTimeoutNs config_.fenceTimeoutNanoseconds
  /// @param previouslyDegraded whether an earlier attempt returned anything other than
  /// VK_SUCCESS
  [[nodiscard]] static uint64_t descriptorSetWaitTimeout(uint64_t configuredTimeoutNs,
                                                         bool previouslyDegraded) noexcept {
    return previouslyDegraded ? 0u : configuredTimeoutNs;
  }

 private:
  Result present() const;
  void createInstance();
  VkResult checkAndUpdateDescriptorSets();
  void pruneTextures();
  void querySurfaceCapabilities();
  void processDeferredTasks() const;
  /// @brief Returns `arenas`, the descriptor pool arenas of a reusable command buffer, or the
  /// context's arenas if it is null
  [[nodiscard]] DescriptorPoolsArenas& getDescriptorPoolsArenas(
      DescriptorPoolsArenas* IGL_NULLABLE arenas) const;
  void growBindlessDescriptorPool(uint32_t newMaxTextures, uint32_t newMaxSamplers);
  BindGroupTextureHandle createBindGroup(const BindGroupTextureDesc& desc,
                                         const IRenderPipelineState* IGL_NULLABLE
                                             compatiblePipeline,
                                         Result* IGL_NULLABLE outResult);
  BindGroupBufferHandle createBindGroup(const BindGroupBufferDesc& desc,
                                        Result* IGL_NULLABLE outResult);
  void destroy(BindGroupTextureHandle handle);
  void destroy(BindGroupBufferHandle handle);
  void destroy(SamplerHandle handle);
  void destroy(TextureHandle handle);
  VkDescriptorSet getBindGroupDescriptorSet(BindGroupTextureHandle handle) const;
  VkDescriptorSet getBindGroupDescriptorSet(BindGroupBufferHandle handle) const;
  uint32_t getBindGroupUsageMask(BindGroupTextureHandle handle) const;
  uint32_t getBindGroupUsageMask(BindGroupBufferHandle handle) const;

 private:
  friend class HWDevice;
  friend class Device;
  friend class VulkanStagingDevice;
  friend class VulkanSwapchain;
  friend class CommandQueue;
  friend class ComputeCommandEncoder;
  friend class RenderCommandEncoder;

  // should be kept on the heap, otherwise global Vulkan functions can cause arbitrary crashes.
  std::unique_ptr<VulkanFunctionTable> tableImpl_;

  VkInstance IGL_NULLABLE vkInstance_ = VK_NULL_HANDLE;
  bool loaderVersionSupported_ = false;
  VkDebugUtilsMessengerEXT vkDebugUtilsMessenger_ = VK_NULL_HANDLE;
  VkSurfaceKHR vkSurface_ = VK_NULL_HANDLE;
  VkPhysicalDevice IGL_NULLABLE vkPhysicalDevice_ = VK_NULL_HANDLE;
  // Provided by VK_EXT_descriptor_buffer
  VkPhysicalDeviceDescriptorBufferPropertiesEXT vkPhysicalDeviceDescriptorBufferProperties_{};
  VkPhysicalDeviceDescriptorIndexingPropertiesEXT vkPhysicalDeviceDescriptorIndexingProperties_{};
  // Provided by VK_KHR_driver_properties
  VkPhysicalDeviceDriverPropertiesKHR vkPhysicalDeviceDriverProperties_{};
  // Provided by VK_VERSION_1_1
  VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2_{};
  // Provided by VK_EXT_mesh_shader
  VkPhysicalDeviceMeshShaderPropertiesEXT vkPhysicalDeviceMeshShaderPropertiesEXT_{};

  uint32_t graphicsQueueTimestampValidBits_ = 0;

  std::vector<VkFormat> deviceDepthFormats_;
  std::vector<VkSurfaceFormatKHR> deviceSurfaceFormats_;
  VkSurfaceCapabilitiesKHR deviceSurfaceCaps_{};
  std::vector<VkPresentModeKHR> devicePresentModes_;

  VulkanFeatures features_;

 public:
  // NOLINTBEGIN(readability-identifier-naming)
  const VulkanFunctionTable& vf_;
  DeviceQueues deviceQueues_;
  VkDevice vkDevice_ = VK_NULL_HANDLE;
  std::unique_ptr<VulkanSwapchain> swapchain_;
  std::unique_ptr<VulkanSemaphore> timelineSemaphore_;
  std::unique_ptr<VulkanImmediateCommands> immediate_;
  std::unique_ptr<VulkanStagingDevice> stagingDevice_;

  std::unique_ptr<VulkanBuffer> dummyUniformBuffer_;
  std::unique_ptr<VulkanBuffer> dummyStorageBuffer_;
  // don't use staging on devices with device-local host-visible memory
  bool useStagingForBuffers_ = true;

  std::unique_ptr<VulkanContextImpl> pimpl_;

  // the journals of the reusable command buffers that record a slot now
  std::vector<ImageLayoutJournal*> imageLayoutJournals_;

  VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;

  mutable std::unordered_map<VkFormat, VkSamplerYcbcrConversionInfo> ycbcrConversionInfos_;

  // Context-owned cache for Android AHB external-format YCbCr conversions.
  struct ExternalYcbcrConversionEntry {
    uint64_t externalFormat = 0;
    VkSamplerYcbcrModelConversion ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY;
    VkSamplerYcbcrRange ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
    VkChromaLocation xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN;
    VkChromaLocation yChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN;
    VkComponentMapping components{}; // raw-plane swizzle from samplerYcbcrConversionComponents
    VkFilter chromaFilter = VK_FILTER_LINEAR;
    VkBool32 forceExplicitReconstruction = VK_FALSE;
    VkSamplerYcbcrConversion conversion = VK_NULL_HANDLE;
  };
  mutable std::vector<ExternalYcbcrConversionEntry> externalYcbcrConversions_;

  // 1. Textures can be safely deleted once they are not in use by GPU, hence our Vulkan context
  // owns all allocated textures (images+image views). The IGL interface vulkan::Texture does not
  // delete the underlying VulkanTexture but instead informs the context that it should be
  // deallocated. The context deallocates textures in a deferred way when it is safe to do so.
  // 2. Descriptor sets can be updated when they are not in use.
  mutable ldr::Pool<TextureTag, std::shared_ptr<VulkanTexture>> textures_;
  mutable ldr::Pool<SamplerTag, VulkanSampler> samplers_;
  // a texture/sampler was created since the last descriptor set update
  mutable bool awaitingCreation_ = false;
  // an earlier descriptor-set fence wait returned something other than VK_SUCCESS, so
  // subsequent attempts poll rather than blocking for the full fenceTimeoutNanoseconds
  bool descriptorSetWaitDegraded_ = false;

  mutable std::atomic<size_t> drawCallCount_{0};
  mutable std::atomic<size_t> shaderCompilationCount_{0};

  // Latched by isDeviceLost() once one of the command queues (VulkanImmediateCommands) reports a
  // loss.
  mutable std::atomic<bool> deviceLost_{false};

  // stores an index into renderPasses_
  mutable std::
      unordered_map<VulkanRenderPassBuilder, uint8_t, VulkanRenderPassBuilder::HashFunction>
          renderPassesHash_;
  // maps VulkanRenderPassBuilder::compatibilityKey() to RenderPassHandle::compatIndex
  mutable std::
      unordered_map<VulkanRenderPassBuilder, uint8_t, VulkanRenderPassBuilder::HashFunction>
          renderPassCompatHash_;
  mutable std::vector<RenderPassHandle> renderPasses_;

  VulkanContextConfig config_;

  // NOLINTEND(readability-identifier-naming)

  void updateBindingsTextures(VkCommandBuffer IGL_NONNULL cmdBuf,
                              VkPipelineLayout layout,
                              VkPipelineBindPoint bindPoint,
                              VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
                              DescriptorPoolsArenas* IGL_NULLABLE arenas,
                              const BindingsTextures& data,
                              const VulkanDescriptorSetLayout& dsl,
                              const util::SpvModuleInfo& info) const;
  void updateBindingsBuffers(VkCommandBuffer IGL_NONNULL cmdBuf,
                             VkPipelineLayout layout,
                             VkPipelineBindPoint bindPoint,
                             VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
                             DescriptorPoolsArenas* IGL_NULLABLE arenas,
                             BindingsBuffers& data,
                             const VulkanDescriptorSetLayout& dsl,
                             const util::SpvModuleInfo& info) const;
  void updateBindingsStorageImages(VkCommandBuffer IGL_NONNULL cmdBuf,
                                   VkPipelineLayout layout,
                                   VkPipelineBindPoint bindPoint,
                                   VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
                                   DescriptorPoolsArenas* IGL_NULLABLE arenas,
                                   const BindingsStorageImages& data,
                                   const VulkanDescriptorSetLayout& dsl,
                                   const util::SpvModuleInfo& info) const;
  void updateBindingsTexturesByDescriptorBuffer(
      VkCommandBuffer IGL_NONNULL cmdBuf,
      VkPipelineLayout layout,
      VkPipelineBindPoint bindPoint,
      VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
      const BindingsTextures& data,
      const VulkanDescriptorSetLayout& dsl,
      const util::SpvModuleInfo& info) const;
  void updateBindingsBuffersByDescriptorBuffer(
      VkCommandBuffer IGL_NONNULL cmdBuf,
      VkPipelineLayout layout,
      VkPipelineBindPoint bindPoint,
      VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
      BindingsBuffers& data,
      const VulkanDescriptorSetLayout& dsl,
      const util::SpvModuleInfo& info);
  void updateBindingsStorageImagesByDescriptorBuffer(
      VkCommandBuffer IGL_NONNULL cmdBuf,
      VkPipelineLayout layout,
      VkPipelineBindPoint bindPoint,
      VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
      const BindingsStorageImages& data,
      const VulkanDescriptorSetLayout& dsl,
      const util::SpvModuleInfo& info) const;

  struct DeferredTask {
    DeferredTask(std::packaged_task<void()>&& task, SubmitHandle handle) :
      task(std::move(task)), handle(handle) {}
    std::packaged_task<void()> task;
    SubmitHandle handle;
    uint64_t frameId = 0;
  };

  mutable std::deque<DeferredTask> deferredTasks;

  // sync resources
  uint32_t syncCurrentIndex = 0u;
  std::vector<SubmitHandle> syncSubmitHandles;
  mutable bool ringBufferReadSinceSubmit = false;

  VkPhysicalDeviceMemoryProperties memoryProperties{};

#if IGL_VULKAN_VALIDATION_LAYER_ERROR_SUMMARY
  std::unordered_map<std::string, uint32_t> validationErrorsSummary_;
#endif
};

} // namespace igl::vulkan
