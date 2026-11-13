/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// For vk_mem_alloc.h, define this before including VulkanContext.h in exactly
// one CPP file
#if defined(IGL_CMAKE_BUILD)
#define VMA_IMPLEMENTATION
#endif // IGL_CMAKE_BUILD

// For volk.h, define this before including volk.h in exactly one CPP file.
// @fb-only
#if defined(IGL_CMAKE_BUILD)
#define VOLK_IMPLEMENTATION
#endif // IGL_CMAKE_BUILD

#include <igl/SamplerState.h>
#include <igl/glslang/GlslCompiler.h>
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/RenderPipelineState.h>
#include <igl/vulkan/ResourcesBinder.h>
#include <igl/vulkan/SamplerState.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanDescriptorSetLayout.h>
#include <igl/vulkan/VulkanFeatures.h>
#include <igl/vulkan/VulkanImageView.h>
#include <igl/vulkan/VulkanPipelineBuilder.h>
#include <igl/vulkan/VulkanSwapchain.h>
#include <igl/vulkan/VulkanTexture.h>
#include <igl/vulkan/VulkanVma.h>
#include <igl/vulkan/util/SpvReflection.h>

#if IGL_PLATFORM_APPLE
#include <igl/vulkan/moltenvk/MoltenVkHelpers.h>
#endif

namespace {

[[maybe_unused]] const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";
const char* kGfxReconstructLayerName = "VK_LAYER_LUNARG_gfxreconstruct";

/*
 BINDLESS ONLY: these bindings should match GLSL declarations injected into shaders in
 Device::compileShaderModule(). Same with SparkSL.
 */
// NOLINTBEGIN(readability-identifier-naming)
const uint32_t kBinding_Texture2D = 0;
const uint32_t kBinding_Texture2DArray = 1;
const uint32_t kBinding_Texture3D = 2;
const uint32_t kBinding_TextureCube = 3;
const uint32_t kBinding_Sampler = 4;
const uint32_t kBinding_SamplerShadow = 5;
const uint32_t kBinding_StorageImages = 6;
// NOLINTEND(readability-identifier-naming)

#if !IGL_PLATFORM_APPLE
VKAPI_ATTR VkBool32 VKAPI_CALL
vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT msgSeverity,
                    VkDebugUtilsMessageTypeFlagsEXT msgType,
                    const VkDebugUtilsMessengerCallbackDataEXT* cbData,
                    void* userData) {
  if (msgSeverity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    return VK_FALSE;
  }

  const bool isError = (msgSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
  // Driver-emitted performance hints (e.g. Adreno's VKDBGUTILWARN###) are
  // suggestions, not Vulkan spec violations. Some drivers emit them at ERROR
  // severity; treat them as non-fatal regardless so they can't terminate the
  // process when terminateOnValidationError is enabled.
  const bool isPerformanceHint = (msgType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) != 0;

  auto* ctx = static_cast<igl::vulkan::VulkanContext*>(userData);

#if IGL_LOGGING_ENABLED
  std::array<char, 128> errorName = {};
  int object = 0;
  void* handle = nullptr;
  std::array<char, 128> typeName = {};
  void* messageID = nullptr;

  if (sscanf(cbData->pMessage,
             "Validation Error : [ %127s ] Object %i: handle = %p, type = %127s | MessageID = %p",
             errorName.data(),
             &object,
             &handle,
             typeName.data(),
             &messageID) >= 2) {
    const char* message = strrchr(cbData->pMessage, '|') + 1;
    IGL_LOG_INFO(
        "%sValidation layer:\n Validation Error: %s \n Object %i: handle = %p, type = %s\n "
        "MessageID = %p \n%s \n",
        isError ? "\nERROR:\n" : "",
        errorName.data(),
        object,
        handle,
        typeName.data(),
        messageID,
        message);
#if IGL_VULKAN_VALIDATION_LAYER_ERROR_SUMMARY
    ctx.validationErrorsSummary_[errorName.data()]++;
#endif
  } else {
    const bool isWarning = (msgSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0;

    if (isError || isWarning || ctx->config_.enableExtraLogs) {
      const char* prefix = isError ? (isPerformanceHint ? "\nPERFORMANCE:\n" : "\nERROR:\n") : "";
      const char* idName = cbData->pMessageIdName ? cbData->pMessageIdName : "<no id>";
      IGL_LOG_INFO("%sValidation layer:\n[%s : %d] %s\n",
                   prefix,
                   idName,
                   cbData->messageIdNumber,
                   cbData->pMessage);
    }
  }
#endif

  if (ctx->config_.terminateOnValidationError && !isPerformanceHint) {
    if (IGL_DEBUG_VERIFY_NOT(isError)) {
      std::terminate();
    }
  }

  return VK_FALSE;
}
#endif // !IGL_PLATFORM_ANDROID

std::vector<VkFormat> getCompatibleDepthStencilFormats(igl::TextureFormat format) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case igl::TextureFormat::Z_UNorm16:
    return {VK_FORMAT_D16_UNORM,
            VK_FORMAT_D16_UNORM_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
            VK_FORMAT_D32_SFLOAT};
  case igl::TextureFormat::Z_UNorm24:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM_S8_UINT};
  case igl::TextureFormat::Z_UNorm32:
    return {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
  case igl::TextureFormat::S8_UInt_Z24_UNorm:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT};
  case igl::TextureFormat::S8_UInt_Z32_UNorm:
    return {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT};
  case igl::TextureFormat::S_UInt8:
    return {VK_FORMAT_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    // default
  default:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT};
  }
}

bool validateImageLimits(VkImageType imageType,
                         VkSampleCountFlagBits samples,
                         const VkExtent3D& extent,
                         const VkPhysicalDeviceLimits& limits,
                         igl::Result* IGL_NULLABLE outResult) {
  using igl::Result;

  if (samples != VK_SAMPLE_COUNT_1_BIT && !IGL_DEBUG_VERIFY(imageType == VK_IMAGE_TYPE_2D)) {
    Result::setResult(
        outResult,
        Result(Result::Code::InvalidOperation, "Multisampling is supported only for 2D images"));
    return false;
  }

  if (imageType == VK_IMAGE_TYPE_1D &&
      !IGL_DEBUG_VERIFY(extent.width <= limits.maxImageDimension1D)) {
    Result::setResult(outResult,
                      Result(Result::Code::InvalidOperation, "1D texture size exceeded"));
    return false;
  } else if (imageType == VK_IMAGE_TYPE_2D &&
             !IGL_DEBUG_VERIFY(extent.width <= limits.maxImageDimension2D &&
                               extent.height <= limits.maxImageDimension2D)) {
    Result::setResult(outResult,
                      Result(Result::Code::InvalidOperation, "2D texture size exceeded"));
    return false;
  } else if (imageType == VK_IMAGE_TYPE_3D &&
             !IGL_DEBUG_VERIFY(extent.width <= limits.maxImageDimension3D &&
                               extent.height <= limits.maxImageDimension3D &&
                               extent.depth <= limits.maxImageDimension3D)) {
    Result::setResult(outResult,
                      Result(Result::Code::InvalidOperation, "3D texture size exceeded"));
    return false;
  }

  Result::setOk(outResult);

  return true;
}

} // namespace

namespace igl::vulkan {

static_assert(std::is_trivially_copyable_v<DeviceQueues>);

// @fb-only
class DescriptorPoolsArena final {
 public:
  DescriptorPoolsArena(const VulkanContext& ctx,
                       VkDescriptorType type,
                       VkDescriptorSetLayout dsl,
                       uint32_t numDescriptorsPerDSet,
                       const char* IGL_NULLABLE debugName) :
    ctx_(ctx),
    device_(ctx.getVkDevice()),
    numTypes_(1),
    types_{type},
    numDescriptorsPerDSet_{numDescriptorsPerDSet, 0},
    dsl_(dsl) {
    IGL_DEBUG_ASSERT(debugName);
    dpDebugName_ = IGL_FORMAT("Descriptor Pool: {}", debugName ? debugName : "");
  }
  DescriptorPoolsArena(const VulkanContext& ctx,
                       VkDescriptorType type0,
                       VkDescriptorType type1,
                       VkDescriptorSetLayout dsl,
                       uint32_t numDescriptorsPerDSet,
                       const char* debugName) :
    DescriptorPoolsArena(ctx,
                         type0,
                         type1,
                         dsl,
                         numDescriptorsPerDSet,
                         numDescriptorsPerDSet,
                         debugName) {}
  // `numDescriptorsPerDSet0` and `numDescriptorsPerDSet1` are the numbers of descriptors of `type0`
  // and `type1` in a descriptor set
  DescriptorPoolsArena(const VulkanContext& ctx,
                       VkDescriptorType type0,
                       VkDescriptorType type1,
                       VkDescriptorSetLayout dsl,
                       uint32_t numDescriptorsPerDSet0,
                       uint32_t numDescriptorsPerDSet1,
                       const char* debugName) :
    ctx_(ctx),
    device_(ctx.getVkDevice()),
    numTypes_(2),
    types_{type0, type1},
    numDescriptorsPerDSet_{numDescriptorsPerDSet0, numDescriptorsPerDSet1},
    dsl_(dsl) {
    IGL_DEBUG_ASSERT(debugName);
    dpDebugName_ = IGL_FORMAT("Descriptor Pool: {}", debugName ? debugName : "");
  }
  DescriptorPoolsArena(const DescriptorPoolsArena&) = delete;
  DescriptorPoolsArena& operator=(const DescriptorPoolsArena&) = delete;
  DescriptorPoolsArena(DescriptorPoolsArena&&) = delete;
  DescriptorPoolsArena& operator=(DescriptorPoolsArena&&) = delete;
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~DescriptorPoolsArena() {
    if (pool_ != VK_NULL_HANDLE) {
      extinct_.push_back({.pool = pool_, .handle = {}});
    }
    // releasePools() handed them over
    if (extinct_.empty()) {
      return;
    }
    ctx_.deferredTask(std::packaged_task<void()>(
        [extinct = std::move(extinct_), vf = ctx_.vf_, device = device_]() {
          for (const auto& p : extinct) {
            vf.vkDestroyDescriptorPool(device, p.pool, nullptr);
          }
        }));
  }
  [[nodiscard]] VkDescriptorSetLayout getVkDescriptorSetLayout() const {
    return dsl_;
  }
  // Moves the descriptor pools of the arena into `pools`, for their new owner to destroy them
  void releasePools(std::vector<VkDescriptorPool>& pools) {
    for (const ExtinctDescriptorPool& p : extinct_) {
      pools.push_back(p.pool);
    }
    extinct_.clear();
    if (pool_ != VK_NULL_HANDLE) {
      pools.push_back(pool_);
    }
    pool_ = VK_NULL_HANDLE;
    allocatedDSet_.clear();
    numRemainingDSetsInPool_ = 0;
  }
  // @fb-only
  [[nodiscard]] VkDescriptorSet getNextDescriptorSet(
      VulkanImmediateCommands& ic,
      VulkanImmediateCommands::SubmitHandle nextSubmitHandle) {
    IGL_DEBUG_ASSERT(!nextSubmitHandle.empty());

    VkDescriptorSet dset = VK_NULL_HANDLE;
    if (!numRemainingDSetsInPool_) {
      switchToNewDescriptorPool(ic, nextSubmitHandle);
    }
    if (isNewPool_) {
      VK_ASSERT(ivkAllocateDescriptorSet(&ctx_.vf_, device_, pool_, dsl_, &dset));
      allocatedDSet_.emplace_back(dset);
    } else {
      // Safe: allocatedDSet_.size() is always kNumDSetsPerPool because pools are only retired when
      // numRemainingDSetsInPool_ reaches 0, meaning all kNumDSetsPerPool sets were allocated.
      IGL_DEBUG_ASSERT(dsetCursor_ < allocatedDSet_.size());
      // @fb-only
      dset = allocatedDSet_[dsetCursor_++];
    }
    numRemainingDSetsInPool_--;
    return dset;
  }

 private:
  void switchToNewDescriptorPool(VulkanImmediateCommands& ic,
                                 VulkanImmediateCommands::SubmitHandle nextSubmitHandle) {
    numRemainingDSetsInPool_ = kNumDSetsPerPool;

    if (pool_ != VK_NULL_HANDLE) {
      extinct_.push_back(
          {.pool = pool_, .handle = nextSubmitHandle, .allocatedDSet = std::move(allocatedDSet_)});
    }
    // first, let's try to reuse the oldest extinct pool (never reuse pools that are tagged with the
    // same SubmitHandle because they have not yet been submitted)
    if (extinct_.size() > 1 && extinct_.front().handle != nextSubmitHandle) {
      ExtinctDescriptorPool& p = extinct_.front();
      if (ic.isReady(p.handle)) {
        pool_ = p.pool;
        allocatedDSet_ = std::move(p.allocatedDSet);
        dsetCursor_ = 0;
        isNewPool_ = false;
        extinct_.pop_front();
        return;
      }
    }
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    VkDescriptorPoolSize poolSizes[IGL_ARRAY_NUM_ELEMENTS(types_)];
    for (uint32_t i = 0; i != numTypes_; i++) {
      poolSizes[i] = VkDescriptorPoolSize{
          .type = types_[i],
          .descriptorCount =
              numDescriptorsPerDSet_[i] ? kNumDSetsPerPool * numDescriptorsPerDSet_[i] : 1u,
      };
    }
    VK_ASSERT(ivkCreateDescriptorPool(&ctx_.vf_,
                                      device_,
                                      VkDescriptorPoolCreateFlags{},
                                      kNumDSetsPerPool,
                                      numTypes_,
                                      poolSizes,
                                      &pool_));
    VK_ASSERT(ivkSetDebugObjectName(
        &ctx_.vf_, device_, VK_OBJECT_TYPE_DESCRIPTOR_POOL, (uint64_t)pool_, dpDebugName_.c_str()));
    allocatedDSet_.clear();
    dsetCursor_ = 0;
    isNewPool_ = true;
  }

 private:
  static constexpr uint32_t kNumDSetsPerPool = 64;

  const VulkanContext& ctx_;
  VkDevice device_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  uint32_t dsetCursor_ = 0;
  std::vector<VkDescriptorSet> allocatedDSet_;
  bool isNewPool_ = true; // is a new created pool or an old cached pool
  const uint32_t numTypes_ = 0;
  VkDescriptorType types_[2] = {VK_DESCRIPTOR_TYPE_MAX_ENUM, VK_DESCRIPTOR_TYPE_MAX_ENUM};
  // the number of descriptors of types_[i] in a descriptor set
  const std::array<uint32_t, 2> numDescriptorsPerDSet_ = {};
  uint32_t numRemainingDSetsInPool_ = 0;
  std::string dpDebugName_;

  VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE; // owned elsewhere

  struct ExtinctDescriptorPool {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VulkanImmediateCommands::SubmitHandle handle = {};
    std::vector<VkDescriptorSet> allocatedDSet;
  };

  std::deque<ExtinctDescriptorPool> extinct_;
};

namespace {

inline size_t alignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

uint32_t queryTimestampValidBits(const VulkanFunctionTable& vf,
                                 VkPhysicalDevice physicalDevice,
                                 uint32_t queueFamilyIndex) {
  uint32_t numFamilies = 0;
  vf.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numFamilies, nullptr);
  std::vector<VkQueueFamilyProperties> families(numFamilies);
  vf.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numFamilies, families.data());
  return queueFamilyIndex < numFamilies ? families[queueFamilyIndex].timestampValidBits : 0;
}

class DescriptorBuffersArena final {
 public:
  explicit DescriptorBuffersArena(const VulkanContext& ctx) : ctx_(ctx) {
    buffer_ = createNewBuffer();
  }

  /// Whether the first descriptor buffer was created.
  [[nodiscard]] bool valid() const {
    return buffer_.buffer != nullptr;
  }

  /// Returns null when a new descriptor buffer is needed and cannot be created. The arena is then
  /// untouched (the replacement is obtained before anything changes), so a later call retries.
  // @fb-only
  DescriptorBuffer* getDescriptorBuffer(size_t requireSize,
                                        size_t alignment,
                                        VulkanImmediateCommands& ic,
                                        VulkanImmediateCommands::SubmitHandle nextSubmitHandle) {
    IGL_DEBUG_ASSERT(requireSize && alignment);
    IGL_DEBUG_ASSERT(requireSize <= kBufferSize);
    const size_t alignedOffset = alignUp(buffer_.offset, alignment);
    if ((alignedOffset + requireSize) <= kBufferSize) {
      buffer_.offset = alignedOffset;
      return &buffer_;
    }

    // An earlier buffer whose submission has finished can be reused; otherwise a new one is made.
    bool canReuse = false;
    DescriptorBuffer replacement;
    if (!extinct_.empty() && extinct_.front().handle != nextSubmitHandle &&
        ic.isReady(extinct_.front().handle)) {
      replacement = extinct_.front();
      canReuse = true;
    } else {
      replacement = createNewBuffer();
    }
    if (replacement.buffer == nullptr) {
      return nullptr;
    }

    buffer_.handle = nextSubmitHandle;
    extinct_.push_back(buffer_);
    if (canReuse) {
      extinct_.pop_front();
      replacement.offset = 0;
      replacement.bindCmdBuffer = VK_NULL_HANDLE;
    }
    buffer_ = std::move(replacement);
    return &buffer_;
  }

 private:
  [[nodiscard]] DescriptorBuffer createNewBuffer() const {
    return DescriptorBuffer{
        .buffer = ctx_.createBuffer(kBufferSize,
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                        VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                                        VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    nullptr,
                                    "DescriptorBuffer")};
  }

 private:
  static constexpr uint32_t kBufferSize = 512 * 1024;
  const VulkanContext& ctx_;
  DescriptorBuffer buffer_;
  std::deque<DescriptorBuffer> extinct_;
};

struct BindGroupMetadataTextures {
  // cold
  BindGroupTextureDesc desc = {};
  VkDescriptorPool pool = VK_NULL_HANDLE;
  // hot
  VkDescriptorSet dset = VK_NULL_HANDLE;
  uint32_t usageMask = 0;
};

struct BindGroupMetadataBuffers {
  // cold
  BindGroupBufferDesc desc = {};
  VkDescriptorPool pool = VK_NULL_HANDLE;
  // hot
  VkDescriptorSet dset = VK_NULL_HANDLE;
  uint32_t usageMask = 0;
};

} // namespace

namespace {

// Cache key for VkDescriptorSetLayout de-duplication.
// Two VulkanDescriptorSetLayout constructions with identical bindings/flags will share the same
// underlying VkDescriptorSetLayout handle, which in turn ensures they share a single
// DescriptorPoolsArena entry (arenas are keyed by VkDescriptorSetLayout handle below).
struct DescriptorSetLayoutCacheKey {
  VkDescriptorSetLayoutCreateFlags flags = 0;
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  std::vector<VkDescriptorBindingFlags> bindingFlags;

  bool operator==(const DescriptorSetLayoutCacheKey& other) const noexcept {
    if (flags != other.flags) {
      return false;
    }
    if (bindings.size() != other.bindings.size()) {
      return false;
    }
    if (bindingFlags.size() != other.bindingFlags.size()) {
      return false;
    }
    for (size_t i = 0; i != bindings.size(); i++) {
      const auto& a = bindings[i];
      const auto& b = other.bindings[i];
      if (a.binding != b.binding || a.descriptorType != b.descriptorType ||
          a.descriptorCount != b.descriptorCount || a.stageFlags != b.stageFlags ||
          a.pImmutableSamplers != b.pImmutableSamplers) {
        return false;
      }
    }
    for (size_t i = 0; i != bindingFlags.size(); i++) {
      if (bindingFlags[i] != other.bindingFlags[i]) {
        return false;
      }
    }
    return true;
  }
};

// Boost-style hash combiner. Mixes `v` into the accumulated seed `h` with a golden-ratio
// constant plus a self-shift so that the seed's existing bits are re-distributed before each
// new field is folded in.
namespace {
inline void hashCombine(size_t& h, size_t v) noexcept {
  h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
}
} // namespace

struct DescriptorSetLayoutCacheKeyHash {
  size_t operator()(const DescriptorSetLayoutCacheKey& key) const noexcept {
    size_t h = std::hash<uint32_t>{}(static_cast<uint32_t>(key.flags));
    for (const auto& b : key.bindings) {
      hashCombine(h, std::hash<uint32_t>{}(b.binding));
      hashCombine(h, std::hash<uint32_t>{}(static_cast<uint32_t>(b.descriptorType)));
      hashCombine(h, std::hash<uint32_t>{}(b.descriptorCount));
      hashCombine(h, std::hash<uint32_t>{}(static_cast<uint32_t>(b.stageFlags)));
      // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
      hashCombine(h, std::hash<const void*>{}(static_cast<const void*>(b.pImmutableSamplers)));
    }
    for (const auto& f : key.bindingFlags) {
      hashCombine(h, std::hash<uint32_t>{}(static_cast<uint32_t>(f)));
    }
    return h;
  }
};

} // namespace

// The descriptor pool arenas of the descriptor set layouts. A reusable command buffer has arenas of
// its own, whose descriptor sets live as long as its recordings.
// @fb-only
struct DescriptorPoolsArenas final {
  // :)
  std::unordered_map<VkDescriptorSetLayout, std::unique_ptr<DescriptorPoolsArena>>
      arenaCombinedImageSamplers;
  std::unordered_map<VkDescriptorSetLayout, std::unique_ptr<DescriptorPoolsArena>> arenaBuffers;
  std::unordered_map<VkDescriptorSetLayout, std::unique_ptr<DescriptorPoolsArena>>
      arenaStorageImages;

  // Returns the descriptor pools of all the arenas, which no longer destroy them
  [[nodiscard]] std::vector<VkDescriptorPool> releasePools() {
    std::vector<VkDescriptorPool> pools;
    for (auto* arenas : {&arenaCombinedImageSamplers, &arenaBuffers, &arenaStorageImages}) {
      for (auto& entry : *arenas) {
        entry.second->releasePools(pools);
      }
    }
    return pools;
  }

  // NOLINTBEGIN(readability-identifier-naming)
  DescriptorPoolsArena& getOrCreateArena_CombinedImageSamplers(const VulkanContext& ctx,
                                                               VkDescriptorSetLayout dsl,
                                                               uint32_t numBindings)
  // NOLINTEND(readability-identifier-naming)
  {
    auto it = arenaCombinedImageSamplers.find(dsl);
    if (it != arenaCombinedImageSamplers.end()) {
      return *it->second;
    }
    arenaCombinedImageSamplers[dsl] =
        std::make_unique<DescriptorPoolsArena>(ctx,
                                               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                               dsl,
                                               numBindings,
                                               "arenaCombinedImageSamplers_");
    return *arenaCombinedImageSamplers[dsl].get();
  }
  // NOLINTBEGIN(readability-identifier-naming)
  DescriptorPoolsArena& getOrCreateArena_StorageImages(const VulkanContext& ctx,
                                                       VkDescriptorSetLayout dsl,
                                                       uint32_t numStorageImages,
                                                       uint32_t numInputAttachments)
  // NOLINTEND(readability-identifier-naming)
  {
    auto it = arenaStorageImages.find(dsl);
    if (it != arenaStorageImages.end()) {
      return *it->second;
    }
    if (numInputAttachments) {
      arenaStorageImages[dsl] =
          std::make_unique<DescriptorPoolsArena>(ctx,
                                                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                                                 dsl,
                                                 numStorageImages,
                                                 numInputAttachments,
                                                 "arenaStorageImagesAndInputAttachments_");
    } else {
      arenaStorageImages[dsl] = std::make_unique<DescriptorPoolsArena>(
          ctx, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, dsl, numStorageImages, "arenaStorageImages_");
    }
    return *arenaStorageImages[dsl].get();
  }
  // NOLINTBEGIN(readability-identifier-naming)
  DescriptorPoolsArena& getOrCreateArena_Buffers(const VulkanContext& ctx,
                                                 VkDescriptorSetLayout dsl,
                                                 uint32_t numBindings)
  // NOLINTEND(readability-identifier-naming)
  {
    auto it = arenaBuffers.find(dsl);
    if (it != arenaBuffers.end()) {
      return *it->second;
    }
    arenaBuffers[dsl] = std::make_unique<DescriptorPoolsArena>(ctx,
                                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                               dsl,
                                                               numBindings,
                                                               "arenaBuffers_");
    return *arenaBuffers[dsl].get();
  }
};

struct VulkanContextImpl final {
  std::thread::id contextThread = std::this_thread::get_id();

  // Vulkan Memory Allocator
  VmaAllocator vma = VK_NULL_HANDLE;
  // the arenas of all the command buffers that are not reusable
  DescriptorPoolsArenas arenas;
  // Cache of VkDescriptorSetLayout handles keyed by their descriptor bindings/flags. Since the
  // engine only uses a handful of distinct binding layouts, this collapses per-pipeline dsl
  // handles into a small set of shared handles, so that arenas above (keyed by handle) also
  // become shared automatically. Owned by the context; entries live until the context is
  // destroyed.
  std::unordered_map<DescriptorSetLayoutCacheKey,
                     VkDescriptorSetLayout,
                     DescriptorSetLayoutCacheKeyHash>
      dslCache;
  std::unique_ptr<VulkanDescriptorSetLayout> dslBindless; // everything
  std::unique_ptr<DescriptorBuffersArena> descriptorBuffersArena;
  VkDescriptorPool dpBindless = VK_NULL_HANDLE;
  VkDescriptorSet dsBindless = VK_NULL_HANDLE;
  uint32_t currentMaxBindlessTextures = 8;
  uint32_t currentMaxBindlessSamplers = 8;

  ldr::Pool<BindGroupBufferTag, BindGroupMetadataBuffers> bindGroupBuffersPool;
  ldr::Pool<BindGroupTextureTag, BindGroupMetadataTextures> bindGroupTexturesPool;

  SamplerHandle dummySampler = {};
  TextureHandle dummyTexture = {};
};

VulkanContext::VulkanContext(VulkanContextConfig config,
                             void* IGL_NULLABLE window,
                             void* IGL_NULLABLE display) :
  tableImpl_(std::make_unique<VulkanFunctionTable>()),
  // NOLINTBEGIN(clang-diagnostic-missing-designated-field-initializers)
  vkPhysicalDeviceDescriptorBufferProperties_({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT,
      .pNext = nullptr,
  }),
  vkPhysicalDeviceDescriptorIndexingProperties_({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT,
      .pNext = nullptr,
  }),
  vkPhysicalDeviceDriverProperties_({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR,
      .pNext = &vkPhysicalDeviceDescriptorIndexingProperties_,
  }),
  vkPhysicalDeviceProperties2_({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &vkPhysicalDeviceDriverProperties_,
  }),
  vkPhysicalDeviceMeshShaderPropertiesEXT_({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
      .pNext = nullptr,
  }),
  // NOLINTEND(clang-diagnostic-missing-designated-field-initializers)
  features_(config),
  vf_(*tableImpl_),
  config_(config) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_PROFILER_THREAD("MainThread");

  pimpl_ = std::make_unique<VulkanContextImpl>();

#if defined(IGL_CMAKE_BUILD)
  const auto result = volkInitialize();

  // Do not remove for backward compatibility with projects using global functions.
  if (result != VK_SUCCESS) {
    IGL_LOG_ERROR("volkInitialize() failed with error code %d\n", static_cast<int>(result));
    abort();
  };
#endif // IGL_CMAKE_BUILD
  vulkan::functions::initialize(*tableImpl_);

  glslang::initializeCompiler();

  createInstance();

  if (!loaderVersionSupported_) {
    return;
  }

  if (config_.headless) {
    IGL_DEBUG_ASSERT(features_.has_VK_EXT_headless_surface,
                     "VK_EXT_headless_surface extension is not supported");
    createHeadlessSurface();
  } else if (window || display) {
    createSurface(window, display);
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
VulkanContext::~VulkanContext() {
  IGL_PROFILER_FUNCTION();

  if (vkDevice_) {
    waitIdle();
  }

#if defined(IGL_WITH_TRACY_GPU)
  if (tracyCtx_) {
    TracyVkDestroy(tracyCtx_);
    vf_.vkDestroyCommandPool(vkDevice_, profilingCommandPool_, nullptr);
  }
#endif

  dummyStorageBuffer_.reset();
  dummyUniformBuffer_.reset();
  pimpl_->descriptorBuffersArena.reset(nullptr);

#if IGL_DEBUG_ABORT_ENABLED
  for (const auto& t : pimpl_->bindGroupTexturesPool.objects_) {
    if (t.dset != VK_NULL_HANDLE) {
      IGL_DEBUG_ABORT("Leaked texture bind group detected! %s", t.desc.debugName.c_str());
    }
  }
  for (const auto& t : pimpl_->bindGroupBuffersPool.objects_) {
    if (t.dset != VK_NULL_HANDLE) {
      IGL_DEBUG_ABORT("Leaked buffer bind group detected! %s", t.desc.debugName.c_str());
    }
  }
#endif // IGL_DEBUG_ABORT_ENABLED

  // BindGroups can hold shared pointers to textures/samplers/buffers. Release them here.
  pimpl_->bindGroupTexturesPool.clear();
  pimpl_->bindGroupBuffersPool.clear();

  destroy(pimpl_->dummySampler);
  destroy(pimpl_->dummyTexture);

  pruneTextures();

#if IGL_LOGGING_ENABLED
  if (textures_.numObjects()) {
    IGL_LOG_ERROR("Leaked %u textures\n", textures_.numObjects());
  }
  if (samplers_.numObjects()) {
    IGL_LOG_ERROR("Leaked %u samplers\n", samplers_.numObjects());
  }
#endif // IGL_LOGGING_ENABLED
  textures_.clear();
  samplers_.clear();

  // This will free an internal buffer that was allocated by VMA
  stagingDevice_.reset(nullptr);

  if (vkDevice_) {
    for (const RenderPassHandle& r : renderPasses_) {
      vf_.vkDestroyRenderPass(vkDevice_, r.pass, nullptr);
    }
  }

  pimpl_->dslBindless.reset(nullptr);

  swapchain_.reset(nullptr); // Swapchain has to be destroyed prior to Surface

  pimpl_->arenas.arenaCombinedImageSamplers.clear();
  pimpl_->arenas.arenaStorageImages.clear();
  pimpl_->arenas.arenaBuffers.clear();

  // Destroy all cached VkDescriptorSetLayout handles owned by the context.
  // VulkanDescriptorSetLayout instances only reference these handles through the cache and no
  // longer own them.
  if (vkDevice_) {
    for (auto& kv : pimpl_->dslCache) {
      if (kv.second != VK_NULL_HANDLE) {
        vf_.vkDestroyDescriptorSetLayout(vkDevice_, kv.second, nullptr);
      }
    }
  }
  pimpl_->dslCache.clear();

  waitDeferredTasks();

  immediate_.reset(nullptr);
  timelineSemaphore_.reset(nullptr);

  if (vkDevice_) {
    if (pimpl_->dpBindless != VK_NULL_HANDLE) {
      vf_.vkDestroyDescriptorPool(vkDevice_, pimpl_->dpBindless, nullptr);
    }
    for (auto& p : ycbcrConversionInfos_) {
      if (p.second.conversion != VK_NULL_HANDLE) {
        vf_.vkDestroySamplerYcbcrConversion(vkDevice_, p.second.conversion, nullptr);
      }
    }
    for (const auto& entry : externalYcbcrConversions_) {
      if (entry.conversion != VK_NULL_HANDLE) {
        vf_.vkDestroySamplerYcbcrConversion(vkDevice_, entry.conversion, nullptr);
      }
    }
    vf_.vkDestroyPipelineCache(vkDevice_, pipelineCache_, nullptr);
  }

  if (vkSurface_ != VK_NULL_HANDLE) {
    vf_.vkDestroySurfaceKHR(vkInstance_, vkSurface_, nullptr);
  }

  // Clean up VMA
  if (IGL_VULKAN_USE_VMA) {
    vmaDestroyAllocator(pimpl_->vma);
  }

  if (vkDevice_) {
    vf_.vkDestroyDevice(vkDevice_, nullptr); // Device has to be destroyed prior to Instance
  }
#if !IGL_PLATFORM_APPLE
  if (vf_.vkDestroyDebugUtilsMessengerEXT) {
    vf_.vkDestroyDebugUtilsMessengerEXT(vkInstance_, vkDebugUtilsMessenger_, nullptr);
  }
#endif // !IGL_PLATFORM_APPLE
  if (vf_.vkDestroyInstance) {
    vf_.vkDestroyInstance(vkInstance_, nullptr);
  }

  glslang::finalizeCompiler();

#if IGL_LOGGING_ENABLED
  if (config_.enableExtraLogs) {
    IGL_LOG_INFO("Vulkan graphics pipelines created: %u\n",
                 VulkanPipelineBuilder::getNumPipelinesCreated());
    IGL_LOG_INFO("Vulkan compute pipelines created: %u\n",
                 VulkanComputePipelineBuilder::getNumPipelinesCreated());
  }
#endif // IGL_LOGGING_ENABLED

#if defined(IGL_CMAKE_BUILD)
  volkFinalize();
#endif

#if IGL_VULKAN_VALIDATION_LAYER_ERROR_SUMMARY
  if (!validationErrorsSummary_.empty()) {
    IGL_LOG_INFO("Vulkan Validation Layer errors found: %u\n", validationErrorsSummary_.size());
    for (const auto& error : validationErrorsSummary_) {
      IGL_LOG_ERROR("\t%s: %u\n", error.first.c_str(), error.second);
    }
  }
#endif
}

void VulkanContext::createInstance() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_DEBUG_ASSERT(vkInstance_ == VK_NULL_HANDLE, "createInstance() is not reentrant");

  uint32_t instanceApiVersion = VK_API_VERSION_1_0;
  if (vf_.vkEnumerateInstanceVersion) {
    VK_ASSERT(vf_.vkEnumerateInstanceVersion(&instanceApiVersion));
  }
  const Result versionResult = checkVulkanApiVersion(instanceApiVersion);
  loaderVersionSupported_ = versionResult.isOk();
  if (!loaderVersionSupported_) {
    IGL_LOG_ERROR("%s\n", versionResult.message.c_str());
    return;
  }

  // Enumerate all instance extensions
  features_.enumerate(vf_);
  // NOLINTBEGIN(readability-identifier-naming)
  features_.enableCommonInstanceExtensions(config_);
  for (size_t index = 0; index < config_.numExtraInstanceExtensions; ++index) {
    features_.enable(config_.extraInstanceExtensions[index],
                     VulkanFeatures::ExtensionType::Instance);
  }
  // NOLINTEND(readability-identifier-naming)
  const auto instanceExtensions = features_.allEnabled(VulkanFeatures::ExtensionType::Instance);

  // Enumerate available instance layers before requesting them
  uint32_t layerCount = 0;
  vf_.vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vf_.vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  IGL_LOG_INFO("Found %u Vulkan instance layers\n", layerCount);
  for ([[maybe_unused]] const auto& layer : availableLayers) {
    IGL_LOG_INFO("\t%s - %u.%u.%u.%u, %u\n",
                 layer.layerName,
                 VK_API_VERSION_MAJOR(layer.specVersion),
                 VK_API_VERSION_MINOR(layer.specVersion),
                 VK_API_VERSION_VARIANT(layer.specVersion),
                 VK_API_VERSION_PATCH(layer.specVersion),
                 layer.implementationVersion);
  }

  std::vector<const char*> layers;
  // @fb-only
#if !IGL_PLATFORM_MACOSX
#if IGL_PLATFORM_ANDROID
  if (!config_.enableValidationOnAndroid) {
    config_.enableValidation = false;
  }
#endif
  if (config_.enableValidation) {
    const auto hasLayer = [&availableLayers](const char* name) {
      return std::any_of(availableLayers.begin(), availableLayers.end(), [name](const auto& layer) {
        return strcmp(layer.layerName, name) == 0;
      });
    };
    if (hasLayer(kValidationLayerName)) {
      layers.emplace_back(kValidationLayerName);
    } else {
      IGL_LOG_INFO("Validation layers requested but %s not available — skipping\n",
                   kValidationLayerName);
      config_.enableValidation = false;
    }
  }
#else
  // No layer is ever requested here, so areValidationLayersEnabled() must report false rather than
  // what the caller asked for. Must stay below enableCommonInstanceExtensions(), which reads the
  // requested value to decide on VK_EXT_validation_features.
  config_.enableValidation = false;
#endif
  if (config_.enableGfxReconstruct) {
    layers.emplace_back(kGfxReconstructLayerName);
  }

#if !IGL_PLATFORM_MACOSX
  std::vector<VkValidationFeatureEnableEXT> valFeatures;
  if (config_.enableGPUAssistedValidation) {
    valFeatures.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
  }
  const VkValidationFeaturesEXT features = {
      .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
      .enabledValidationFeatureCount = static_cast<uint32_t>(valFeatures.size()),
      .pEnabledValidationFeatures = valFeatures.empty() ? nullptr : valFeatures.data(),
  };
#endif // !IGL_PLATFORM_MACOSX

  // Request the highest API version the loader supports (>= 1.3 on modern loaders). We use core
  // entry points that were promoted in Vulkan 1.3 (the dynamic-state setters vkCmdSetCullMode(),
  // vkCmdSetDepthTestEnable(), etc.); if the app declares a lower apiVersion than those functions'
  // core version, the loader returns null for them from vkGetDeviceProcAddr() even on a 1.3+
  // device, causing a null-pointer crash at draw time.
  const VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = config_.applicationName,
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = config_.engineName,
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = instanceApiVersion,
  };

  const VkInstanceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
#if !IGL_PLATFORM_MACOSX
      .pNext = config_.enableValidation ? &features : nullptr,
#endif
      .flags = features_.has_VK_KHR_portability_enumeration
                   ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
                   : static_cast<VkInstanceCreateFlags>(0),
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(layers.size()),
      .ppEnabledLayerNames = !layers.empty() ? layers.data() : nullptr,
      .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
      .ppEnabledExtensionNames = instanceExtensions.data(),
  };

  const VkResult result = vf_.vkCreateInstance(&ci, nullptr, &vkInstance_);

  VK_ASSERT(result);

#if defined(IGL_CMAKE_BUILD)
  // Do not remove for backward compatibility with projects using global functions.
  volkLoadInstance(vkInstance_);
#endif
  const bool enableExtDebugUtils =
      features_.enable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VulkanFeatures::ExtensionType::Instance);
  vulkan::functions::loadInstanceFunctions(*tableImpl_, vkInstance_, enableExtDebugUtils);

#if !IGL_PLATFORM_APPLE
  if (features_.enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
    VK_ASSERT(ivkCreateDebugUtilsMessenger(
        &vf_, vkInstance_, &vulkanDebugCallback, this, &vkDebugUtilsMessenger_));
  }
#endif // !IGL_PLATFORM_APPLE

#if IGL_LOGGING_ENABLED
  if (config_.enableExtraLogs) {
    // log available instance extensions
    IGL_LOG_INFO("Vulkan instance extensions:\n");
    for (const auto& extension :
         features_.allAvailableExtensions(VulkanFeatures::ExtensionType::Instance)) {
      IGL_LOG_INFO("  %s\n", extension.c_str());
    }
  }
#endif
}

void VulkanContext::createHeadlessSurface() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  const VkHeadlessSurfaceCreateInfoEXT ci = {
      .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
      .pNext = nullptr,
      .flags = 0,
  };

  VK_ASSERT(vf_.vkCreateHeadlessSurfaceEXT(vkInstance_, &ci, nullptr, &vkSurface_));
}

void VulkanContext::createSurface(void* IGL_NULLABLE window, void* IGL_NULLABLE display) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  [[maybe_unused]] void* layer = nullptr;
#if IGL_PLATFORM_APPLE
  layer = igl::vulkan::getCAMetalLayer(window);
#endif
  VK_ASSERT(ivkCreateSurface(&vf_, vkInstance_, window, display, layer, &vkSurface_));
}

Result VulkanContext::queryDevices(const HWDeviceQueryDesc& desc,
                                   std::vector<HWDeviceDesc>& outDevices) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  outDevices.clear();

  // Physical devices
  uint32_t deviceCount = 0;

  if (!vf_.vkEnumeratePhysicalDevices) {
    return Result(Result::Code::Unsupported, "Vulkan functions are not loaded");
  }

  VK_ASSERT_RETURN(vf_.vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, nullptr));
  std::vector<VkPhysicalDevice> vkDevices(deviceCount);
  VK_ASSERT_RETURN(vf_.vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, vkDevices.data()));

  auto convertVulkanDeviceTypeToIGL = [](VkPhysicalDeviceType vkDeviceType) -> HWDeviceType {
    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (vkDeviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return HWDeviceType::IntegratedGpu;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return HWDeviceType::DiscreteGpu;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return HWDeviceType::ExternalGpu;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return HWDeviceType::SoftwareGpu;
    default:
      return HWDeviceType::Unknown;
    }
  };

  const HWDeviceType desiredDeviceType = desc.hardwareType;
  uint32_t highestSkippedVersion = 0;
  bool skippedLegacyDevice = false;

  for (uint32_t i = 0; i < deviceCount; ++i) {
    VkPhysicalDevice physicalDevice = vkDevices[i];
    VkPhysicalDeviceProperties deviceProperties{};
    vf_.vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

    const HWDeviceType deviceType = convertVulkanDeviceTypeToIGL(deviceProperties.deviceType);

    // filter non-suitable hardware devices
    if (desiredDeviceType != HWDeviceType::Unknown && deviceType != desiredDeviceType) {
      continue;
    }

    if (!checkVulkanApiVersion(deviceProperties.apiVersion).isOk()) {
      highestSkippedVersion = std::max(highestSkippedVersion, deviceProperties.apiVersion);
      skippedLegacyDevice = true;
      continue;
    }

    outDevices.emplace_back(reinterpret_cast<uintptr_t>(vkDevices[i]),
                            deviceType,
                            deviceProperties.vendorID,
                            deviceProperties.deviceName,
                            std::to_string(deviceProperties.vendorID));
  }

  if (outDevices.empty()) {
    if (skippedLegacyDevice) {
      return checkVulkanApiVersion(highestSkippedVersion);
    }
    return Result(Result::Code::Unsupported, "No Vulkan devices matching your criteria");
  }

  return Result();
}

Result VulkanContext::initContext(const HWDeviceDesc& desc,
                                  size_t numExtraDeviceExtensions,
                                  const char* IGL_NULLABLE* IGL_NULLABLE extraDeviceExtensions,
                                  const VulkanFeatures* IGL_NULLABLE requestedFeatures,
                                  const char* IGL_NULLABLE debugName) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_DEBUG_ASSERT(vkDevice_ == VK_NULL_HANDLE);

  if (desc.guid == 0UL) {
    IGL_LOG_ERROR("Invalid hardwareGuid(%lu)", desc.guid);
    return Result(Result::Code::Unsupported, "Vulkan is not supported");
  }

  vkPhysicalDevice_ =
      reinterpret_cast<VkPhysicalDevice>(desc.guid); // NOLINT(performance-no-int-to-ptr)

  VkPhysicalDeviceProperties deviceProperties{};
  vf_.vkGetPhysicalDeviceProperties(vkPhysicalDevice_, &deviceProperties);
  const Result versionResult = checkVulkanApiVersion(deviceProperties.apiVersion);
  if (!versionResult.isOk()) {
    return versionResult;
  }
  if (!vf_.vkGetPhysicalDeviceFeatures2 || !vf_.vkGetPhysicalDeviceProperties2) {
    return Result(Result::Code::Unsupported, "Vulkan 1.1 functions are not loaded");
  }

  // Caches the memory types
  vf_.vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice_, &memoryProperties);

  useStagingForBuffers_ = !ivkIsHostVisibleSingleHeapMemory(&memoryProperties);

  // Get the available physical device features
  VulkanFeatures availableFeatures(config_);
  availableFeatures.populateWithAvailablePhysicalDeviceFeatures(*this, vkPhysicalDevice_);

  // Use the requested features passed to the function (if any) or use the default features
  if (requestedFeatures) {
    constexpr auto instanceType = static_cast<size_t>(VulkanFeatures::ExtensionType::Instance);
    auto instanceExtensions = std::move(features_.extensions_[instanceType]);
    auto enabledInstanceExtensions = std::move(features_.enabledExtensions_[instanceType]);
    features_ = *requestedFeatures;
    features_.extensions_[instanceType] = std::move(instanceExtensions);
    features_.enabledExtensions_[instanceType] = std::move(enabledInstanceExtensions);
  }

  features_.extensions_[static_cast<size_t>(VulkanFeatures::ExtensionType::Device)].clear();
  features_.enabledExtensions_[static_cast<size_t>(VulkanFeatures::ExtensionType::Device)].clear();
  features_.resetRequestedFeatures();
  features_.populateWithAvailablePhysicalDeviceFeatures(*this, vkPhysicalDevice_);

  // The query above fills the request structs with supported values, so resolve any mutually
  // exclusive pairs before the request is checked and passed to vkCreateDevice().
  features_.resolveMutuallyExclusiveFeatures();

  // ... and check whether they are available in the physical device (they should be)
  {
    auto featureCheckResult = features_.checkSelectedFeatures(availableFeatures);
    if (!featureCheckResult.isOk()) {
      return featureCheckResult;
    }
  }

  vf_.vkGetPhysicalDeviceProperties2(vkPhysicalDevice_, &vkPhysicalDeviceProperties2_);

  const uint32_t apiVersion = vkPhysicalDeviceProperties2_.properties.apiVersion;

  if (config_.enableExtraLogs) {
    IGL_LOG_INFO("Device: %s\n", debugName ? debugName : "igl/vulkan/VulkanContext.cpp");
    IGL_LOG_INFO("Vulkan physical device: %s\n",
                 vkPhysicalDeviceProperties2_.properties.deviceName);
    IGL_LOG_INFO("           API version: %i.%i.%i.%i\n",
                 VK_API_VERSION_MAJOR(apiVersion),
                 VK_API_VERSION_MINOR(apiVersion),
                 VK_API_VERSION_PATCH(apiVersion),
                 VK_API_VERSION_VARIANT(apiVersion));
    IGL_LOG_INFO("           Driver info: %s %s\n",
                 vkPhysicalDeviceDriverProperties_.driverName,
                 vkPhysicalDeviceDriverProperties_.driverInfo);
  }

  features_.enumerate(vf_, vkPhysicalDevice_);

#if IGL_LOGGING_ENABLED
  if (config_.enableExtraLogs) {
    IGL_LOG_INFO("Vulkan physical device extensions:\n");
    // log available physical device extensions
    for (const auto& extension :
         features_.allAvailableExtensions(VulkanFeatures::ExtensionType::Device)) {
      IGL_LOG_INFO("  %s\n", extension.c_str());
    }
  }
#endif

  features_.enableCommonDeviceExtensions(config_);
  // Enable extra device extensions
  for (size_t i = 0; i < numExtraDeviceExtensions; i++) {
    features_.enable(extraDeviceExtensions[i], VulkanFeatures::ExtensionType::Device);
  }

  if (features_.enabled(VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME)) {
    VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT availableAstcHdr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 availableFeatures2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &availableAstcHdr};
    vf_.vkGetPhysicalDeviceFeatures2(vkPhysicalDevice_, &availableFeatures2);
    if (availableAstcHdr.textureCompressionASTC_HDR == VK_TRUE) {
      features_.featuresTextureCompressionAstcHdr.textureCompressionASTC_HDR = VK_TRUE;
      ivkAddNext(&features_.vkPhysicalDeviceFeatures2,
                 &features_.featuresTextureCompressionAstcHdr);
    }
  }

  if (features_.enabled(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
    // Extra opt-in device extensions are enabled after the chain was last assembled in
    // populateWithAvailablePhysicalDeviceFeatures(), so descriptorBuffer was gated out then.
    // Append just that struct rather than re-running assembleFeatureChain(), which would re-log
    // unrelated warnings. Its pNext is null. Does not touch has_VK_EXT_descriptor_buffer.
    ivkAddNext(&features_.vkPhysicalDeviceFeatures2, &features_.featuresDescriptorBuffer);
    vkPhysicalDeviceDescriptorBufferProperties_.pNext = vkPhysicalDeviceProperties2_.pNext;
    vkPhysicalDeviceProperties2_.pNext = &vkPhysicalDeviceDescriptorBufferProperties_;
    vf_.vkGetPhysicalDeviceProperties2(vkPhysicalDevice_, &vkPhysicalDeviceProperties2_);
  }

  if (features_.available(VK_EXT_MESH_SHADER_EXTENSION_NAME,
                          VulkanFeatures::ExtensionType::Device)) {
    vkPhysicalDeviceDescriptorIndexingProperties_.pNext = &vkPhysicalDeviceMeshShaderPropertiesEXT_;
    vf_.vkGetPhysicalDeviceProperties2(vkPhysicalDevice_, &vkPhysicalDeviceProperties2_);
  }

  VulkanQueuePool queuePool(vf_, vkPhysicalDevice_);

  // Reserve IGL Vulkan queues
  const auto graphicsQueueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  const auto computeQueueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_COMPUTE_BIT);

  if (!graphicsQueueDescriptor.isValid()) {
    IGL_LOG_ERROR("VK_QUEUE_GRAPHICS_BIT is not supported");
    return Result(Result::Code::Unsupported, "VK_QUEUE_GRAPHICS_BIT is not supported");
  }

  if (!computeQueueDescriptor.isValid()) {
    IGL_LOG_ERROR("VK_QUEUE_COMPUTE_BIT is not supported");
    return Result(Result::Code::Unsupported, "VK_QUEUE_COMPUTE_BIT is not supported");
  }

  deviceQueues_.graphicsQueueFamilyIndex = graphicsQueueDescriptor.familyIndex;
  deviceQueues_.computeQueueFamilyIndex = computeQueueDescriptor.familyIndex;
  graphicsQueueTimestampValidBits_ =
      queryTimestampValidBits(vf_, vkPhysicalDevice_, deviceQueues_.graphicsQueueFamilyIndex);

  queuePool.reserveQueue(graphicsQueueDescriptor);
  queuePool.reserveQueue(computeQueueDescriptor);

  const auto qcis = queuePool.getQueueCreationInfos();

  auto deviceExtensions = features_.allEnabled(VulkanFeatures::ExtensionType::Device);

  features_.enableRequestedFeatures(*this, vkPhysicalDevice_, apiVersion);
  features_.resolveMutuallyExclusiveFeatures();

  VkDevice device = nullptr;
  VK_ASSERT_RETURN(ivkCreateDevice(&vf_,
                                   vkPhysicalDevice_,
                                   qcis.size(),
                                   qcis.data(),
                                   deviceExtensions.size(),
                                   deviceExtensions.data(),
                                   &features_.vkPhysicalDeviceFeatures2,
                                   &device));

  // Check that device is not null before proceeding
  if (device == VK_NULL_HANDLE) {
    return Result(Result::Code::InvalidOperation, "Failed to create Vulkan device");
  }
#if defined(IGL_CMAKE_BUILD)
  if (!config_.enableConcurrentVkDevicesSupport) {
    // Do not remove for backward compatibility with projects using global functions.
    volkLoadDevice(device);
  }
#endif

  // Table functions are always bound to a device. Project using enableConcurrentVkDevicesSupport
  // should use own copy of function table bound to a device.
  vulkan::functions::loadDeviceFunctions(
      *tableImpl_, device, getVkPhysicalDeviceProperties().apiVersion);

  if (features_.has_VK_KHR_buffer_device_address && !vf_.vkGetBufferDeviceAddressKHR) {
    return Result(Result::Code::InvalidOperation, "Cannot initialize VK_KHR_buffer_device_address");
  }

#if !IGL_VULKAN_HAS_LEGACY_RENDERPASS
  // Promote VK_KHR_create_renderpass2 extension functions to Vulkan 1.2 core function pointers
  // when core is unavailable (e.g., devices that support Vulkan 1.1 + VK_KHR_create_renderpass2)
  if (!tableImpl_->vkCreateRenderPass2) {
    tableImpl_->vkCreateRenderPass2 = (PFN_vkCreateRenderPass2)tableImpl_->vkCreateRenderPass2KHR;
    tableImpl_->vkCmdBeginRenderPass2 =
        (PFN_vkCmdBeginRenderPass2)tableImpl_->vkCmdBeginRenderPass2KHR;
    tableImpl_->vkCmdEndRenderPass2 = (PFN_vkCmdEndRenderPass2)tableImpl_->vkCmdEndRenderPass2KHR;
    tableImpl_->vkCmdNextSubpass2 = (PFN_vkCmdNextSubpass2)tableImpl_->vkCmdNextSubpass2KHR;
  }
#endif // !IGL_VULKAN_HAS_LEGACY_RENDERPASS

  vf_.vkGetDeviceQueue(
      device, deviceQueues_.graphicsQueueFamilyIndex, 0, &deviceQueues_.graphicsQueue);
  vf_.vkGetDeviceQueue(
      device, deviceQueues_.computeQueueFamilyIndex, 0, &deviceQueues_.computeQueue);

  vkDevice_ = device;

  VK_ASSERT(ivkSetDebugObjectName(&vf_,
                                  vkDevice_,
                                  VK_OBJECT_TYPE_DEVICE,
                                  (uint64_t)vkDevice_,
                                  IGL_FORMAT("Device: VulkanContext::device_ {}",
                                             debugName ? debugName : "igl/vulkan/VulkanContext.cpp")
                                      .c_str()));

  VK_ASSERT(ivkSetDebugObjectName(
      &vf_,
      vkDevice_,
      VK_OBJECT_TYPE_QUEUE,
      (uint64_t)deviceQueues_.graphicsQueue,
      IGL_FORMAT("Graphics{} queue: {}",
                 deviceQueues_.graphicsQueue == deviceQueues_.computeQueue ? "/compute" : "",
                 debugName ? debugName : "igl/vulkan/VulkanContext.cpp")
          .c_str()));
  if (deviceQueues_.graphicsQueue != deviceQueues_.computeQueue) {
    VK_ASSERT(ivkSetDebugObjectName(
        &vf_,
        vkDevice_,
        VK_OBJECT_TYPE_QUEUE,
        (uint64_t)deviceQueues_.computeQueue,
        IGL_FORMAT("Compute queue: {}", debugName ? debugName : "igl/vulkan/VulkanContext.cpp")
            .c_str()));
  }

  immediate_ = std::make_unique<VulkanImmediateCommands>(vf_,
                                                         device,
                                                         deviceQueues_.graphicsQueueFamilyIndex,
                                                         config_.exportableFences,
                                                         features_.has_VK_KHR_timeline_semaphore,
                                                         features_.has_VK_KHR_synchronization2,
                                                         "VulkanContext::immediate_",
                                                         config_.chainSubmitsWithSemaphores);
  IGL_DEBUG_ASSERT(config_.maxResourceCount > 0,
                   "Max resource count needs to be greater than zero");
  syncSubmitHandles.resize(config_.maxResourceCount);

  // create Vulkan pipeline cache
  {
    const VkPipelineCacheCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .flags = VkPipelineCacheCreateFlags(0),
        .initialDataSize = config_.pipelineCacheDataSize,
        .pInitialData = config_.pipelineCacheData,
    };
    vf_.vkCreatePipelineCache(device, &ci, nullptr, &pipelineCache_);
    VK_ASSERT(ivkSetDebugObjectName(&vf_,
                                    device,
                                    VK_OBJECT_TYPE_PIPELINE_CACHE,
                                    (uint64_t)pipelineCache_,
                                    "Pipeline Cache: VulkanContext::pipelineCache_"));
  }

  // Create Vulkan Memory Allocator
  if (IGL_VULKAN_USE_VMA) {
    VK_ASSERT_RETURN(
        ivkVmaCreateAllocator(&vf_,
                              vkPhysicalDevice_,
                              vkDevice_,
                              vkInstance_,
                              apiVersion > VK_API_VERSION_1_3 ? VK_API_VERSION_1_3 : apiVersion,
                              features_.has_VK_KHR_buffer_device_address,
                              static_cast<VkDeviceSize>(config_.vmaPreferredLargeHeapBlockSize),
                              &pimpl_->vma));
  }

  // Caches the memory types
  vf_.vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice_, &memoryProperties);

  // The staging device will use VMA to allocate a buffer, so this needs
  // to happen after VMA has been initialized.
  stagingDevice_ = std::make_unique<VulkanStagingDevice>(*this);

  // Unextended Vulkan 1.1 does not allow sparse (VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT)
  // bindings. Our descriptor set layout emulates OpenGL binding slots but we cannot put
  // VK_NULL_HANDLE into empty slots. We use dummy buffers to stick them into those empty slots.
  dummyUniformBuffer_ = createBuffer(256,
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     nullptr,
                                     "Buffer: dummy uniform");
  dummyStorageBuffer_ = createBuffer(256,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     nullptr,
                                     "Buffer: dummy storage");
  if (dummyUniformBuffer_ == nullptr || dummyStorageBuffer_ == nullptr) {
    // Nothing has used the one that exists.
    if (dummyUniformBuffer_ != nullptr) {
      dummyUniformBuffer_->discardNeverSubmitted();
    }
    if (dummyStorageBuffer_ != nullptr) {
      dummyStorageBuffer_->discardNeverSubmitted();
    }
    return Result(Result::Code::RuntimeError, "Cannot create the dummy uniform/storage buffers");
  }

  // default texture, also bound in place of an unbound input attachment
  {
    const VkFormat dummyTextureFormat = VK_FORMAT_R8G8B8A8_UNORM;
    Result result;
    auto image = createImage(VK_IMAGE_TYPE_2D,
                             VkExtent3D{.width = 1, .height = 1, .depth = 1},
                             dummyTextureFormat,
                             1,
                             1,
                             VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             0,
                             VK_SAMPLE_COUNT_1_BIT,
                             &result,
                             "Image: dummy 1x1");
    if (!result.isOk()) {
      return result; // NOLINT(clang-diagnostic-nrvo)
    }
    if (!image.valid()) {
      return Result(Result::Code::InvalidOperation, "Cannot create VulkanImage");
    }
    auto imageView = image.createImageView(VK_IMAGE_VIEW_TYPE_2D,
                                           dummyTextureFormat,
                                           VK_IMAGE_ASPECT_COLOR_BIT,
                                           0,
                                           VK_REMAINING_MIP_LEVELS,
                                           0,
                                           1,
                                           "Image View: dummy 1x1");
    if (!imageView.valid()) {
      image.discardNeverSubmitted(); // nothing has used the dummy image
      return Result(Result::Code::InvalidOperation, "Cannot create VulkanImageView");
    }
    pimpl_->dummyTexture =
        textures_.create(std::make_shared<VulkanTexture>(std::move(image), std::move(imageView)));
    IGL_DEBUG_ASSERT(textures_.numObjects() == 1);
    const uint32_t pixel = 0xFF000000;

    const VkImageAspectFlags imageAspectFlags =
        (*textures_.get(pimpl_->dummyTexture))->imageView_.getVkImageAspectFlags();
    stagingDevice_->imageData(
        (*textures_.get(pimpl_->dummyTexture))->image,
        TextureType::TwoD,
        TextureRangeDesc::new2D(0, 0, 1, 1),
        TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8),
        0,
        imageAspectFlags,
        &pixel);
  }

  if (features_.has_VK_EXT_descriptor_buffer) {
    pimpl_->descriptorBuffersArena = std::make_unique<DescriptorBuffersArena>(*this);
    if (!pimpl_->descriptorBuffersArena->valid()) {
      return Result(Result::Code::RuntimeError, "Cannot create the first descriptor buffer");
    }
  }

  // default sampler
  pimpl_->dummySampler = createSampler(
      {
          .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
          .pNext = nullptr,
          .flags = 0,
          .magFilter = VK_FILTER_LINEAR,
          .minFilter = VK_FILTER_LINEAR,
          .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
          .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
          .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
          .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
          .mipLodBias = 0.0f,
          .anisotropyEnable = VK_FALSE,
          .maxAnisotropy = 0.0f,
          .compareEnable = VK_FALSE,
          .compareOp = VK_COMPARE_OP_ALWAYS,
          .minLod = 0.0f,
          .maxLod = 0.0f,
          .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
          .unnormalizedCoordinates = VK_FALSE,
      },
      VK_FORMAT_UNDEFINED,
      nullptr,
      "Sampler: default");
  IGL_DEBUG_ASSERT(samplers_.numObjects() == 1);

  growBindlessDescriptorPool(pimpl_->currentMaxBindlessTextures,
                             pimpl_->currentMaxBindlessSamplers);

  querySurfaceCapabilities();

#if defined(IGL_WITH_TRACY_GPU)
  VK_ASSERT(ivkCreateCommandPool(&vf_,
                                 vkDevice_,
                                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                                     VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                 deviceQueues_.graphicsQueueFamilyIndex,
                                 &profilingCommandPool_));

  ivkSetDebugObjectName(&vf_,
                        vkDevice_,
                        VK_OBJECT_TYPE_COMMAND_POOL,
                        (uint64_t)profilingCommandPool_,
                        "VulkanContext::profilingCommandPool_ (Tracy)");

  profilingCommandBuffer_ = VK_NULL_HANDLE;
  VK_ASSERT(
      ivkAllocateCommandBuffer(&vf_, vkDevice_, profilingCommandPool_, &profilingCommandBuffer_));

  ivkSetDebugObjectName(&vf_,
                        vkDevice_,
                        VK_OBJECT_TYPE_COMMAND_BUFFER,
                        (uint64_t)profilingCommandBuffer_,
                        "VulkanContext::profilingCommandBuffer_ (Tracy)");

#if defined(VK_EXT_calibrated_timestamps)
  if (features_.enabled(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
    tracyCtx_ = TracyVkContextCalibrated(vkInstance_,
                                         getVkPhysicalDevice(),
                                         getVkDevice(),
                                         deviceQueues_.graphicsQueue,
                                         profilingCommandBuffer_,
                                         tableImpl_->vkGetInstanceProcAddr,
                                         tableImpl_->vkGetDeviceProcAddr);
  }
#endif // VK_EXT_calibrated_timestamps
  // If VK_EXT_calibrated_timestamps is not available or it has not been enabled, use the
  // uncalibrated Tracy context
  if (!tracyCtx_) {
    tracyCtx_ = TracyVkContext(vkInstance_,
                               getVkPhysicalDevice(),
                               getVkDevice(),
                               deviceQueues_.graphicsQueue,
                               profilingCommandBuffer_,
                               tableImpl_->vkGetInstanceProcAddr,
                               tableImpl_->vkGetDeviceProcAddr);
  }

  IGL_DEBUG_ASSERT(tracyCtx_, "Failed to create Tracy GPU profiling context");
#endif // IGL_WITH_TRACY_GPU

  return Result();
}

void VulkanContext::growBindlessDescriptorPool(uint32_t newMaxTextures, uint32_t newMaxSamplers) {
  // only do allocations if actually enabled
  if (!config_.enableDescriptorIndexing) {
    return;
  }

  IGL_PROFILER_FUNCTION();

  pimpl_->currentMaxBindlessTextures = newMaxTextures;
  pimpl_->currentMaxBindlessSamplers = newMaxSamplers;

#if IGL_VULKAN_PRINT_COMMANDS
  IGL_LOG_INFO("growBindlessDescriptorPool(%u, %u)\n", newMaxTextures, newMaxSamplers);
#endif // IGL_VULKAN_PRINT_COMMANDS

  // macOS: MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS is required when using this with MoltenVK
  IGL_DEBUG_ASSERT(
      newMaxTextures <= vkPhysicalDeviceDescriptorIndexingProperties_
                            .maxDescriptorSetUpdateAfterBindSampledImages,
      "Max Textures exceeded: %u (hardware max %u)",
      newMaxTextures,
      vkPhysicalDeviceDescriptorIndexingProperties_.maxDescriptorSetUpdateAfterBindSampledImages);

  // macOS: MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS is required when using this with MoltenVK
  IGL_DEBUG_ASSERT(
      newMaxSamplers <=
          vkPhysicalDeviceDescriptorIndexingProperties_.maxDescriptorSetUpdateAfterBindSamplers,
      "Max Samplers exceeded %u (hardware max %u)",
      newMaxSamplers,
      vkPhysicalDeviceDescriptorIndexingProperties_.maxDescriptorSetUpdateAfterBindSamplers);

  VkDevice device = getVkDevice();

  if (pimpl_->dpBindless != VK_NULL_HANDLE) {
    deferredTask(std::packaged_task<void()>([vf = &vf_, device, dp = pimpl_->dpBindless]() {
      vf->vkDestroyDescriptorPool(device, dp, nullptr);
    }));
  }

  // create default descriptor set layout which is going to be shared by graphics pipelines
  constexpr uint32_t kNumBindings = 7;
  constexpr VkShaderStageFlags stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
  const std::array<VkDescriptorSetLayoutBinding, kNumBindings> bindings = {
      VkDescriptorSetLayoutBinding{
          .binding = kBinding_Texture2D,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = pimpl_->currentMaxBindlessTextures,
          .stageFlags = stageFlags,
      },
      VkDescriptorSetLayoutBinding{
          .binding = kBinding_Texture2DArray,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = pimpl_->currentMaxBindlessTextures,
          .stageFlags = stageFlags,
      },
      VkDescriptorSetLayoutBinding{
          .binding = kBinding_Texture3D,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = pimpl_->currentMaxBindlessTextures,
          .stageFlags = stageFlags,
      },
      VkDescriptorSetLayoutBinding{
          .binding = kBinding_TextureCube,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = pimpl_->currentMaxBindlessTextures,
          .stageFlags = stageFlags,
      },
      VkDescriptorSetLayoutBinding{
          .binding = kBinding_Sampler,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = pimpl_->currentMaxBindlessSamplers,
          .stageFlags = stageFlags,
      },
      VkDescriptorSetLayoutBinding{
          .binding = kBinding_SamplerShadow,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = pimpl_->currentMaxBindlessSamplers,
          .stageFlags = stageFlags,
      },
      VkDescriptorSetLayoutBinding{
          .binding = kBinding_StorageImages,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = pimpl_->currentMaxBindlessTextures,
          .stageFlags = stageFlags,
      },
  };
  const uint32_t flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                         VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                         VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
  const std::array<VkDescriptorBindingFlags, kNumBindings> bindingFlags = {
      flags, flags, flags, flags, flags, flags, flags};
  IGL_DEBUG_ASSERT(bindingFlags.back() == flags);
  pimpl_->dslBindless = std::make_unique<VulkanDescriptorSetLayout>(
      *this,
      VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT,
      kNumBindings,
      bindings.data(),
      bindingFlags.data(),
      "Descriptor Set Layout: VulkanContext::dslBindless_");
  // create default descriptor pool and allocate 1 descriptor set
  const std::array<VkDescriptorPoolSize, kNumBindings> poolSizes = {
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                           .descriptorCount = pimpl_->currentMaxBindlessTextures},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                           .descriptorCount = pimpl_->currentMaxBindlessTextures},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                           .descriptorCount = pimpl_->currentMaxBindlessTextures},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                           .descriptorCount = pimpl_->currentMaxBindlessTextures},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_SAMPLER,
                           .descriptorCount = pimpl_->currentMaxBindlessSamplers},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_SAMPLER,
                           .descriptorCount = pimpl_->currentMaxBindlessSamplers},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                           .descriptorCount = pimpl_->currentMaxBindlessTextures},
  };
  VK_ASSERT(ivkCreateDescriptorPool(&vf_,
                                    device,
                                    VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                                    1,
                                    static_cast<uint32_t>(poolSizes.size()),
                                    poolSizes.data(),
                                    &pimpl_->dpBindless));
  VK_ASSERT(ivkSetDebugObjectName(&vf_,
                                  device,
                                  VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                  (uint64_t)pimpl_->dpBindless,
                                  "Descriptor Pool: dpBindless_"));
  VK_ASSERT(ivkAllocateDescriptorSet(&vf_,
                                     device,
                                     pimpl_->dpBindless,
                                     pimpl_->dslBindless->getVkDescriptorSetLayout(),
                                     &pimpl_->dsBindless));
  VK_ASSERT(ivkSetDebugObjectName(&vf_,
                                  device,
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                  (uint64_t)pimpl_->dsBindless,
                                  "Descriptor Set: dsBindless_"));
}

Result VulkanContext::initSwapchain(uint32_t width, uint32_t height) {
  IGL_PROFILER_FUNCTION();

  if (!vkDevice_ || !immediate_) {
    IGL_LOG_ERROR("Call initContext() first");
    return Result(Result::Code::Unsupported, "Call initContext() first");
  }

  if (swapchain_) {
    vf_.vkDeviceWaitIdle(vkDevice_);
    swapchain_ = nullptr; // Destroy old swapchain first
  }

  if (!width || !height) {
    return Result();
  }

  swapchain_ = std::make_unique<VulkanSwapchain>(*this, width, height);
  if (!swapchain_->valid()) {
    swapchain_.reset();
    return Result(Result::Code::RuntimeError, "Cannot create the swapchain image views");
  }

  if (features_.has_VK_KHR_timeline_semaphore && features_.has_VK_KHR_synchronization2) {
    timelineSemaphore_ = std::make_unique<VulkanSemaphore>(
        vf_, getVkDevice(), 0, false, "Semaphore: VulkanContext::timelineSemaphore_");
  }

  return swapchain_ ? Result() : Result(Result::Code::RuntimeError, "Failed to create Swapchain");
}

VkExtent2D VulkanContext::getSwapchainExtent() const {
  return hasSwapchain() ? swapchain_->getExtent() : VkExtent2D{.width = 0, .height = 0};
}

Result VulkanContext::waitIdle() const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  for (VkQueue queue : {deviceQueues_.graphicsQueue, deviceQueues_.computeQueue}) {
    VK_ASSERT_RETURN(vf_.vkQueueWaitIdle(queue));
  }

  return getResultFromVkResult(VK_SUCCESS);
}

Result VulkanContext::present() const {
  if (!hasSwapchain()) {
    return Result(Result::Code::InvalidOperation, "No swapchain available");
  }

  return swapchain_->present(swapchain_->getPresentSemaphore());
}

std::unique_ptr<VulkanBuffer> VulkanContext::createBuffer(VkDeviceSize bufferSize,
                                                          VkBufferUsageFlags usageFlags,
                                                          VkMemoryPropertyFlags memFlags,
                                                          Result* IGL_NULLABLE outResult,
                                                          const char* IGL_NULLABLE
                                                              debugName) const {
  IGL_PROFILER_FUNCTION();

#define ENSURE_BUFFER_SIZE(flag, maxSize)                                                      \
  if ((usageFlags & flag) != 0) {                                                              \
    if (!IGL_DEBUG_VERIFY(bufferSize <= maxSize)) {                                            \
      IGL_LOG_INFO("Max size of buffer exceeded " #flag ": %llu > %llu", bufferSize, maxSize); \
      Result::setResult(outResult,                                                             \
                        Result(Result::Code::InvalidOperation, "Buffer size exceeded" #flag)); \
      return nullptr;                                                                          \
    }                                                                                          \
  }

  const VkPhysicalDeviceLimits& limits = getVkPhysicalDeviceProperties().limits;

  // any buffer
  ENSURE_BUFFER_SIZE(VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM, limits.maxStorageBufferRange);
#undef ENSURE_BUFFER_SIZE

  auto buffer =
      std::make_unique<VulkanBuffer>(*this, vkDevice_, bufferSize, usageFlags, memFlags, debugName);
  if (!buffer->valid()) {
    Result::setResult(outResult, getResultFromVkResult(buffer->initResult()));
    return nullptr;
  }
  Result::setOk(outResult);
  return buffer;
}

VulkanImage VulkanContext::createImage(VkImageType imageType,
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
                                       const char* IGL_NULLABLE debugName) const {
  IGL_PROFILER_FUNCTION();

  if (!validateImageLimits(
          imageType, samples, extent, getVkPhysicalDeviceProperties().limits, outResult)) {
    return VulkanImage();
  }

  VulkanImage image{*this,
                    extent,
                    imageType,
                    format,
                    mipLevels,
                    arrayLayers,
                    tiling,
                    usageFlags,
                    memFlags,
                    flags,
                    samples,
                    debugName};
  if (!image.valid()) {
    Result::setResult(outResult, getResultFromVkResult(image.initResult()));
    return VulkanImage();
  }
  Result::setOk(outResult);
  return image;
}

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
std::unique_ptr<VulkanImage> VulkanContext::createImageFromAndroidHardwareBuffer(
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
    const char* IGL_NULLABLE debugName) const {
  if (!validateImageLimits(
          imageType, samples, extent, getVkPhysicalDeviceProperties().limits, outResult)) {
    return nullptr;
  }
  auto image = std::make_unique<VulkanImage>(*this,
                                             androidHardwareBuffer,
                                             memoryAllocationSize,
                                             extent,
                                             imageType,
                                             format,
                                             mipLevels,
                                             arrayLayers,
                                             tiling,
                                             usageFlags,
                                             flags,
                                             samples,
                                             debugName);
  if (!image->valid()) {
    Result::setResult(outResult, getResultFromVkResult(image->initResult()));
    return nullptr;
  }
  Result::setOk(outResult);
  return image;
}
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

std::unique_ptr<VulkanImage> VulkanContext::createImageFromFileDescriptor(
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
    const char* IGL_NULLABLE debugName) const {
  if (!validateImageLimits(
          imageType, samples, extent, getVkPhysicalDeviceProperties().limits, outResult)) {
    return nullptr;
  }

  return std::make_unique<VulkanImage>(*this,
                                       fileDescriptor,
                                       memoryAllocationSize,
                                       extent,
                                       imageType,
                                       format,
                                       mipLevels,
                                       arrayLayers,
                                       tiling,
                                       usageFlags,
                                       flags,
                                       samples,
                                       debugName);
}

void VulkanContext::pruneTextures() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);

  // here we remove deleted textures - everything which has only 1 reference is owned by this
  // context and can be released safely. Releasing a texture view can leave its base texture with
  // only this context's reference, so repeat until nothing more is released.
  bool released = true;
  while (released) {
    released = false;
    for (uint32_t i = 1; i < static_cast<uint32_t>(textures_.objects_.size()); i++) {
      if (textures_.objects_[i] && textures_.objects_[i].use_count() == 1) {
        textures_.destroy(textures_.getHandle(i));
        released = true;
      }
    }
  }
}

VkResult VulkanContext::checkAndUpdateDescriptorSets() {
  if (!awaitingCreation_) {
    // nothing to update here
    return VK_SUCCESS;
  }

  // newly created resources can be used immediately - make sure they are put into descriptor sets
  IGL_PROFILER_FUNCTION();

  // The bindless descriptor set is not double buffered, so wait for the previous submission
  // before rewriting it. Waiting up here rather than just before vkUpdateDescriptorSets() means
  // a wait that does not succeed mutates nothing: awaitingCreation_ stays set and a later pass
  // retries. Callers treat any non-success return as fatal, so that deferral must not be
  // reported as an error. After a wait that does not succeed, later attempts poll instead of
  // re-arming the full fenceTimeoutNanoseconds on every pass.
  //
  // Without descriptor indexing there is no such set to protect, and pruneTextures() below is
  // safe on its own because it destroys image views through deferred tasks that already wait on
  // the submit handle. Waiting on that path would only let a timeout stop reclaiming textures.
  if (config_.enableDescriptorIndexing) {
    const VkResult waitResult = immediate_->wait(
        immediate_->getLastSubmitHandle(),
        descriptorSetWaitTimeout(config_.fenceTimeoutNanoseconds, descriptorSetWaitDegraded_));
    if (waitResult != VK_SUCCESS) {
      descriptorSetWaitDegraded_ = true;
      IGL_LOG_ERROR_ONCE("Deferring the bindless descriptor set update: fence wait returned %d\n",
                         static_cast<int>(waitResult));
      return VK_SUCCESS;
    }
    descriptorSetWaitDegraded_ = false;
  }

  pruneTextures();

  // update Vulkan bindless descriptor sets here
  if (!config_.enableDescriptorIndexing) {
    return VK_SUCCESS;
  }

  uint32_t newMaxTextures = pimpl_->currentMaxBindlessTextures;
  uint32_t newMaxSamplers = pimpl_->currentMaxBindlessSamplers;

  while (textures_.objects_.size() > newMaxTextures) {
    newMaxTextures *= 2;
  }
  while (samplers_.objects_.size() > newMaxSamplers) {
    newMaxSamplers *= 2;
  }
  if (newMaxTextures != pimpl_->currentMaxBindlessTextures ||
      newMaxSamplers != pimpl_->currentMaxBindlessSamplers) {
    growBindlessDescriptorPool(newMaxTextures, newMaxSamplers);
  }

  // make sure the guard values are always there
  IGL_DEBUG_ASSERT(!textures_.objects_.empty());
  IGL_DEBUG_ASSERT(!samplers_.objects_.empty());

  // 1. Sampled and storage images
  std::vector<VkDescriptorImageInfo> infoSampledImages;
  std::vector<VkDescriptorImageInfo> infoStorageImages;
  infoSampledImages.reserve(textures_.objects_.size());
  infoStorageImages.reserve(textures_.objects_.size());

  // use the dummy texture/sampler to avoid sparse array
  VkImageView dummyImageView = textures_.objects_[0]->imageView_.getVkImageView();
  VkSampler dummySampler = samplers_.objects_[0].vkSampler;

  for (const auto& entry : textures_.objects_) {
    const VulkanTexture* texture = entry.get();
    if (texture) {
      // multisampled images cannot be directly accessed from shaders
      const bool isTextureAvailable =
          (texture->image.samples_ & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT;
      const bool isSampledImage = isTextureAvailable && texture->image.isSampledImage();
      const bool isStorageImage = isTextureAvailable && texture->image.isStorageImage();
      infoSampledImages.push_back(
          {.sampler = dummySampler,
           .imageView = isSampledImage ? texture->imageView_.getVkImageView() : dummyImageView,
           .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
      infoStorageImages.push_back(VkDescriptorImageInfo{
          .sampler = VK_NULL_HANDLE,
          .imageView = isStorageImage ? texture->imageView_.getVkImageView() : dummyImageView,
          .imageLayout = VK_IMAGE_LAYOUT_GENERAL});
    } else {
      infoSampledImages.push_back({.sampler = dummySampler,
                                   .imageView = dummyImageView,
                                   .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
      infoStorageImages.push_back(VkDescriptorImageInfo{.sampler = VK_NULL_HANDLE,
                                                        .imageView = dummyImageView,
                                                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL});
    }
    IGL_DEBUG_ASSERT(infoSampledImages.back().imageView != VK_NULL_HANDLE);
    IGL_DEBUG_ASSERT(infoStorageImages.back().imageView != VK_NULL_HANDLE);
  }

  // 2. Samplers
  std::vector<VkDescriptorImageInfo> infoSamplers;
  infoSamplers.reserve(samplers_.objects_.size());

  for (const auto& entry : samplers_.objects_) {
    const VulkanSampler* sampler = &entry;
    infoSamplers.push_back({.sampler = sampler ? sampler->vkSampler : dummySampler,
                            .imageView = VK_NULL_HANDLE,
                            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED});
  }

  std::vector<VkWriteDescriptorSet> write;

  if (!infoSampledImages.empty()) {
    // use the same indexing for every texture type
    for (uint32_t i = kBinding_Texture2D; i != kBinding_TextureCube + 1; i++) {
      write.push_back(
          ivkGetWriteDescriptorSetImageInfo(pimpl_->dsBindless,
                                            i,
                                            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                            static_cast<uint32_t>(infoSampledImages.size()),
                                            infoSampledImages.data()));
    }
  };

  if (!infoSamplers.empty()) {
    for (uint32_t i = kBinding_Sampler; i != kBinding_SamplerShadow + 1; i++) {
      write.push_back(ivkGetWriteDescriptorSetImageInfo(pimpl_->dsBindless,
                                                        i,
                                                        VK_DESCRIPTOR_TYPE_SAMPLER,
                                                        static_cast<uint32_t>(infoSamplers.size()),
                                                        infoSamplers.data()));
    }
  }

  if (!infoStorageImages.empty()) {
    write.push_back(
        ivkGetWriteDescriptorSetImageInfo(pimpl_->dsBindless,
                                          kBinding_StorageImages,
                                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          static_cast<uint32_t>(infoStorageImages.size()),
                                          infoStorageImages.data()));
  };

  // do not switch to the next descriptor set if there is nothing to update
  if (!write.empty()) {
#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("Updating descriptor set dsBindless_\n");
#endif // IGL_VULKAN_PRINT_COMMANDS
    vf_.vkUpdateDescriptorSets(
        vkDevice_, static_cast<uint32_t>(write.size()), write.data(), 0, nullptr);
  }

  awaitingCreation_ = false;
  return VK_SUCCESS;
}

std::shared_ptr<VulkanTexture> VulkanContext::createTexture(
    VulkanImage&& image,
    VulkanImageView&& imageView,
    [[maybe_unused]] const char* IGL_NULLABLE debugName) const {
  IGL_PROFILER_FUNCTION();

  if (!image.valid() || !imageView.valid()) {
    IGL_LOG_ERROR("VulkanContext::createTexture(): refused, the %s is invalid (VkResult %d)\n",
                  !image.valid() ? "image" : "image view",
                  static_cast<int>(!image.valid() ? image.initResult() : imageView.initResult()));
    // The valid part was created for this texture only: nothing can have used it.
    imageView.destroyNeverSubmitted();
    image.discardNeverSubmitted();
    return nullptr;
  }

  const TextureHandle handle =
      textures_.create(std::make_shared<VulkanTexture>(std::move(image), std::move(imageView)));

  auto texture = *textures_.get(handle);

  if (!IGL_DEBUG_VERIFY(texture)) {
    return nullptr;
  }

  texture->textureId_ = handle.index();

  awaitingCreation_ = true;

  return texture; // NOLINT(clang-diagnostic-nrvo)
}

std::shared_ptr<VulkanTexture> VulkanContext::createTextureFromVkImage(
    VkImage vkImage,
    VulkanImageCreateInfo imageCreateInfo,
    VulkanImageViewCreateInfo imageViewCreateInfo,
    const char* IGL_NULLABLE debugName) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  auto iglImage = VulkanImage(*this, vkImage, imageCreateInfo, debugName);
  // `debugName` is an optional (IGL_NULLABLE) debug label forwarded to createImageView(), which
  // passes it through to ivkSetDebugObjectName() where null is handled; this is a false positive.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  auto imageView = iglImage.createImageView(imageViewCreateInfo, debugName);
  return createTexture(std::move(iglImage), std::move(imageView), debugName);
}

SamplerHandle VulkanContext::createSampler(const VkSamplerCreateInfo& ci,
                                           VkFormat yuvVkFormat,
                                           Result* IGL_NULLABLE outResult,
                                           const char* IGL_NULLABLE debugName) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  VkSamplerCreateInfo cInfo = ci;
  VkSamplerYcbcrConversionInfo conversionInfo{};

  if (yuvVkFormat != VK_FORMAT_UNDEFINED) {
    conversionInfo = getOrCreateYcbcrConversionInfo(yuvVkFormat);
    cInfo.pNext = &conversionInfo;
    // must be CLAMP_TO_EDGE
    // https://vulkan.lunarg.com/doc/view/1.3.268.0/windows/1.3-extensions/vkspec.html#VUID-VkSamplerCreateInfo-addressModeU-01646
    cInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cInfo.anisotropyEnable = VK_FALSE;
    cInfo.unnormalizedCoordinates = VK_FALSE;
  }

  VkDevice device = getVkDevice();
  VulkanSampler sampler;
  VK_ASSERT(vf_.vkCreateSampler(device, &cInfo, nullptr, &sampler.vkSampler));
  VK_ASSERT(ivkSetDebugObjectName(
      &vf_, device, VK_OBJECT_TYPE_SAMPLER, (uint64_t)sampler.vkSampler, debugName));
  const SamplerHandle handle = samplers_.create(static_cast<VulkanSampler&&>(sampler));

  samplers_.get(handle)->samplerId = handle.index();

  awaitingCreation_ = true;

  return handle;
}

void VulkanContext::querySurfaceCapabilities() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  // This is not an exhaustive list. It's only formats that we are using.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  const VkFormat depthFormats[] = {VK_FORMAT_D32_SFLOAT_S8_UINT,
                                   VK_FORMAT_D24_UNORM_S8_UINT,
                                   VK_FORMAT_D16_UNORM_S8_UINT,
                                   VK_FORMAT_D32_SFLOAT,
                                   VK_FORMAT_D16_UNORM,
                                   VK_FORMAT_S8_UINT};
  deviceDepthFormats_.reserve(IGL_ARRAY_NUM_ELEMENTS(depthFormats));
  for (const auto& depthFormat : depthFormats) {
    VkFormatProperties formatProps{};
    vf_.vkGetPhysicalDeviceFormatProperties(vkPhysicalDevice_, depthFormat, &formatProps);

    if ((formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0 ||
        (formatProps.bufferFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0 ||
        (formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
      deviceDepthFormats_.push_back(depthFormat);
    }
  }

  if (vkSurface_ != VK_NULL_HANDLE) {
    vf_.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        vkPhysicalDevice_, vkSurface_, &deviceSurfaceCaps_);

    uint32_t formatCount = 0;
    vf_.vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_, vkSurface_, &formatCount, nullptr);

    if (formatCount) {
      deviceSurfaceFormats_.resize(formatCount);
      vf_.vkGetPhysicalDeviceSurfaceFormatsKHR(
          vkPhysicalDevice_, vkSurface_, &formatCount, deviceSurfaceFormats_.data());
    }

    uint32_t presentModeCount = 0;
    vf_.vkGetPhysicalDeviceSurfacePresentModesKHR(
        vkPhysicalDevice_, vkSurface_, &presentModeCount, nullptr);

    if (presentModeCount) {
      devicePresentModes_.resize(presentModeCount);
      vf_.vkGetPhysicalDeviceSurfacePresentModesKHR(
          vkPhysicalDevice_, vkSurface_, &presentModeCount, devicePresentModes_.data());
    }
  }
}

VkFormat VulkanContext::getClosestDepthStencilFormat(TextureFormat desiredFormat) const {
  IGL_DEBUG_ASSERT(!deviceDepthFormats_.empty());
  // get a list of compatible depth formats for a given desired format
  // The list will contain depth format that are ordered from most to least closest
  const std::vector<VkFormat> compatibleDepthStencilFormatList =
      getCompatibleDepthStencilFormats(desiredFormat);

  // check if any of the format in compatible list is supported
  for (auto depthStencilFormat : compatibleDepthStencilFormatList) {
    if (std::find(deviceDepthFormats_.begin(), deviceDepthFormats_.end(), depthStencilFormat) !=
        deviceDepthFormats_.end()) {
      return depthStencilFormat;
    }
  }

  // no matching found, choose the first supported format
  return !deviceDepthFormats_.empty() ? deviceDepthFormats_[0] : VK_FORMAT_D24_UNORM_S8_UINT;
}

VulkanContext::RenderPassHandle VulkanContext::getRenderPass(uint8_t index) const {
  return renderPasses_[index];
}

VulkanContext::RenderPassHandle VulkanContext::findRenderPass(
    const VulkanRenderPassBuilder& builder) const {
  IGL_PROFILER_FUNCTION();

  auto it = renderPassesHash_.find(builder);

  if (it != renderPassesHash_.end()) {
    return renderPasses_[it->second];
  }

  VulkanRenderPassBuilder compatibilityKey = builder.compatibilityKey();
  const auto compatIt = renderPassCompatHash_.find(compatibilityKey);
  const bool isNewCompatibilityClass = compatIt == renderPassCompatHash_.end();

  // both indices have to fit into 8 bits
  constexpr size_t kMaxIndex = std::numeric_limits<uint8_t>::max();
  const size_t index = renderPasses_.size();
  const size_t compatIndex = isNewCompatibilityClass ? renderPassCompatHash_.size()
                                                     : compatIt->second;
  // failures are not cached, so they would be logged for every render pass that is encoded
  if (index > kMaxIndex || compatIndex > kMaxIndex) {
    IGL_LOG_ERROR_ONCE(
        "Cannot create a Vulkan render pass: there are already %zu render passes and %zu "
        "classes of compatible render passes\n",
        renderPasses_.size(),
        renderPassCompatHash_.size());
    return {};
  }

  VkRenderPass pass = VK_NULL_HANDLE;
  const VkResult result = builder.build(vf_, vkDevice_, &pass);
  if (result != VK_SUCCESS || pass == VK_NULL_HANDLE) {
    IGL_LOG_ERROR_ONCE("Failed to create a Vulkan render pass: %d\n", static_cast<int>(result));
    if (pass != VK_NULL_HANDLE) {
      vf_.vkDestroyRenderPass(vkDevice_, pass, nullptr);
    }
    return {};
  }

  const RenderPassHandle handle = {
      .pass = pass,
      .index = static_cast<uint8_t>(index),
      .compatIndex = static_cast<uint8_t>(compatIndex),
      .numSubpasses = static_cast<uint8_t>(builder.getNumSubpasses()),
  };

  renderPassesHash_[builder] = handle.index;
  if (isNewCompatibilityClass) {
    renderPassCompatHash_[std::move(compatibilityKey)] = handle.compatIndex;
  }
  // @fb-only
  // @lint-ignore CLANGTIDY
  renderPasses_.push_back(handle);

  return handle;
}

std::vector<uint8_t> VulkanContext::getPipelineCacheData() const {
  size_t size = 0;
  vf_.vkGetPipelineCacheData(vkDevice_, pipelineCache_, &size, nullptr);

  std::vector<uint8_t> data(size);

  if (size) {
    vf_.vkGetPipelineCacheData(vkDevice_, pipelineCache_, &size, data.data());
  }

  return data;
}

uint64_t VulkanContext::getFrameNumber() const {
  return swapchain_ ? swapchain_->getFrameNumber() : 0u;
}

void VulkanContext::updateBindingsTextures(VkCommandBuffer IGL_NONNULL cmdBuf,
                                           VkPipelineLayout layout,
                                           VkPipelineBindPoint bindPoint,
                                           VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
                                           DescriptorPoolsArenas* IGL_NULLABLE arenas,
                                           const BindingsTextures& data,
                                           const VulkanDescriptorSetLayout& dsl,
                                           const util::SpvModuleInfo& info) const {
  IGL_PROFILER_FUNCTION();

  DescriptorPoolsArena& arena =
      getDescriptorPoolsArenas(arenas).getOrCreateArena_CombinedImageSamplers(
          *this, dsl.getVkDescriptorSetLayout(), dsl.numBindings);

  VkDescriptorSet dset = arena.getNextDescriptorSet(*immediate_, nextSubmitHandle);

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDescriptorImageInfo infoSampledImages[IGL_TEXTURE_SAMPLERS_MAX]; // uninitialized
  uint32_t numImages = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkWriteDescriptorSet writes[IGL_TEXTURE_SAMPLERS_MAX]; // uninitialized
  uint32_t numWrites = 0;

  // make sure the guard value is always there
  IGL_DEBUG_ASSERT(!textures_.objects_.empty());
  IGL_DEBUG_ASSERT(!samplers_.objects_.empty());

  // use the dummy texture/sampler to avoid sparse array
  VkImageView dummyImageView = textures_.objects_[0]->imageView_.getVkImageView();
  VkSampler dummySampler = samplers_.objects_[0].vkSampler;

  const bool isGraphics = bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS;

  for (const util::TextureDescription& d : info.textures) {
    IGL_DEBUG_ASSERT(d.descriptorSet == kBindPoint_CombinedImageSamplers);
    const uint32_t loc = d.bindingLocation;
    IGL_DEBUG_ASSERT(loc < IGL_TEXTURE_SAMPLERS_MAX);
    VkImageView texture = data.textures[loc];
    const bool hasTexture = texture != VK_NULL_HANDLE;
    if (hasTexture && isGraphics) {
      IGL_DEBUG_ASSERT(data.samplers[loc], "A sampler should be bound to every bound texture slot");
    }
    VkSampler sampler = data.samplers[loc] ? data.samplers[loc] : dummySampler;
    writes[numWrites++] = ivkGetWriteDescriptorSetImageInfo(
        dset, loc, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &infoSampledImages[numImages]);
    infoSampledImages[numImages++] = VkDescriptorImageInfo{
        .sampler = hasTexture ? sampler : dummySampler,
        .imageView = hasTexture ? texture : dummyImageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
  }

  if (numWrites) {
    IGL_PROFILER_ZONE("vkUpdateDescriptorSets()", IGL_PROFILER_COLOR_UPDATE);
    vf_.vkUpdateDescriptorSets(vkDevice_, numWrites, writes, 0, nullptr);
    IGL_PROFILER_ZONE_END();

#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkCmdBindDescriptorSets(%u) - textures\n", cmdBuf, bindPoint);
#endif // IGL_VULKAN_PRINT_COMMANDS
    vf_.vkCmdBindDescriptorSets(
        cmdBuf, bindPoint, layout, kBindPoint_CombinedImageSamplers, 1, &dset, 0, nullptr);
  }
}

void VulkanContext::updateBindingsStorageImages(
    VkCommandBuffer IGL_NONNULL cmdBuf,
    VkPipelineLayout layout,
    VkPipelineBindPoint bindPoint,
    VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
    DescriptorPoolsArenas* IGL_NULLABLE arenas,
    const BindingsStorageImages& data,
    const VulkanDescriptorSetLayout& dsl,
    const util::SpvModuleInfo& info) const {
  IGL_PROFILER_FUNCTION();

  DescriptorPoolsArena& arena = getDescriptorPoolsArenas(arenas).getOrCreateArena_StorageImages(
      *this,
      dsl.getVkDescriptorSetLayout(),
      static_cast<uint32_t>(info.images.size()),
      static_cast<uint32_t>(info.inputAttachments.size()));

  VkDescriptorSet dset = arena.getNextDescriptorSet(*immediate_, nextSubmitHandle);

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDescriptorImageInfo infoStorageImages[IGL_STORAGE_IMAGES_MAX]; // uninitialized
  uint32_t numStorageImages = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDescriptorImageInfo infoInputAttachments[IGL_COLOR_ATTACHMENTS_MAX]; // uninitialized
  uint32_t numInputAttachments = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkWriteDescriptorSet writes[IGL_STORAGE_IMAGES_MAX + IGL_COLOR_ATTACHMENTS_MAX]; // uninitialized
  uint32_t numWrites = 0;

  // make sure the guard value is always there
  IGL_DEBUG_ASSERT(!textures_.objects_.empty());

  // use the dummy texture to avoid sparse array
  VkImageView dummyImageView = textures_.objects_[0]->imageView_.getVkImageView();

  for (const util::ImageDescription& d : info.images) {
    IGL_DEBUG_ASSERT(d.descriptorSet == kBindPoint_StorageImages);
    const uint32_t loc = d.bindingLocation;
    IGL_DEBUG_ASSERT(loc < IGL_STORAGE_IMAGES_MAX);
    VkImageView imageView = data.images[loc];
    writes[numWrites++] = ivkGetWriteDescriptorSetImageInfo(
        dset, loc, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &infoStorageImages[numStorageImages]);
    infoStorageImages[numStorageImages++] = VkDescriptorImageInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = imageView ? imageView : dummyImageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
  }

  for (const util::InputAttachmentDescription& d : info.inputAttachments) {
    IGL_DEBUG_ASSERT(d.descriptorSet == kBindPoint_StorageImages);
    const uint32_t index = d.inputAttachmentIndex;
    if (index >= IGL_ARRAY_NUM_ELEMENTS(data.inputAttachments) ||
        numInputAttachments >= IGL_ARRAY_NUM_ELEMENTS(infoInputAttachments)) {
      IGL_LOG_ERROR_ONCE("updateBindingsStorageImages: cannot bind input attachment %u\n", index);
      continue;
    }
    VkImageView imageView = data.inputAttachments[index];
    if (!imageView) {
      IGL_LOG_ERROR_ONCE("Input attachment %u has no image view bound; using a dummy image view\n",
                         index);
      imageView = dummyImageView;
    }
    writes[numWrites++] =
        ivkGetWriteDescriptorSetImageInfo(dset,
                                          d.bindingLocation,
                                          VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                                          1,
                                          &infoInputAttachments[numInputAttachments]);
    infoInputAttachments[numInputAttachments++] = VkDescriptorImageInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = imageView,
        .imageLayout = kInputAttachmentImageLayout,
    };
  }

  if (numWrites) {
    IGL_PROFILER_ZONE("vkUpdateDescriptorSets()", IGL_PROFILER_COLOR_UPDATE);
    vf_.vkUpdateDescriptorSets(vkDevice_, numWrites, writes, 0, nullptr);
    IGL_PROFILER_ZONE_END();

#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkCmdBindDescriptorSets(%u) - storage images\n", cmdBuf, bindPoint);
#endif // IGL_VULKAN_PRINT_COMMANDS
    vf_.vkCmdBindDescriptorSets(
        cmdBuf, bindPoint, layout, kBindPoint_StorageImages, 1, &dset, 0, nullptr);
  }
}

void VulkanContext::updateBindingsBuffers(VkCommandBuffer IGL_NONNULL cmdBuf,
                                          VkPipelineLayout layout,
                                          VkPipelineBindPoint bindPoint,
                                          VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
                                          DescriptorPoolsArenas* IGL_NULLABLE arenas,
                                          BindingsBuffers& data,
                                          const VulkanDescriptorSetLayout& dsl,
                                          const util::SpvModuleInfo& info) const {
  IGL_PROFILER_FUNCTION();

  DescriptorPoolsArena& arena = getDescriptorPoolsArenas(arenas).getOrCreateArena_Buffers(
      *this, dsl.getVkDescriptorSetLayout(), dsl.numBindings);

  VkDescriptorSet dset = arena.getNextDescriptorSet(*immediate_, nextSubmitHandle);

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkWriteDescriptorSet writes[IGL_UNIFORM_BLOCKS_BINDING_MAX]; // uninitialized
  uint32_t numWrites = 0;

  for (const util::BufferDescription& b : info.buffers) {
    IGL_DEBUG_ASSERT(b.descriptorSet == kBindPoint_Buffers);
    if (numWrites >= IGL_ARRAY_NUM_ELEMENTS(writes)) {
      IGL_LOG_ERROR("updateBindingsBuffers: number of writes exceeds max %u\n",
                    (uint32_t)IGL_ARRAY_NUM_ELEMENTS(writes));
      break;
    }
    if (b.bindingLocation >= IGL_ARRAY_NUM_ELEMENTS(data.buffers)) {
      IGL_LOG_ERROR("updateBindingsBuffers: binding location %u out of range\n", b.bindingLocation);
      continue;
    }
    IGL_DEBUG_ASSERT(
        data.buffers[b.bindingLocation].buffer != VK_NULL_HANDLE,
        IGL_FORMAT("Did you forget to call bindBuffer() for a buffer at the binding location {}?",
                   b.bindingLocation)
            .c_str());
    writes[numWrites++] = ivkGetWriteDescriptorSetBufferInfo(
        dset,
        b.bindingLocation,
        b.isStorage ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        1,
        &data.buffers[b.bindingLocation]);
  }

  if (numWrites) {
    IGL_PROFILER_ZONE("vkUpdateDescriptorSets()", IGL_PROFILER_COLOR_UPDATE);
    vf_.vkUpdateDescriptorSets(vkDevice_, numWrites, writes, 0, nullptr);
    IGL_PROFILER_ZONE_END();

#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkCmdBindDescriptorSets(%u) - buffers\n", cmdBuf, bindPoint);
#endif // IGL_VULKAN_PRINT_COMMANDS
    vf_.vkCmdBindDescriptorSets(
        cmdBuf, bindPoint, layout, kBindPoint_Buffers, 1, &dset, 0, nullptr);
  }
}

bool VulkanContext::updateBindingsTexturesByDescriptorBuffer(
    VkCommandBuffer IGL_NONNULL cmdBuf,
    VkPipelineLayout layout,
    VkPipelineBindPoint bindPoint,
    VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
    const BindingsTextures& data,
    const VulkanDescriptorSetLayout& dsl,
    const util::SpvModuleInfo& info) const {
  if (info.textures.empty()) {
    return true;
  }

  // make sure the guard value is always there
  IGL_DEBUG_ASSERT(!textures_.objects_.empty());
  IGL_DEBUG_ASSERT(!samplers_.objects_.empty());

  // use the dummy texture/sampler to avoid sparse array
  VkImageView dummyImageView = textures_.objects_[0]->imageView_.getVkImageView();
  VkSampler dummySampler = samplers_.objects_[0].vkSampler;

  const bool isGraphics = bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS;

  auto combinedSize =
      vkPhysicalDeviceDescriptorBufferProperties_.combinedImageSamplerDescriptorSize;
  auto alignment = vkPhysicalDeviceDescriptorBufferProperties_.descriptorBufferOffsetAlignment;
  auto layoutSize = dsl.layoutSize;

  DescriptorBuffer* const descriptorBufferPtr = pimpl_->descriptorBuffersArena->getDescriptorBuffer(
      layoutSize, alignment, *immediate_, nextSubmitHandle);
  if (descriptorBufferPtr == nullptr) {
    IGL_LOG_ERROR_ONCE("No descriptor buffer is available: the bindings are not updated\n");
    return false;
  }
  DescriptorBuffer& descriptorBuffer = *descriptorBufferPtr;

  void* mappedPtr = descriptorBuffer.buffer->getMappedPtr();
  auto originOffset = descriptorBuffer.offset;
  descriptorBuffer.offset += layoutSize;

  for (const util::TextureDescription& d : info.textures) {
    IGL_DEBUG_ASSERT(d.descriptorSet == kBindPoint_CombinedImageSamplers);
    const uint32_t loc = d.bindingLocation;
    IGL_DEBUG_ASSERT(loc < IGL_TEXTURE_SAMPLERS_MAX);
    VkImageView texture = data.textures[loc];
    const bool hasTexture = texture != VK_NULL_HANDLE;
    if (hasTexture && isGraphics) {
      IGL_DEBUG_ASSERT(data.samplers[loc], "A sampler should be bound to every bound texture slot");
    }
    VkSampler sampler = data.samplers[loc] ? data.samplers[loc] : dummySampler;
    VkDescriptorImageInfo imageInfo{
        .sampler = hasTexture ? sampler : dummySampler,
        .imageView = hasTexture ? texture : dummyImageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDeviceSize bindingOffset = 0;
    vf_.vkGetDescriptorSetLayoutBindingOffsetEXT(
        getVkDevice(), dsl.getVkDescriptorSetLayout(), loc, &bindingOffset);

    VkDescriptorGetInfoEXT getInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .data = {.pCombinedImageSampler = &imageInfo},
    };

    vf_.vkGetDescriptorEXT(getVkDevice(),
                           &getInfo,
                           combinedSize,
                           static_cast<char*>(mappedPtr) + originOffset + bindingOffset);
  }

  if (descriptorBuffer.bindCmdBuffer != cmdBuf || descriptorBuffer.handle != nextSubmitHandle) {
    VkDescriptorBufferBindingInfoEXT bindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = descriptorBuffer.buffer->getVkDeviceAddress(),
        .usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                 VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    };
    vf_.vkCmdBindDescriptorBuffersEXT(cmdBuf, 1, &bindingInfo);
    descriptorBuffer.bindCmdBuffer = cmdBuf;
    descriptorBuffer.handle = nextSubmitHandle;
  }

  uint32_t bufferIndex = 0;
  VkDeviceSize offsets = originOffset;
  vf_.vkCmdSetDescriptorBufferOffsetsEXT(
      cmdBuf, bindPoint, layout, kBindPoint_CombinedImageSamplers, 1, &bufferIndex, &offsets);

  return true;
}

bool VulkanContext::updateBindingsStorageImagesByDescriptorBuffer(
    VkCommandBuffer IGL_NONNULL cmdBuf,
    VkPipelineLayout layout,
    VkPipelineBindPoint bindPoint,
    VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
    const BindingsStorageImages& data,
    const VulkanDescriptorSetLayout& dsl,
    const util::SpvModuleInfo& info) const {
  IGL_DEBUG_ASSERT(info.inputAttachments.empty(),
                   "Input attachments are not supported with VK_EXT_descriptor_buffer, "
                   "Device::createRenderPipeline() rejects such pipelines");
  if (info.images.empty()) {
    return true;
  }

  // make sure the guard value is always there
  IGL_DEBUG_ASSERT(!textures_.objects_.empty());

  // use the dummy texture to avoid sparse array
  VkImageView dummyImageView = textures_.objects_[0]->imageView_.getVkImageView();

  auto storageImageSize = vkPhysicalDeviceDescriptorBufferProperties_.storageImageDescriptorSize;
  auto alignment = vkPhysicalDeviceDescriptorBufferProperties_.descriptorBufferOffsetAlignment;
  auto layoutSize = dsl.layoutSize;

  DescriptorBuffer* const descriptorBufferPtr = pimpl_->descriptorBuffersArena->getDescriptorBuffer(
      layoutSize, alignment, *immediate_, nextSubmitHandle);
  if (descriptorBufferPtr == nullptr) {
    IGL_LOG_ERROR_ONCE("No descriptor buffer is available: the bindings are not updated\n");
    return false;
  }
  DescriptorBuffer& descriptorBuffer = *descriptorBufferPtr;

  void* mappedPtr = descriptorBuffer.buffer->getMappedPtr();
  auto originOffset = descriptorBuffer.offset;
  descriptorBuffer.offset += layoutSize;

  for (const util::ImageDescription& d : info.images) {
    IGL_DEBUG_ASSERT(d.descriptorSet == kBindPoint_StorageImages);
    const uint32_t loc = d.bindingLocation;
    IGL_DEBUG_ASSERT(loc < IGL_STORAGE_IMAGES_MAX);
    VkImageView imageView = data.images[loc];

    VkDescriptorImageInfo imageInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = imageView ? imageView : dummyImageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    VkDeviceSize bindingOffset = 0;
    vf_.vkGetDescriptorSetLayoutBindingOffsetEXT(
        getVkDevice(), dsl.getVkDescriptorSetLayout(), loc, &bindingOffset);

    VkDescriptorGetInfoEXT getInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .data = {.pStorageImage = &imageInfo},
    };

    vf_.vkGetDescriptorEXT(getVkDevice(),
                           &getInfo,
                           storageImageSize,
                           static_cast<char*>(mappedPtr) + originOffset + bindingOffset);
  }

  if (descriptorBuffer.bindCmdBuffer != cmdBuf || descriptorBuffer.handle != nextSubmitHandle) {
    VkDescriptorBufferBindingInfoEXT bindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = descriptorBuffer.buffer->getVkDeviceAddress(),
        .usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                 VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    };
    vf_.vkCmdBindDescriptorBuffersEXT(cmdBuf, 1, &bindingInfo);
    descriptorBuffer.bindCmdBuffer = cmdBuf;
    descriptorBuffer.handle = nextSubmitHandle;
  }

  uint32_t bufferIndex = 0;
  VkDeviceSize offsets = originOffset;
  vf_.vkCmdSetDescriptorBufferOffsetsEXT(
      cmdBuf, bindPoint, layout, kBindPoint_StorageImages, 1, &bufferIndex, &offsets);

  return true;
}

bool VulkanContext::updateBindingsBuffersByDescriptorBuffer(
    VkCommandBuffer IGL_NONNULL cmdBuf,
    VkPipelineLayout layout,
    VkPipelineBindPoint bindPoint,
    VulkanImmediateCommands::SubmitHandle nextSubmitHandle,
    BindingsBuffers& data,
    const VulkanDescriptorSetLayout& dsl,
    const util::SpvModuleInfo& info) {
  IGL_PROFILER_FUNCTION();
  if (info.buffers.empty()) {
    return true;
  }

  auto uniformBufferDescriptorSize =
      vkPhysicalDeviceDescriptorBufferProperties_.uniformBufferDescriptorSize;
  auto storageBufferDescriptorSize =
      vkPhysicalDeviceDescriptorBufferProperties_.storageBufferDescriptorSize;
  auto alignment = vkPhysicalDeviceDescriptorBufferProperties_.descriptorBufferOffsetAlignment;
  auto layoutSize = dsl.layoutSize;

  DescriptorBuffer* const descriptorBufferPtr = pimpl_->descriptorBuffersArena->getDescriptorBuffer(
      layoutSize, alignment, *immediate_, nextSubmitHandle);
  if (descriptorBufferPtr == nullptr) {
    IGL_LOG_ERROR_ONCE("No descriptor buffer is available: the bindings are not updated\n");
    return false;
  }
  DescriptorBuffer& descriptorBuffer = *descriptorBufferPtr;

  void* mappedPtr = descriptorBuffer.buffer->getMappedPtr();
  auto originOffset = descriptorBuffer.offset;
  descriptorBuffer.offset += layoutSize;

  for (const util::BufferDescription& b : info.buffers) {
    IGL_DEBUG_ASSERT(b.descriptorSet == kBindPoint_Buffers);
    IGL_DEBUG_ASSERT(
        data.buffers[b.bindingLocation].buffer != VK_NULL_HANDLE,
        IGL_FORMAT("Did you forget to call bindBuffer() for a buffer at the binding location {}?",
                   b.bindingLocation)
            .c_str());

    VkDeviceSize bindingOffset = 0;
    vf_.vkGetDescriptorSetLayoutBindingOffsetEXT(
        getVkDevice(), dsl.getVkDescriptorSetLayout(), b.bindingLocation, &bindingOffset);

    VkDescriptorAddressInfoEXT addressInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
        .address = data.addresses[b.bindingLocation],
        .range = data.buffers[b.bindingLocation].range,
    };

    VkDescriptorGetInfoEXT getInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
        .type = b.isStorage ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .data = {.pUniformBuffer = &addressInfo},
    };

    vf_.vkGetDescriptorEXT(getVkDevice(),
                           &getInfo,
                           b.isStorage ? storageBufferDescriptorSize : uniformBufferDescriptorSize,
                           static_cast<char*>(mappedPtr) + originOffset + bindingOffset);
  }

  if (descriptorBuffer.bindCmdBuffer != cmdBuf || descriptorBuffer.handle != nextSubmitHandle) {
    VkDescriptorBufferBindingInfoEXT bindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = descriptorBuffer.buffer->getVkDeviceAddress(),
        .usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                 VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    };
    vf_.vkCmdBindDescriptorBuffersEXT(cmdBuf, 1, &bindingInfo);
    descriptorBuffer.bindCmdBuffer = cmdBuf;
    descriptorBuffer.handle = nextSubmitHandle;
  }

  uint32_t bufferIndex = 0;
  VkDeviceSize offsets = originOffset;
  vf_.vkCmdSetDescriptorBufferOffsetsEXT(
      cmdBuf, bindPoint, layout, kBindPoint_Buffers, 1, &bufferIndex, &offsets);

  return true;
}

std::shared_ptr<DescriptorPoolsArenas> VulkanContext::createDescriptorPoolsArenas() const {
  return std::make_shared<DescriptorPoolsArenas>();
}

DescriptorPoolsArenas& VulkanContext::getDescriptorPoolsArenas(
    DescriptorPoolsArenas* IGL_NULLABLE arenas) const {
  if (arenas != nullptr) {
    return *arenas;
  }
  return pimpl_->arenas;
}

void VulkanContext::destroyRecordings(
    VkCommandPool commandPool,
    const std::vector<std::shared_ptr<DescriptorPoolsArenas>>& arenas,
    SubmitHandle lastSubmission) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);

  std::vector<VkDescriptorPool> pools;
  for (const std::shared_ptr<DescriptorPoolsArenas>& slotArenas : arenas) {
    if (slotArenas) {
      std::vector<VkDescriptorPool> slotPools = slotArenas->releasePools();
      pools.insert(pools.end(), slotPools.begin(), slotPools.end());
    }
  }
  // freeing the command pool frees the command buffers of the recordings
  std::packaged_task<void()> destroy(
      [vf = &vf_, device = getVkDevice(), commandPool, pools = std::move(pools)]() {
        if (commandPool != VK_NULL_HANDLE) {
          vf->vkDestroyCommandPool(device, commandPool, nullptr);
        }
        for (VkDescriptorPool pool : pools) {
          vf->vkDestroyDescriptorPool(device, pool, nullptr);
        }
      });
  if (immediate_->isReady(lastSubmission)) {
    destroy();
  } else {
    deferredTask(std::move(destroy), lastSubmission);
  }
}

void VulkanContext::journalImageLayout(VkCommandBuffer cmdBuf, const VulkanImage& image) const {
  for (ImageLayoutJournal* journal : imageLayoutJournals_) {
    const auto layout = std::find_if(
        journal->layouts.begin(),
        journal->layouts.end(),
        [&image](const ImageLayoutJournal::Layout& entry) { return entry.image == &image; });
    if (journal->cmdBuf == cmdBuf) {
      if (layout == journal->layouts.end()) {
        journal->layouts.push_back({.image = &image, .layout = image.imageLayout_});
      } else if (layout->isChangedElsewhere) {
        // the recording changes the layout that the change elsewhere left
        *layout = {.image = &image, .layout = image.imageLayout_};
      }
    } else if (layout != journal->layouts.end()) {
      layout->isChangedElsewhere = true;
    }
  }
}

bool VulkanContext::isRecordingSlot(VkCommandBuffer cmdBuf) const {
  return std::any_of(
      imageLayoutJournals_.begin(),
      imageLayoutJournals_.end(),
      [cmdBuf](const ImageLayoutJournal* journal) { return journal->cmdBuf == cmdBuf; });
}

void VulkanContext::deferredTask(std::packaged_task<void()>&& task, SubmitHandle handle) const {
  if (handle.empty()) {
    handle = immediate_->getNextSubmitHandle();
  }
  deferredTasks.emplace_back(std::move(task), handle);
  deferredTasks.back().frameId = this->getFrameNumber();
}

bool VulkanContext::areValidationLayersEnabled() const {
  return config_.enableValidation;
}

void* IGL_NULLABLE VulkanContext::getVmaAllocator() const {
  return pimpl_->vma;
}

void VulkanContext::processDeferredTasks() const {
  IGL_PROFILER_FUNCTION();

  const uint64_t frameId = getFrameNumber();
  constexpr uint64_t kNumWaitFrames = 3u;

  while (!deferredTasks.empty() && immediate_->isReady(deferredTasks.front().handle)) {
    if (frameId && frameId <= deferredTasks.front().frameId + kNumWaitFrames) {
      // do not check anything if it is not yet older than kNumWaitFrames
      break;
    }
    deferredTasks.front().task();
    deferredTasks.pop_front();
  }
}

void VulkanContext::waitDeferredTasks() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  for (auto& task : deferredTasks) {
    immediate_->wait(task.handle, config_.fenceTimeoutNanoseconds);
    task.task();
  }
  deferredTasks.clear();
}

// @fb-only
VkFence VulkanContext::getVkFenceFromSubmitHandle(igl::SubmitHandle handle) const noexcept {
  if (handle == 0) {
    IGL_LOG_ERROR("Invalid submit handle passed to getVkFenceFromSubmitHandle");
    return VK_NULL_HANDLE;
  }

  return immediate_->getVkFenceFromSubmitHandle(VulkanImmediateCommands::SubmitHandle(handle));
}

// @fb-only
int VulkanContext::getFenceFdFromSubmitHandle(igl::SubmitHandle handle) const noexcept {
  int fenceFd = -1;
#if defined(IGL_PLATFORM_ANDROID) && defined(VK_KHR_external_fence_fd)
  if (handle == 0) {
    IGL_LOG_ERROR("Invalid submit handle passed to getFenceFDFromSubmitHandle");
    return -1;
  }

  // Although the extension is present at compile time, software/emulated Vulkan
  // ICDs (e.g. SwiftShader) may not implement VK_KHR_external_fence_fd at
  // runtime, leaving vkGetFenceFdKHR() null. Return the no-fence sentinel (-1)
  // rather than dereferencing a null function pointer.
  if (!vf_.vkGetFenceFdKHR) {
    IGL_LOG_ERROR_ONCE("VK_KHR_external_fence_fd not loaded; vkGetFenceFdKHR is null\n");
    return -1;
  }

  const VkFence vkFence = getVkFenceFromSubmitHandle(handle);
  IGL_DEBUG_ASSERT(vkFence != VK_NULL_HANDLE);

  const VkFenceGetFdInfoKHR getFdInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
      .fence = vkFence,
      .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT,
  };

  const VkResult result = vf_.vkGetFenceFdKHR(vkDevice_, &getFdInfo, &fenceFd);
  if (result != VK_SUCCESS) {
    IGL_LOG_ERROR("Unable to get fence fd from submit handle: %lu", handle);
  }
  immediate_->storeFDInSubmitHandle(VulkanImmediateCommands::SubmitHandle(handle), fenceFd);
#endif // defined(IGL_PLATFORM_ANDROID)
  return fenceFd;
}

VkDescriptorSetLayout VulkanContext::getBindlessVkDescriptorSetLayout() const {
  return config_.enableDescriptorIndexing ? pimpl_->dslBindless->getVkDescriptorSetLayout()
                                          : VK_NULL_HANDLE;
}

VkDescriptorSet VulkanContext::getBindlessVkDescriptorSet() const {
  return config_.enableDescriptorIndexing ? pimpl_->dsBindless : VK_NULL_HANDLE;
}

VkSamplerYcbcrConversionInfo VulkanContext::getOrCreateYcbcrConversionInfo(VkFormat format) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  auto it = ycbcrConversionInfos_.find(format);

  if (it != ycbcrConversionInfos_.end()) {
    return it->second;
  }

  if (!IGL_DEBUG_VERIFY(features_.featuresSamplerYcbcrConversion.samplerYcbcrConversion)) {
    IGL_DEBUG_ABORT("Ycbcr samplers are not supported");
    return {};
  }

  VkFormatProperties props{};
  vf_.vkGetPhysicalDeviceFormatProperties(getVkPhysicalDevice(), format, &props);

  const bool cosited =
      (props.optimalTilingFeatures & VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT) != 0;
  const bool midpoint =
      (props.optimalTilingFeatures & VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT) != 0;

  if (!IGL_DEBUG_VERIFY(cosited || midpoint)) {
    IGL_DEBUG_ASSERT(cosited || midpoint, "Unsupported Ycbcr feature");
    return {};
  }

  const VkSamplerYcbcrConversionCreateInfo ciYcbcr = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
      .format = format,
      .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
      .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
      .components =
          {
              .r = VK_COMPONENT_SWIZZLE_IDENTITY,
              .g = VK_COMPONENT_SWIZZLE_IDENTITY,
              .b = VK_COMPONENT_SWIZZLE_IDENTITY,
              .a = VK_COMPONENT_SWIZZLE_IDENTITY,
          },
      .xChromaOffset = midpoint ? VK_CHROMA_LOCATION_MIDPOINT : VK_CHROMA_LOCATION_COSITED_EVEN,
      .yChromaOffset = midpoint ? VK_CHROMA_LOCATION_MIDPOINT : VK_CHROMA_LOCATION_COSITED_EVEN,
      .chromaFilter = VK_FILTER_LINEAR,
      .forceExplicitReconstruction = VK_FALSE,
  };

  VkSamplerYcbcrConversionInfo info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
      .conversion = VK_NULL_HANDLE,
  };
  vf_.vkCreateSamplerYcbcrConversion(getVkDevice(), &ciYcbcr, nullptr, &info.conversion);

  // Per IGL Vulkan rules, every Vk* handle (including YCbCr conversions) must
  // get a debug name immediately after creation so it shows up in RenderDoc /
  // validation output.
  VK_ASSERT(ivkSetDebugObjectName(
      &vf_,
      getVkDevice(),
      VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION,
      (uint64_t)info.conversion,
      IGL_FORMAT("YCbCr Conversion: VulkanContext::getOrCreateYcbcrConversionInfo() format={}",
                 static_cast<int>(format))
          .c_str()));

  // check properties
  VkSamplerYcbcrConversionImageFormatProperties samplerYcbcrConversionImageFormatProps = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
      .combinedImageSamplerDescriptorCount = 0,
  };
  VkImageFormatProperties2 imageFormatProps = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &samplerYcbcrConversionImageFormatProps,
      .imageFormatProperties = {},
  };
  const VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .flags = VK_IMAGE_CREATE_DISJOINT_BIT,
  };
  vf_.vkGetPhysicalDeviceImageFormatProperties2(
      getVkPhysicalDevice(), &imageFormatInfo, &imageFormatProps);

  IGL_DEBUG_ASSERT(samplerYcbcrConversionImageFormatProps.combinedImageSamplerDescriptorCount <= 3);

  ycbcrConversionInfos_[format] = info;

  return info;
}

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
VkSamplerYcbcrConversion VulkanContext::getOrCreateExternalYcbcrConversion(
    const VkSamplerYcbcrConversionCreateInfo& info) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  // `info.pNext` must carry a non-zero VkExternalFormatANDROID.
  uint64_t externalFormat = 0;
  for (const auto* p = static_cast<const VkBaseInStructure*>(info.pNext); p;
       p = static_cast<const VkBaseInStructure*>(p->pNext)) {
    if (p->sType == VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID) {
      externalFormat = reinterpret_cast<const VkExternalFormatANDROID*>(p)->externalFormat;
      break;
    }
  }
  IGL_DEBUG_ASSERT(externalFormat != 0,
                   "getOrCreateExternalYcbcrConversion called without VkExternalFormatANDROID "
                   "in pNext");

  // Compare every field that affects the conversion object.
  for (const auto& entry : externalYcbcrConversions_) {
    if (entry.externalFormat == externalFormat && entry.ycbcrModel == info.ycbcrModel &&
        entry.ycbcrRange == info.ycbcrRange && entry.xChromaOffset == info.xChromaOffset &&
        entry.yChromaOffset == info.yChromaOffset && entry.components.r == info.components.r &&
        entry.components.g == info.components.g && entry.components.b == info.components.b &&
        entry.components.a == info.components.a && entry.chromaFilter == info.chromaFilter &&
        entry.forceExplicitReconstruction == info.forceExplicitReconstruction) {
      return entry.conversion;
    }
  }

  VkSamplerYcbcrConversion conversion = VK_NULL_HANDLE;
  const VkResult result =
      vf_.vkCreateSamplerYcbcrConversion(getVkDevice(), &info, nullptr, &conversion);
  if (result != VK_SUCCESS || conversion == VK_NULL_HANDLE) {
    IGL_LOG_ERROR(
        "getOrCreateExternalYcbcrConversion(): vkCreateSamplerYcbcrConversion failed "
        "(result=%d) for externalFormat=%llu\n",
        static_cast<int>(result),
        static_cast<unsigned long long>(externalFormat));
    return VK_NULL_HANDLE;
  }
  VK_ASSERT(ivkSetDebugObjectName(
      &vf_,
      getVkDevice(),
      VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION,
      (uint64_t)conversion,
      IGL_FORMAT("YCbCr Conversion (external): externalFormat={}", externalFormat).c_str()));

  externalYcbcrConversions_.push_back(
      {.externalFormat = externalFormat,
       .ycbcrModel = info.ycbcrModel,
       .ycbcrRange = info.ycbcrRange,
       .xChromaOffset = info.xChromaOffset,
       .yChromaOffset = info.yChromaOffset,
       .components = info.components,
       .chromaFilter = info.chromaFilter,
       .forceExplicitReconstruction = info.forceExplicitReconstruction,
       .conversion = conversion});
  return conversion;
}
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

void VulkanContext::freeResourcesForDescriptorSetLayout(VkDescriptorSetLayout /*dsl*/) const {
  // Deprecated: VkDescriptorSetLayout handles are now shared/de-duplicated through the
  // dslCache in VulkanContextImpl and their arenas are shared per-handle. Freeing arenas by
  // dsl handle at pipeline teardown is unsafe because other pipelines may still be using the
  // same shared handle. Cached dsl handles and their arenas are cleaned up when the
  // VulkanContext itself is destroyed.
}

// @fb-only
VkDescriptorSetLayout VulkanContext::getOrCreateVkDescriptorSetLayout(
    VkDescriptorSetLayoutCreateFlags flags,
    uint32_t numBindings,
    const VkDescriptorSetLayoutBinding* bindings,
    const VkDescriptorBindingFlags* bindingFlags,
    const char* debugName) const {
  DescriptorSetLayoutCacheKey key;
  key.flags = flags;
  IGL_DEBUG_ASSERT(bindings || numBindings == 0);
  // @fb-only
  key.bindings.assign(bindings, bindings + numBindings);
  if (bindingFlags) {
    key.bindingFlags.assign(bindingFlags, bindingFlags + numBindings);
  }

  auto it = pimpl_->dslCache.find(key);
  if (it != pimpl_->dslCache.end()) {
    return it->second;
  }

  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VK_ASSERT(ivkCreateDescriptorSetLayout(
      &vf_, getVkDevice(), flags, numBindings, bindings, bindingFlags, &dsl));
  VK_ASSERT(ivkSetDebugObjectName(&vf_,
                                  getVkDevice(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                  reinterpret_cast<uint64_t>(dsl),
                                  debugName));
  pimpl_->dslCache.emplace(std::move(key), dsl);
  return dsl;
}

BindGroupTextureHandle VulkanContext::createBindGroup(const BindGroupTextureDesc& desc,
                                                      const IRenderPipelineState* IGL_NULLABLE
                                                          compatiblePipeline,
                                                      Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  VkDevice device = getVkDevice();

  BindGroupMetadataTextures metadata{.desc = desc};

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDescriptorSetLayoutBinding bindings[IGL_TEXTURE_SAMPLERS_MAX]; // uninitialized
  uint32_t numBindings = 0;

  const VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  // The `*compatiblePipeline` dereference is inside the `compatiblePipeline ? ... : 0UL` ternary,
  // so it only runs when the pointer is non-null; clang-tidy does not credit the multi-line
  // ternary guard, so these are false positives.
  // NOLINTBEGIN(facebook-hte-NullableDereference)
  const uint32_t usageMaskPipeline =
      compatiblePipeline ? static_cast<const RenderPipelineState&>(*compatiblePipeline)
                               .getSpvModuleInfo()
                               .usageMaskTextures
                         : 0UL;
  // NOLINTEND(facebook-hte-NullableDereference)

  for (uint32_t loc = 0; loc != IGL_ARRAY_NUM_ELEMENTS(desc.textures); loc++) {
    const bool isInPipeline = (usageMaskPipeline & (1UL << loc)) != 0;
    if (compatiblePipeline ? isInPipeline : desc.samplers[loc] != nullptr) {
      IGL_DEBUG_ASSERT(compatiblePipeline || desc.samplers[loc]);
      bindings[numBindings++] = VkDescriptorSetLayoutBinding{
          .binding = loc,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1,
          .stageFlags = stageFlags,
      };
      metadata.usageMask |= 1UL << loc;
    }
  }

  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;

  {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const VkDescriptorBindingFlags bindingFlags[IGL_TEXTURE_SAMPLERS_MAX] = {};

    VK_ASSERT(ivkCreateDescriptorSetLayout(&vf_,
                                           device,
                                           VkDescriptorSetLayoutCreateFlags{},
                                           numBindings,
                                           bindings,
                                           bindingFlags,
                                           &dsl));
    VK_ASSERT(ivkSetDebugObjectName(
        &vf_,
        device,
        VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
        (uint64_t)dsl,
        IGL_FORMAT("Descriptor Set Layout (COMBINED_IMAGE_SAMPLER): BindGroup = {}", desc.debugName)
            .c_str()));

    const VkDescriptorPoolSize poolSize = VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = numBindings};

    VK_ASSERT(ivkCreateDescriptorPool(
        &vf_, device, VkDescriptorPoolCreateFlags{}, 1u, 1u, &poolSize, &metadata.pool));
    VK_ASSERT(ivkSetDebugObjectName(
        &vf_,
        device,
        VK_OBJECT_TYPE_DESCRIPTOR_POOL,
        (uint64_t)metadata.pool,
        IGL_FORMAT("Descriptor Pool (COMBINED_IMAGE_SAMPLER): BindGroup = {}", desc.debugName)
            .c_str()));

    VK_ASSERT(ivkAllocateDescriptorSet(&vf_, device, metadata.pool, dsl, &metadata.dset));
  }

  // make sure the guard values are always there
  IGL_DEBUG_ASSERT(!textures_.objects_.empty());
  IGL_DEBUG_ASSERT(!samplers_.objects_.empty());
  // use the dummy texture to ensure pipeline compatibility
  VkImageView dummyImageView = textures_.objects_[0]->imageView_.getVkImageView();

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDescriptorImageInfo images[IGL_TEXTURE_SAMPLERS_MAX]; // uninitialized
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkWriteDescriptorSet writes[IGL_TEXTURE_SAMPLERS_MAX]; // uninitialized
  uint32_t numWrites = 0;

  for (uint32_t loc = 0; loc != IGL_ARRAY_NUM_ELEMENTS(desc.textures); loc++) {
    if (compatiblePipeline ? (usageMaskPipeline & (1UL << loc)) == 0 : !desc.textures[loc]) {
      continue;
    }
    const VulkanTexture& texture =
        desc.textures[loc] ? static_cast<Texture*>(desc.textures[loc].get())->getVulkanTexture()
                           : *textures_.objects_[0]; // use a dummy texture when necessary
    const VulkanSampler& sampler =
        desc.samplers[loc]
            ? *samplers_.get(static_cast<SamplerState&>(*desc.samplers[loc]).sampler_)
            : samplers_.objects_[0]; // use a dummy sampler when necessary

    // multisampled images cannot be directly accessed from shaders
    const bool isTextureAvailable =
        (texture.image.samples_ & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT;
    const bool isSampledImage = isTextureAvailable && texture.image.isSampledImage();

    if (!IGL_DEBUG_VERIFY(isSampledImage)) {
      IGL_LOG_ERROR("Each bound texture should have TextureUsageBits::Sampled (slot = %u)", loc);
      continue;
    }

    writes[numWrites] = ivkGetWriteDescriptorSetImageInfo(
        metadata.dset, loc, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &images[numWrites]);
    images[numWrites++] = {
        .sampler = sampler.vkSampler,
        .imageView = isSampledImage ? texture.imageView_.getVkImageView() : dummyImageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
  }

  if (!IGL_DEBUG_VERIFY(numWrites)) {
    IGL_LOG_ERROR("Cannot create an empty bind group");
    Result::setResult(outResult,
                      Result(Result::Code::RuntimeError, "Cannot create an empty bind group"));
    return {};
  }

  IGL_PROFILER_ZONE("vkUpdateDescriptorSets() - textures bind group", IGL_PROFILER_COLOR_UPDATE);
  vf_.vkUpdateDescriptorSets(vkDevice_, numWrites, writes, 0, nullptr);
  IGL_PROFILER_ZONE_END();

  // once a descriptor set has been updated, destroy the DSL
  vf_.vkDestroyDescriptorSetLayout(device, dsl, nullptr);

  Result::setOk(outResult);

  return pimpl_->bindGroupTexturesPool.create(std::move(metadata));
}

BindGroupBufferHandle VulkanContext::createBindGroup(const BindGroupBufferDesc& desc,
                                                     Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  VkDevice device = getVkDevice();

  BindGroupMetadataBuffers metadata{.desc = desc};

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDescriptorSetLayoutBinding bindings[IGL_UNIFORM_BLOCKS_BINDING_MAX]; // uninitialized
  uint32_t numBindings = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDescriptorPoolSize poolSizes[] = {
      {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 0},
      {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 0},
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, .descriptorCount = 0},
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 0},
  };

  const VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  for (uint32_t loc = 0; loc != IGL_ARRAY_NUM_ELEMENTS(desc.buffers); loc++) {
    if (!desc.buffers[loc]) {
      continue;
    }
    auto* buf = static_cast<Buffer*>(desc.buffers[loc].get());
    // The descriptor set is written once, below, with the Ring buffer copy current at creation.
    // Uploads go to the other copies and reads through the bind group are not tracked, so Ring
    // buffers must be bound with bindBuffer() instead.
    if (!IGL_DEBUG_VERIFY(!buf->isRingBuffer(),
                          "Ring buffers are not supported in bind groups (binding location %u)",
                          loc)) {
      IGL_LOG_ERROR_ONCE("[IGL] Ring buffers are not supported in bind groups; use bindBuffer()\n");
      Result::setResult(outResult,
                        Result::Code::ArgumentInvalid,
                        "Ring buffers are not supported in bind groups");
      return {};
    }
    const bool isDynamic = (desc.isDynamicBufferMask & (1UL << loc)) != 0;
    const bool isUniform = ((buf->getBufferType() & BufferDesc::BufferTypeBits::Uniform) != 0);
    const VkDescriptorType type =
        isUniform
            ? (isDynamic
                   ? (poolSizes[0].descriptorCount++, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
                   : (poolSizes[1].descriptorCount++, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER))
            : (isDynamic
                   ? (poolSizes[2].descriptorCount++, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                   : (poolSizes[3].descriptorCount++, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    if (isDynamic && !desc.size[loc]) {
      IGL_LOG_ERROR(
          "A buffer at the binding location '%u' is marked as dynamic but the corresponding size "
          "value is 0. You have to specify the binding size for all dynamic buffers.",
          loc);
    }
    if (desc.offset[loc]) {
      const auto& limits = getVkPhysicalDeviceProperties().limits;
      const uint32_t alignment =
          static_cast<uint32_t>(isUniform ? limits.minUniformBufferOffsetAlignment
                                          : limits.minStorageBufferOffsetAlignment);
      if (!IGL_DEBUG_VERIFY((alignment == 0) || (desc.offset[loc] % alignment == 0))) {
        IGL_LOG_ERROR(
            "`desc.offset[loc] = %u` must be a multiple of `VkPhysicalDeviceLimits::%s = %u`",
            static_cast<uint32_t>(desc.offset[loc]),
            isUniform ? "minUniformBufferOffsetAlignment" : "minStorageBufferOffsetAlignment",
            alignment);
      }
    }
    bindings[numBindings++] = VkDescriptorSetLayoutBinding{
        .binding = loc,
        .descriptorType = type,
        .descriptorCount = 1,
        .stageFlags = stageFlags,
    };
    metadata.usageMask |= 1UL << loc;
  }

  // construct a dense array of non-zero VkDescriptorPoolSize elements
  qsort(poolSizes,
        IGL_ARRAY_NUM_ELEMENTS(poolSizes),
        sizeof(VkDescriptorPoolSize),
        [](const void* a, const void* b) {
          return static_cast<const VkDescriptorPoolSize*>(a)->descriptorCount <
                         static_cast<const VkDescriptorPoolSize*>(b)->descriptorCount
                     ? 1
                     : 0;
        });
  uint32_t numPoolSizes = 0;
  while (numPoolSizes < IGL_ARRAY_NUM_ELEMENTS(poolSizes) &&
         poolSizes[numPoolSizes].descriptorCount > 0) {
    numPoolSizes++;
  }
  IGL_DEBUG_ASSERT(numPoolSizes);

  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;

  {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const VkDescriptorBindingFlags bindingFlags[IGL_UNIFORM_BLOCKS_BINDING_MAX] = {};

    VK_ASSERT(ivkCreateDescriptorSetLayout(&vf_,
                                           device,
                                           VkDescriptorSetLayoutCreateFlags{},
                                           numBindings,
                                           bindings,
                                           bindingFlags,
                                           &dsl));
    VK_ASSERT(ivkSetDebugObjectName(
        &vf_,
        device,
        VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
        (uint64_t)dsl,
        IGL_FORMAT("Descriptor Set Layout (BUFFERS): BindGroup = {}", desc.debugName).c_str()));

    VK_ASSERT(ivkCreateDescriptorPool(
        &vf_, device, VkDescriptorPoolCreateFlags{}, 1u, numPoolSizes, poolSizes, &metadata.pool));
    VK_ASSERT(ivkSetDebugObjectName(
        &vf_,
        device,
        VK_OBJECT_TYPE_DESCRIPTOR_POOL,
        (uint64_t)metadata.pool,
        IGL_FORMAT("Descriptor Pool (BUFFERS): BindGroup = {}", desc.debugName).c_str()));

    VK_ASSERT(ivkAllocateDescriptorSet(&vf_, device, metadata.pool, dsl, &metadata.dset));
  }

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkDescriptorBufferInfo buffers[IGL_UNIFORM_BLOCKS_BINDING_MAX]; // uninitialized
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  VkWriteDescriptorSet writes[IGL_UNIFORM_BLOCKS_BINDING_MAX]; // uninitialized
  uint32_t numWrites = 0;

  for (uint32_t loc = 0; loc != IGL_ARRAY_NUM_ELEMENTS(desc.buffers); loc++) {
    if (!desc.buffers[loc]) {
      continue;
    }
    auto* buf = static_cast<Buffer*>(desc.buffers[loc].get());
    const bool isDynamic = (desc.isDynamicBufferMask & (1UL << loc)) != 0;
    const bool isUniform = ((buf->getBufferType() & BufferDesc::BufferTypeBits::Uniform) != 0);
    const VkDescriptorType type = isUniform ? (isDynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                                         : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                                            : (isDynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                                         : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    writes[numWrites] =
        ivkGetWriteDescriptorSetBufferInfo(metadata.dset, loc, type, 1, &buffers[numWrites]);
    buffers[numWrites++] = VkDescriptorBufferInfo{
        .buffer = buf->getVkBuffer(),
        .offset = desc.offset[loc],
        .range = desc.size[loc] ? desc.size[loc] : VK_WHOLE_SIZE,
    };
  }

  if (!IGL_DEBUG_VERIFY(numWrites)) {
    IGL_LOG_ERROR("Cannot create an empty bind group");
    Result::setResult(outResult,
                      Result(Result::Code::RuntimeError, "Cannot create an empty bind group"));
    return {};
  }

  IGL_PROFILER_ZONE("vkUpdateDescriptorSets() - textures bind group", IGL_PROFILER_COLOR_UPDATE);
  vf_.vkUpdateDescriptorSets(vkDevice_, numWrites, writes, 0, nullptr);
  IGL_PROFILER_ZONE_END();

  // once a descriptor set has been updated, destroy the DSL
  vf_.vkDestroyDescriptorSetLayout(vkDevice_, dsl, nullptr);

  Result::setOk(outResult);

  return pimpl_->bindGroupBuffersPool.create(std::move(metadata));
}

void VulkanContext::destroy(BindGroupTextureHandle handle) {
  if (handle.empty()) {
    return;
  }

  deferredTask(std::packaged_task<void()>(
      [vf = &vf_, device = getVkDevice(), pool = pimpl_->bindGroupTexturesPool.get(handle)->pool] {
        vf->vkDestroyDescriptorPool(device, pool, nullptr);
      }));

  pimpl_->bindGroupTexturesPool.destroy(handle);
}

void VulkanContext::destroy(BindGroupBufferHandle handle) {
  if (handle.empty()) {
    return;
  }

  deferredTask(std::packaged_task<void()>(
      [vf = &vf_, device = getVkDevice(), pool = pimpl_->bindGroupBuffersPool.get(handle)->pool] {
        vf->vkDestroyDescriptorPool(device, pool, nullptr);
      }));

  pimpl_->bindGroupBuffersPool.destroy(handle);
}

void VulkanContext::destroy(SamplerHandle handle) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);

  if (handle.empty()) {
    return;
  }

  deferredTask(std::packaged_task<void()>(
      [vf = &vf_, device = getVkDevice(), sampler = samplers_.get(handle)->vkSampler]() {
        vf->vkDestroySampler(device, sampler, nullptr);
      }));

  samplers_.destroy(handle);
}

void VulkanContext::destroy(TextureHandle handle) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);

  if (handle.empty()) {
    return;
  }

  textures_.destroy(handle);
}

VkDescriptorSet VulkanContext::getBindGroupDescriptorSet(BindGroupTextureHandle handle) const {
  return handle.valid() ? pimpl_->bindGroupTexturesPool.get(handle)->dset : VK_NULL_HANDLE;
}

uint32_t VulkanContext::getBindGroupUsageMask(BindGroupTextureHandle handle) const {
  return handle.valid() ? pimpl_->bindGroupTexturesPool.get(handle)->usageMask : 0;
}

VkDescriptorSet VulkanContext::getBindGroupDescriptorSet(BindGroupBufferHandle handle) const {
  return handle.valid() ? pimpl_->bindGroupBuffersPool.get(handle)->dset : VK_NULL_HANDLE;
}

uint32_t VulkanContext::getBindGroupUsageMask(BindGroupBufferHandle handle) const {
  return handle.valid() ? pimpl_->bindGroupBuffersPool.get(handle)->usageMask : 0;
}

const VulkanFeatures& VulkanContext::features() const noexcept {
  return features_;
}

bool VulkanContext::supportsLazilyAllocatedMemory() const {
  return (resourceStorageToVkMemoryPropertyFlags(ResourceStorage::Memoryless, &memoryProperties) &
          VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0 &&
         !features_.enabled("VK_KHR_portability_subset");
}

bool VulkanContext::supportsSubpasses() const {
  // The specification requires every device to have at least 4 color attachments
  // (maxColorAttachments) and 4 input attachment descriptors per shader stage
  // (maxPerStageDescriptorInputAttachments), so there are no limits to check
  static_assert(IGL_COLOR_ATTACHMENTS_MAX <= 4);
  return vf_.vkCreateRenderPass2 != nullptr && !features_.has_VK_EXT_descriptor_buffer &&
         !features_.enabled("VK_KHR_portability_subset");
}

bool VulkanContext::supportsMultiviewSubpasses() const {
  return supportsSubpasses() && features_.featuresMultiview.multiview == VK_TRUE;
}

bool VulkanContext::supportsTimers() const {
  const VkPhysicalDeviceLimits& limits = getVkPhysicalDeviceProperties().limits;
  return limits.timestampPeriod > 0.0f && limits.timestampComputeAndGraphics == VK_TRUE &&
         getGraphicsQueueTimestampValidBits() > 0 &&
         !features_.enabled("VK_KHR_portability_subset");
}

bool VulkanContext::supportsReusableCommandBuffers() const {
  return !config_.enableDescriptorIndexing && !features_.has_VK_EXT_descriptor_buffer;
}

void VulkanContext::syncAcquireNext() noexcept {
  IGL_PROFILER_FUNCTION();

  syncCurrentIndex = (syncCurrentIndex + 1) % config_.maxResourceCount;

  // Wait for the current buffer to become available
  immediate_->wait(syncSubmitHandles[syncCurrentIndex], config_.fenceTimeoutNanoseconds);
}

void VulkanContext::syncMarkSubmitted(VulkanImmediateCommands::SubmitHandle handle) noexcept {
  IGL_PROFILER_FUNCTION();

  // The sync index selects the Ring buffer copy that GPU work reads and Buffer::upload() writes.
  // A copy must not be written while a submit that read it is in flight, so only submits that read
  // a Ring buffer copy record their handle and move to the next copy; other submits change nothing.
  if (!ringBufferReadSinceSubmit) {
    return;
  }
  ringBufferReadSinceSubmit = false;

  syncSubmitHandles[syncCurrentIndex] = handle;

  syncAcquireNext();
}

void VulkanContext::ensureCurrentContextThread() const {
  IGL_DEBUG_ASSERT(
      pimpl_->contextThread == std::this_thread::get_id(),
      "IGL/Vulkan functions can only be accessed by 1 thread at a time. Call "
      "`setCurrentContextThread()` to mark the current thread as the `owning` thread.");
}

void VulkanContext::setCurrentContextThread() {
  pimpl_->contextThread = std::this_thread::get_id();
}

bool VulkanContext::isDeviceLost() const noexcept {
  if (deviceLost_.load(std::memory_order_relaxed)) {
    return true;
  }
  const bool lost =
      (immediate_ && immediate_->isDeviceLost()) ||
      (stagingDevice_ && stagingDevice_->immediate && stagingDevice_->immediate->isDeviceLost());
  if (lost) {
    // Device loss is permanent, so later calls can skip the queue checks.
    deviceLost_.store(true, std::memory_order_relaxed);
  }
  return lost;
}

} // namespace igl::vulkan
