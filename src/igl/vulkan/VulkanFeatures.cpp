/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/VulkanFeatures.h>

#include <algorithm>
#include <cstring>
#include <igl/vulkan/VulkanContext.h>

namespace igl::vulkan {

namespace {
void clearFeaturePayload(VkBaseOutStructure* IGL_NONNULL structure, size_t size) noexcept {
  std::memset(reinterpret_cast<uint8_t*>(structure) + sizeof(VkBaseOutStructure),
              0,
              size - sizeof(VkBaseOutStructure));
}
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
VulkanFeatures::VulkanFeatures(VulkanContextConfig config) noexcept :
  // Vulkan 1.1
  vkPhysicalDeviceFeatures2({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .features =
          {
              .dualSrcBlend = config.enableDualSrcBlend ? VK_TRUE : VK_FALSE,
              .multiDrawIndirect = VK_TRUE,
              .drawIndirectFirstInstance = VK_TRUE,
              .depthBiasClamp = VK_TRUE,
#if IGL_PLATFORM_ANDROID
              .fillModeNonSolid = VK_FALSE, // not well supported on Android
#else
              .fillModeNonSolid = VK_TRUE,
#endif
              .shaderInt16 = config.enableShaderInt16 ? VK_TRUE : VK_FALSE,
          },
  }),
  featuresSamplerYcbcrConversion({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
      .samplerYcbcrConversion = VK_TRUE,
  }),
  featuresShaderDrawParameters({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
      .shaderDrawParameters = config.enableShaderDrawParameters ? VK_TRUE : VK_FALSE,
  }),
  featuresMultiview({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
      .multiview = VK_TRUE,
      .multiviewGeometryShader = VK_FALSE,
      .multiviewTessellationShader = VK_FALSE,
  }),
  featuresBufferDeviceAddress({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
      .bufferDeviceAddress = VK_TRUE,
      .bufferDeviceAddressCaptureReplay = VK_FALSE,
      .bufferDeviceAddressMultiDevice = VK_FALSE,
  }),
  featuresDescriptorIndexing({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
      .shaderInputAttachmentArrayDynamicIndexing = VK_FALSE,
      .shaderUniformTexelBufferArrayDynamicIndexing = VK_FALSE,
      .shaderStorageTexelBufferArrayDynamicIndexing = VK_FALSE,
      .shaderUniformBufferArrayNonUniformIndexing = VK_FALSE,
      .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
      .shaderStorageBufferArrayNonUniformIndexing = VK_FALSE,
      .shaderStorageImageArrayNonUniformIndexing = VK_FALSE,
      .shaderInputAttachmentArrayNonUniformIndexing = VK_FALSE,
      .shaderUniformTexelBufferArrayNonUniformIndexing = VK_FALSE,
      .shaderStorageTexelBufferArrayNonUniformIndexing = VK_FALSE,
      .descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE,
      .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
      .descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
      .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
      .descriptorBindingUniformTexelBufferUpdateAfterBind = VK_FALSE,
      .descriptorBindingStorageTexelBufferUpdateAfterBind = VK_FALSE,
      .descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
      .descriptorBindingPartiallyBound = VK_TRUE,
      .descriptorBindingVariableDescriptorCount = VK_FALSE,
      .runtimeDescriptorArray = VK_TRUE,
  }),
  features16BitStorage({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
      .storageBuffer16BitAccess = config.enableStorageBuffer16BitAccess ? VK_TRUE : VK_FALSE,
      .uniformAndStorageBuffer16BitAccess = VK_FALSE,
      .storagePushConstant16 = VK_FALSE,
      .storageInputOutput16 = VK_FALSE,
  }),
  // Vulkan 1.2
  featuresShaderFloat16Int8({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
      .shaderFloat16 = VK_FALSE,
      .shaderInt8 = VK_FALSE,
  }),
  featuresIndexTypeUint8({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT,
      .indexTypeUint8 = VK_FALSE,
  }),
  featuresSynchronization2({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
      .synchronization2 = VK_TRUE,
  }),
  featuresTimelineSemaphore({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR,
      .timelineSemaphore = VK_TRUE,
  }),
  featuresFragmentDensityMap({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT,
      .fragmentDensityMap = VK_TRUE,
  }),
  featuresVulkanMemoryModel({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR,
      .vulkanMemoryModel = VK_TRUE,
  }),
  features8BitStorage({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
      .storageBuffer8BitAccess = VK_TRUE,
      .uniformAndStorageBuffer8BitAccess = VK_FALSE,
      .storagePushConstant8 = VK_FALSE,
  }),
  featuresUniformBufferStandardLayout({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR,
      .uniformBufferStandardLayout = VK_TRUE,
  }),
  featuresScalarBlockLayout({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT,
      .scalarBlockLayout = VK_TRUE,
  }),
  featuresMultiviewPerViewViewports({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_VIEWPORTS_FEATURES_QCOM,
      .multiviewPerViewViewports = VK_TRUE,
  }),
  featuresMeshShader({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .taskShader = VK_TRUE,
      .meshShader = VK_TRUE,
  }),
  featuresFragmentShadingRate({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
      .pipelineFragmentShadingRate = VK_FALSE,
      .primitiveFragmentShadingRate = VK_FALSE,
      .attachmentFragmentShadingRate = VK_FALSE,
  }),
  featuresDescriptorBuffer({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
      .descriptorBuffer = VK_TRUE,
  }),
  featuresTextureCompressionAstcHdr({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT,
  }),
  featuresShaderIntegerDotProduct({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
      .shaderIntegerDotProduct = VK_FALSE,
  }),
  featuresExtendedDynamicState({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
      .extendedDynamicState = VK_TRUE,
  }),
  featuresExtendedDynamicState2({
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT,
      .extendedDynamicState2 = VK_TRUE,
  }),
  config(config) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  extensions_.resize(kNumberOfExtensionTypes);
  enabledExtensions_.resize(kNumberOfExtensionTypes);

  // All the above get assembled into a feature chain
  assembleFeatureChain(config);
}

VulkanFeatures::VulkanFeatures(const VulkanFeatures& other) noexcept : config(other.config) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  *this = other;
  has_VK_EXT_descriptor_buffer = other.has_VK_EXT_descriptor_buffer;
  has_VK_EXT_descriptor_indexing = other.has_VK_EXT_descriptor_indexing;
  has_VK_EXT_fragment_density_map = other.has_VK_EXT_fragment_density_map;
  has_VK_EXT_headless_surface = other.has_VK_EXT_headless_surface;
  has_VK_EXT_index_type_uint8 = other.has_VK_EXT_index_type_uint8;
  has_VK_EXT_mesh_shader = other.has_VK_EXT_mesh_shader;
  has_VK_EXT_queue_family_foreign = other.has_VK_EXT_queue_family_foreign;
  has_VK_EXT_scalar_block_layout = other.has_VK_EXT_scalar_block_layout;
  has_VK_KHR_8bit_storage = other.has_VK_KHR_8bit_storage;
  has_VK_KHR_buffer_device_address = other.has_VK_KHR_buffer_device_address;
  has_VK_KHR_create_renderpass2 = other.has_VK_KHR_create_renderpass2;
  has_VK_KHR_get_surface_capabilities2 = other.has_VK_KHR_get_surface_capabilities2;
  has_VK_KHR_portability_enumeration = other.has_VK_KHR_portability_enumeration;
  has_VK_KHR_shader_non_semantic_info = other.has_VK_KHR_shader_non_semantic_info;
  has_VK_KHR_synchronization2 = other.has_VK_KHR_synchronization2;
  has_VK_KHR_timeline_semaphore = other.has_VK_KHR_timeline_semaphore;
  has_VK_KHR_uniform_buffer_standard_layout = other.has_VK_KHR_uniform_buffer_standard_layout;
  has_VK_KHR_vulkan_memory_model = other.has_VK_KHR_vulkan_memory_model;
  has_VK_QCOM_multiview_per_view_viewports = other.has_VK_QCOM_multiview_per_view_viewports;
  has_VK_EXT_extended_dynamic_state = other.has_VK_EXT_extended_dynamic_state;
  has_VK_EXT_extended_dynamic_state2 = other.has_VK_EXT_extended_dynamic_state2;
}

void VulkanFeatures::requestFeature(const void* feature,
                                    size_t size,
                                    const char* extensionName,
                                    uint32_t coreVersion) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const auto* header = static_cast<const VkBaseInStructure*>(feature);
  if (size <= sizeof(VkBaseOutStructure) ||
      header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES ||
      header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES ||
      header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
#if defined(VK_VERSION_1_4)
      || header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES
#endif
  ) {
    IGL_LOG_ERROR("Ignoring invalid or aggregate Vulkan feature structure %u\n",
                  static_cast<uint32_t>(header->sType));
    return;
  }
  const size_t words = (size + sizeof(FeatureWord) - 1) / sizeof(FeatureWord);
  FeatureRequest request{.extensionName = extensionName,
                         .coreVersion = coreVersion,
                         .size = size,
                         .requested = std::vector<FeatureWord>(words),
                         .selected = std::vector<FeatureWord>(words)};
  std::memcpy(request.requested.data(), feature, size);
  reinterpret_cast<VkBaseOutStructure*>(request.requested.data())->pNext = nullptr;
  reinterpret_cast<VkBaseOutStructure*>(request.selected.data())->sType = header->sType;
  featureRequests_.push_back(std::move(request));
}

const void* VulkanFeatures::feature(VkStructureType type) const noexcept {
  const auto* current = reinterpret_cast<const VkBaseOutStructure*>(&vkPhysicalDeviceFeatures2);
  for (; current; current = current->pNext) {
    if (current->sType == type) {
      return current;
    }
  }
  return nullptr;
}

std::array<VulkanFeatures::FeatureStructure, 24> VulkanFeatures::featureStructures() noexcept {
  const auto entry = [](auto& feature) {
    return FeatureStructure{.structure = reinterpret_cast<VkBaseOutStructure*>(&feature),
                            .size = sizeof(feature)};
  };
  return {entry(vkPhysicalDeviceFeatures2),
          entry(featuresSamplerYcbcrConversion),
          entry(featuresShaderDrawParameters),
          entry(featuresMultiview),
          entry(featuresIndexTypeUint8),
          entry(featuresSynchronization2),
          entry(featuresTimelineSemaphore),
          entry(featuresVulkanMemoryModel),
          entry(featuresShaderFloat16Int8),
          entry(features16BitStorage),
          entry(featuresBufferDeviceAddress),
          entry(featuresDescriptorIndexing),
          entry(featuresMultiviewPerViewViewports),
          entry(featuresFragmentDensityMap),
          entry(features8BitStorage),
          entry(featuresUniformBufferStandardLayout),
          entry(featuresScalarBlockLayout),
          entry(featuresMeshShader),
          entry(featuresFragmentShadingRate),
          entry(featuresDescriptorBuffer),
          entry(featuresTextureCompressionAstcHdr),
          entry(featuresExtendedDynamicState),
          entry(featuresExtendedDynamicState2),
          entry(featuresShaderIntegerDotProduct)};
}

VkBaseOutStructure* VulkanFeatures::requestedFeatureStructure(FeatureRequest& request) noexcept {
  auto* selected = reinterpret_cast<VkBaseOutStructure*>(request.selected.data());
  for (const auto& feature : featureStructures()) {
    if (feature.structure->sType == selected->sType) {
      return feature.structure;
    }
  }
  return selected;
}

void VulkanFeatures::enableRequestedFeatures(const VulkanContext& context,
                                             VkPhysicalDevice physicalDevice,
                                             uint32_t apiVersion) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const auto availableCoreFeatures = vkPhysicalDeviceFeatures2.features;
  vkPhysicalDeviceFeatures2.features = {
      .independentBlend = availableCoreFeatures.independentBlend,
      .dualSrcBlend = availableCoreFeatures.dualSrcBlend,
      .multiDrawIndirect = availableCoreFeatures.multiDrawIndirect,
      .drawIndirectFirstInstance = availableCoreFeatures.drawIndirectFirstInstance,
      .depthBiasClamp = availableCoreFeatures.depthBiasClamp,
      .fillModeNonSolid = availableCoreFeatures.fillModeNonSolid,
      .samplerAnisotropy = availableCoreFeatures.samplerAnisotropy,
      .shaderInt16 = availableCoreFeatures.shaderInt16,
  };
  for (auto& request : featureRequests_) {
    auto* selected = reinterpret_cast<VkBaseOutStructure*>(request.selected.data());
    const VkStructureType type = selected->sType;
    std::fill(request.selected.begin(), request.selected.end(), FeatureWord{});
    selected->sType = type;
    request.chained = enabled(request.extensionName.c_str(), ExtensionType::Device) ||
                      (request.coreVersion != 0 && apiVersion >= request.coreVersion);
    if (!request.chained) {
      continue;
    }
    VkPhysicalDeviceFeatures2 available{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                        .pNext = selected};
    context.vf_.vkGetPhysicalDeviceFeatures2(
        physicalDevice,
        type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
            ? reinterpret_cast<VkPhysicalDeviceFeatures2*>(selected)
            : &available);
    auto* target = static_cast<VkBaseOutStructure*>(const_cast<void*>(feature(type)));
    const bool alreadyChained = target != nullptr;
    if (!target) {
      target = requestedFeatureStructure(request);
    }
    for (size_t offset = sizeof(VkBaseOutStructure); offset < request.size;
         offset += sizeof(VkBool32)) {
      VkBool32 supported = VK_FALSE;
      std::memcpy(
          &supported, reinterpret_cast<const uint8_t*>(selected) + offset, sizeof(VkBool32));
      if (supported == VK_FALSE) {
        if (!alreadyChained) {
          std::memcpy(reinterpret_cast<uint8_t*>(target) + offset, &supported, sizeof(VkBool32));
        }
        continue;
      }
      VkBool32 requested = VK_FALSE, existing = VK_FALSE;
      std::memcpy(&requested,
                  reinterpret_cast<const uint8_t*>(request.requested.data()) + offset,
                  sizeof(VkBool32));
      if (alreadyChained) {
        std::memcpy(&existing, reinterpret_cast<const uint8_t*>(target) + offset, sizeof(VkBool32));
      }
      const VkBool32 value = existing || requested ? VK_TRUE : VK_FALSE;
      std::memcpy(reinterpret_cast<uint8_t*>(target) + offset, &value, sizeof(VkBool32));
    }
    if (!alreadyChained) {
      target->pNext = nullptr;
      ivkAddNext(&vkPhysicalDeviceFeatures2, target);
    }
    request.addedToChain = !alreadyChained;
  }
}

void VulkanFeatures::resetRequestedFeatures() noexcept {
  for (auto& request : featureRequests_) {
    if (request.addedToChain) {
      auto* structure = requestedFeatureStructure(request);
      clearFeaturePayload(structure, request.size);
    }
    const VkStructureType type =
        reinterpret_cast<const VkBaseOutStructure*>(request.requested.data())->sType;
    std::fill(request.selected.begin(), request.selected.end(), FeatureWord{});
    reinterpret_cast<VkBaseOutStructure*>(request.selected.data())->sType = type;
    request.chained = false;
    request.addedToChain = false;
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void VulkanFeatures::populateWithAvailablePhysicalDeviceFeatures(
    const VulkanContext& context,
    VkPhysicalDevice physicalDevice) noexcept {
  if (!context.vf_.vkGetPhysicalDeviceFeatures2) {
    IGL_LOG_ERROR("Pointer to function vkGetPhysicalDeviceFeatures2() is nullptr\n");
    resetRequestedFeatures();
    for (const auto& feature : featureStructures()) {
      clearFeaturePayload(feature.structure, feature.size);
    }
    assembleFeatureChain(context.config_);
    return;
  }
  uint32_t numExtensions = 0;
  context.vf_.vkEnumerateDeviceExtensionProperties(
      physicalDevice, nullptr, &numExtensions, nullptr);
  extensionProps_.resize(numExtensions);
  context.vf_.vkEnumerateDeviceExtensionProperties(
      physicalDevice, nullptr, &numExtensions, extensionProps_.data());
  assembleFeatureChain(context.config_);
  context.vf_.vkGetPhysicalDeviceFeatures2(physicalDevice, &vkPhysicalDeviceFeatures2);
}

void VulkanFeatures::resolveMutuallyExclusiveFeatures() noexcept {
  if (featuresFragmentDensityMap.fragmentDensityMap == VK_TRUE) {
    featuresFragmentShadingRate.pipelineFragmentShadingRate = VK_FALSE;
    featuresFragmentShadingRate.primitiveFragmentShadingRate = VK_FALSE;
    featuresFragmentShadingRate.attachmentFragmentShadingRate = VK_FALSE;
    // primitiveFragmentShadingRateMeshShader requires primitiveFragmentShadingRate, so it
    // cannot stay enabled once the latter is dropped above.
    featuresMeshShader.primitiveFragmentShadingRateMeshShader = VK_FALSE;
  }
}

bool VulkanFeatures::hasExtension(const char* ext) const {
  for (const VkExtensionProperties& props : extensionProps_) {
    if (std::strcmp(ext, props.extensionName) == 0) {
      return true;
    }
  }
  return false;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
Result VulkanFeatures::checkSelectedFeatures(
    const VulkanFeatures& availableFeatures) const noexcept {
  // Stores missing features
  std::string missingFeatures;

  // Macros for checking whether a requested feature is present. The macro logs an error if the
  // feature is requested and not available Based on
  // https://github.com/corporateshark/lightweightvk/blob/6b5ba5512f0e1ba7b20f4b37d7ec100eb25287c1/lvk/vulkan/VulkanClasses.cpp#L4702
#define ENABLE_VULKAN_FEATURE(requestedFeatureStruct, availableFeatureStruct, feature, version) \
  if ((requestedFeatureStruct.feature == VK_TRUE) &&                                            \
      (availableFeatureStruct.feature == VK_FALSE)) {                                           \
    missingFeatures.append("\n   " version " " #requestedFeatureStruct "." #feature);           \
  }

#define ENABLE_FEATURE_1_1(availableFeatureStruct, feature)                        \
  ENABLE_VULKAN_FEATURE(vkPhysicalDeviceFeatures2.features,                        \
                        availableFeatureStruct.vkPhysicalDeviceFeatures2.features, \
                        feature,                                                   \
                        "1.1")
  ENABLE_FEATURE_1_1(availableFeatures, dualSrcBlend)
  ENABLE_FEATURE_1_1(availableFeatures, shaderInt16)
  ENABLE_FEATURE_1_1(availableFeatures, multiDrawIndirect)
  ENABLE_FEATURE_1_1(availableFeatures, drawIndirectFirstInstance)
  ENABLE_FEATURE_1_1(availableFeatures, depthBiasClamp)
  ENABLE_FEATURE_1_1(availableFeatures, fillModeNonSolid)
#undef ENABLE_FEATURE_1_1

#define ENABLE_FEATURE_1_1_EXT(requestedFeatureStruct, availableFeatureStruct, feature) \
  ENABLE_VULKAN_FEATURE(requestedFeatureStruct, availableFeatureStruct, feature, "1.1 EXT")
  if (config.enableDescriptorIndexing) {
    ENABLE_FEATURE_1_1_EXT(featuresDescriptorIndexing,
                           availableFeatures.featuresDescriptorIndexing,
                           shaderSampledImageArrayNonUniformIndexing)
    ENABLE_FEATURE_1_1_EXT(featuresDescriptorIndexing,
                           availableFeatures.featuresDescriptorIndexing,
                           descriptorBindingUniformBufferUpdateAfterBind)
    ENABLE_FEATURE_1_1_EXT(featuresDescriptorIndexing,
                           availableFeatures.featuresDescriptorIndexing,
                           descriptorBindingSampledImageUpdateAfterBind)
    ENABLE_FEATURE_1_1_EXT(featuresDescriptorIndexing,
                           availableFeatures.featuresDescriptorIndexing,
                           descriptorBindingStorageImageUpdateAfterBind)
    ENABLE_FEATURE_1_1_EXT(featuresDescriptorIndexing,
                           availableFeatures.featuresDescriptorIndexing,
                           descriptorBindingStorageBufferUpdateAfterBind)
    ENABLE_FEATURE_1_1_EXT(featuresDescriptorIndexing,
                           availableFeatures.featuresDescriptorIndexing,
                           descriptorBindingUpdateUnusedWhilePending)
    ENABLE_FEATURE_1_1_EXT(featuresDescriptorIndexing,
                           availableFeatures.featuresDescriptorIndexing,
                           descriptorBindingPartiallyBound)
    ENABLE_FEATURE_1_1_EXT(featuresDescriptorIndexing,
                           availableFeatures.featuresDescriptorIndexing,
                           runtimeDescriptorArray)
  }
  ENABLE_FEATURE_1_1_EXT(
      features16BitStorage, availableFeatures.features16BitStorage, storageBuffer16BitAccess)
  if (has_VK_KHR_buffer_device_address) {
    ENABLE_FEATURE_1_1_EXT(featuresBufferDeviceAddress,
                           availableFeatures.featuresBufferDeviceAddress,
                           bufferDeviceAddress)
  }
  ENABLE_FEATURE_1_1_EXT(featuresMultiview, availableFeatures.featuresMultiview, multiview)
  ENABLE_FEATURE_1_1_EXT(featuresSamplerYcbcrConversion,
                         availableFeatures.featuresSamplerYcbcrConversion,
                         samplerYcbcrConversion)
  ENABLE_FEATURE_1_1_EXT(featuresShaderDrawParameters,
                         availableFeatures.featuresShaderDrawParameters,
                         shaderDrawParameters)
  ENABLE_FEATURE_1_1_EXT(
      featuresSynchronization2, availableFeatures.featuresSynchronization2, synchronization2)
#undef ENABLE_FEATURE_1_1_EXT

#define ENABLE_FEATURE_1_2_EXT(requestedFeatureStruct, availableFeatureStruct, feature) \
  ENABLE_VULKAN_FEATURE(requestedFeatureStruct, availableFeatureStruct, feature, "1.2")
  ENABLE_FEATURE_1_2_EXT(
      featuresShaderFloat16Int8, availableFeatures.featuresShaderFloat16Int8, shaderFloat16)
#undef ENABLE_FEATURE_1_2_EXT

#undef ENABLE_VULKAN_FEATURE

  if (!missingFeatures.empty()) {
#if !IGL_PLATFORM_APPLE
    IGL_DEBUG_ABORT("Missing Vulkan features: %s\n", missingFeatures.c_str());
    return Result(Result::Code::RuntimeError);
#else
    IGL_LOG_INFO("Missing Vulkan features: %s\n", missingFeatures.c_str());
    // For Vulkan 1.3 and MoltenVK, don't return an error as some 1.3 features are available
    // via extensions
#endif
  }

  // Return the value 'Ok'
  return Result{};
}

void VulkanFeatures::assembleFeatureChain(const VulkanContextConfig& contextConfig) noexcept {
  // Versions 1.0 and 1.1 are always present

  // Reset all pNext pointers. We might be copying the chain from another VulkanFeatures object,
  // so we need to reset the pNext pointers to avoid dangling pointers. Some of the extensions'
  // pointers are guarded by #ifdefs below
  for (const auto& feature : featureStructures()) {
    feature.structure->pNext = nullptr;
  }

  // Add the required and optional features to the VkPhysicalDeviceFetaures2_
  ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresSamplerYcbcrConversion);
  ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresShaderDrawParameters);
  ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresMultiview);
  if (hasExtension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresShaderFloat16Int8);
  }
  if (hasExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresBufferDeviceAddress);
  }
  if (hasExtension(VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresVulkanMemoryModel);
  }
  if (hasExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresDescriptorIndexing);
  }
  ivkAddNext(&vkPhysicalDeviceFeatures2, &features16BitStorage);

  if (hasExtension(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresIndexTypeUint8);
  }
  if (hasExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresSynchronization2);
  }
  if (hasExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresTimelineSemaphore);
  }
  if (hasExtension(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresFragmentDensityMap);
  }
  if (hasExtension(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &features8BitStorage);
  }
  if (hasExtension(VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresUniformBufferStandardLayout);
  }
  if (hasExtension(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresScalarBlockLayout);
  }
  if (contextConfig.enableMultiviewPerViewViewports) {
    if (hasExtension(VK_QCOM_MULTIVIEW_PER_VIEW_VIEWPORTS_EXTENSION_NAME)) {
      ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresMultiviewPerViewViewports);
    } else {
      IGL_LOG_ERROR("VK_QCOM_multiview_per_view_viewports extension not supported\n");
    }
  }
  if (enabled(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresDescriptorBuffer);
  }
  // Fragment shading rate must be added before mesh shader to ensure the dependency is properly
  // handled when primitiveFragmentShadingRateMeshShader requires primitiveFragmentShadingRate
  if (hasExtension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresFragmentShadingRate);
  }
  if (hasExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresMeshShader);
  }
  if (hasExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresExtendedDynamicState);
  }
  if (hasExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME)) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresExtendedDynamicState2);
  }
  if (enabled(VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME) &&
      featuresTextureCompressionAstcHdr.textureCompressionASTC_HDR == VK_TRUE) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresTextureCompressionAstcHdr);
  }
  // Gated on the caller's request rather than on the extension being present, unlike the
  // structs above. The feature is core from Vulkan 1.3, where a device may support it without
  // still advertising VK_KHR_shader_integer_dot_product, so an extension gate would silently
  // drop the request on exactly the newest devices. Defaulting to VK_FALSE is what keeps that
  // safe: the struct stays out of the chain for every caller that does not opt in, so this
  // adds nothing to the default device.
  if (featuresShaderIntegerDotProduct.shaderIntegerDotProduct == VK_TRUE) {
    ivkAddNext(&vkPhysicalDeviceFeatures2, &featuresShaderIntegerDotProduct);
  }
  for (auto& request : featureRequests_) {
    reinterpret_cast<VkBaseOutStructure*>(request.selected.data())->pNext = nullptr;
    auto* structure = requestedFeatureStructure(request);
    if (request.chained && !feature(structure->sType)) {
      ivkAddNext(&vkPhysicalDeviceFeatures2, structure);
    }
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
VulkanFeatures& VulkanFeatures::operator=(const VulkanFeatures& other) noexcept {
  if (this == &other) {
    return *this;
  }

  const bool sameConfiguration =
      config.enableDescriptorIndexing == other.config.enableDescriptorIndexing;
  if (!sameConfiguration) {
    return *this;
  }

  vkPhysicalDeviceFeatures2 = other.vkPhysicalDeviceFeatures2;

  featuresSamplerYcbcrConversion = other.featuresSamplerYcbcrConversion;
  featuresShaderDrawParameters = other.featuresShaderDrawParameters;
  featuresMultiview = other.featuresMultiview;
  featuresBufferDeviceAddress = other.featuresBufferDeviceAddress;
  featuresDescriptorIndexing = other.featuresDescriptorIndexing;
  features16BitStorage = other.features16BitStorage;

  // Vulkan 1.2
  featuresVulkanMemoryModel = other.featuresVulkanMemoryModel;
  featuresShaderFloat16Int8 = other.featuresShaderFloat16Int8;
  featuresIndexTypeUint8 = other.featuresIndexTypeUint8;
  featuresSynchronization2 = other.featuresSynchronization2;
  featuresTimelineSemaphore = other.featuresTimelineSemaphore;
  featuresFragmentDensityMap = other.featuresFragmentDensityMap;
  features8BitStorage = other.features8BitStorage;
  featuresUniformBufferStandardLayout = other.featuresUniformBufferStandardLayout;
  featuresScalarBlockLayout = other.featuresScalarBlockLayout;
  featuresMultiviewPerViewViewports = other.featuresMultiviewPerViewViewports;
  featuresMeshShader = other.featuresMeshShader;
  featuresFragmentShadingRate = other.featuresFragmentShadingRate;
  featuresDescriptorBuffer = other.featuresDescriptorBuffer;
  featuresTextureCompressionAstcHdr = other.featuresTextureCompressionAstcHdr;
  featuresExtendedDynamicState = other.featuresExtendedDynamicState;
  featuresExtendedDynamicState2 = other.featuresExtendedDynamicState2;
  featuresShaderIntegerDotProduct = other.featuresShaderIntegerDotProduct;

  extensions_ = other.extensions_;
  enabledExtensions_ = other.enabledExtensions_;
  extensionProps_ = other.extensionProps_;
  featureRequests_ = other.featureRequests_;

  assembleFeatureChain(config);

  return *this;
}

void VulkanFeatures::enumerate(const VulkanFunctionTable& vf) {
  uint32_t count = 0;
  VK_ASSERT(vf.vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));

  std::vector<VkExtensionProperties> allExtensions(count);

  VK_ASSERT(vf.vkEnumerateInstanceExtensionProperties(nullptr, &count, allExtensions.data()));

  constexpr size_t vectorIndex = static_cast<size_t>(ExtensionType::Instance);
  std::transform(allExtensions.cbegin(),
                 allExtensions.cend(),
                 std::back_inserter(extensions_[vectorIndex]),
                 [](const VkExtensionProperties& extensionProperties) {
                   return extensionProperties.extensionName;
                 });
}

void VulkanFeatures::enumerate(const VulkanFunctionTable& vf, VkPhysicalDevice device) {
  uint32_t count = 0;
  VK_ASSERT(vf.vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr));

  std::vector<VkExtensionProperties> allExtensions(count);

  VK_ASSERT(vf.vkEnumerateDeviceExtensionProperties(device, nullptr, &count, allExtensions.data()));

  constexpr size_t vectorIndex = static_cast<size_t>(ExtensionType::Device);
  std::transform(allExtensions.cbegin(),
                 allExtensions.cend(),
                 std::back_inserter(extensions_[vectorIndex]),
                 [](const VkExtensionProperties& extensionProperties) {
                   return extensionProperties.extensionName;
                 });
}

const std::vector<std::string>& VulkanFeatures::allAvailableExtensions(
    ExtensionType extensionType) const {
  const size_t vectorIndex = static_cast<size_t>(extensionType);
  return extensions_[vectorIndex];
}

bool VulkanFeatures::available(const char* extensionName, ExtensionType extensionType) const {
  const size_t vectorIndex = static_cast<size_t>(extensionType);
  const std::string extensionNameStr(extensionName);
  auto result = std::find_if(
      extensions_[vectorIndex].begin(),
      extensions_[vectorIndex].end(),
      [&extensionNameStr](const std::string& extension) { return extension == extensionNameStr; });

  return result != extensions_[vectorIndex].end();
}

bool VulkanFeatures::enable(const char* extensionName, ExtensionType extensionType) {
  const size_t vectorIndex = static_cast<size_t>(extensionType);
  if (available(extensionName, extensionType)) {
    enabledExtensions_[vectorIndex].insert(extensionName);
    return true;
  }
  return false;
}

void VulkanFeatures::enableCommonInstanceExtensions(const VulkanContextConfig& contextConfig) {
  enable(VK_KHR_SURFACE_EXTENSION_NAME, ExtensionType::Instance);
  enable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, ExtensionType::Instance);
#if IGL_PLATFORM_WINDOWS
  enable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, ExtensionType::Instance);
#elif IGL_PLATFORM_ANDROID
  enable(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME, ExtensionType::Instance);
#elif IGL_PLATFORM_LINUX
  enable("VK_KHR_xlib_surface", ExtensionType::Instance);
#elif IGL_PLATFORM_MACOSX
  enable(VK_EXT_METAL_SURFACE_EXTENSION_NAME, ExtensionType::Instance);
#endif

  // https://vulkan.lunarg.com/doc/sdk/1.3.216.0/mac/getting_started.html
  has_VK_KHR_portability_enumeration =
      enable(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, ExtensionType::Instance);

  if (contextConfig.enableValidation &&
      (!IGL_PLATFORM_ANDROID || contextConfig.enableValidationOnAndroid)) {
    enable(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME, ExtensionType::Instance);
  }

  has_VK_KHR_get_surface_capabilities2 =
      enable(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, ExtensionType::Instance);

  has_VK_EXT_headless_surface =
      enable(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME, ExtensionType::Instance);

  if (contextConfig.headless) {
    if (!has_VK_EXT_headless_surface) {
      IGL_LOG_ERROR("VK_EXT_headless_surface extension not supported\n");
    }
  }
  if (contextConfig.swapChainColorSpace != igl::ColorSpace::SRGBNonlinear) {
    const bool enabledExtension =
        enable(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME, ExtensionType::Instance);
    if (!enabledExtension) {
      IGL_LOG_ERROR("VK_EXT_swapchain_colorspace extension not supported\n");
    }
  }
}

void VulkanFeatures::enableCommonDeviceExtensions(const VulkanContextConfig& contextConfig) {
  enable(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME, ExtensionType::Device);
  enable(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME, ExtensionType::Device);
  enable(VK_KHR_SWAPCHAIN_EXTENSION_NAME, ExtensionType::Device);

#if IGL_PLATFORM_ANDROID
  enable(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, ExtensionType::Device);
  enable(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, ExtensionType::Device);
  enable(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME, ExtensionType::Device);
  enable(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME, ExtensionType::Device);
#endif // IGL_PLATFORM_ANDROID

#if !IGL_DEBUG
  has_VK_KHR_shader_non_semantic_info =
      enable(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, ExtensionType::Device);
#endif // !IGL_DEBUG

  // Enable the Portability Subset extension when available (e.g. MoltenVK).
  // Reset the flag if the extension is not available.
  if (has_VK_KHR_portability_enumeration) {
    has_VK_KHR_portability_enumeration = enable("VK_KHR_portability_subset", ExtensionType::Device);
  }

#if IGL_PLATFORM_WINDOWS
  enable(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, ExtensionType::Device);
#endif // IGL_PLATFORM_WINDOWS

#if IGL_PLATFORM_LINUX
  enable(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, ExtensionType::Device);
  enable(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, ExtensionType::Device);
#endif // IGL_PLATFORM_LINUX

#if defined(IGL_WITH_TRACY_GPU)
  enable(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME, ExtensionType::Device);
#endif // IGL_WITH_TRACY_GPU

  has_VK_EXT_index_type_uint8 =
      enable(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME, ExtensionType::Device);
  has_VK_EXT_queue_family_foreign =
      enable(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME, ExtensionType::Device);

  has_VK_KHR_timeline_semaphore =
      enable(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, ExtensionType::Device);

  has_VK_KHR_uniform_buffer_standard_layout =
      enable(VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME, ExtensionType::Device);

  has_VK_EXT_scalar_block_layout =
      enable(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, ExtensionType::Device);

  has_VK_KHR_synchronization2 =
      enable(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, ExtensionType::Device);

  has_VK_KHR_8bit_storage = enable(VK_KHR_8BIT_STORAGE_EXTENSION_NAME, ExtensionType::Device);

  has_VK_KHR_buffer_device_address =
      enable(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, ExtensionType::Device);

  has_VK_KHR_vulkan_memory_model =
      enable(VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME, ExtensionType::Device);

  // VK_KHR_create_renderpass2 (promoted to core in Vulkan 1.2).
  // Only meaningful when device apiVersion < 1.2; on >= 1.2 the core vkCreateRenderPass2() is used
  // directly. Enabling here as an extension also loads the *KHR-suffixed function pointers, which
  // VulkanFunctionTable aliases onto the core slots when those are null. VulkanRenderPassBuilder
  // needs a non-null vkCreateRenderPass2() on every device it supports.
  has_VK_KHR_create_renderpass2 =
      enable(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, ExtensionType::Device);

  // disabled until full VK_EXT_descriptor_buffer support is implemented
  has_VK_EXT_descriptor_buffer =
      false; // enable(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME, ExtensionType::Device);

  has_VK_EXT_descriptor_indexing =
      enable(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, ExtensionType::Device);

  has_VK_EXT_fragment_density_map =
      enable(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME, ExtensionType::Device);

  if (contextConfig.enableMultiviewPerViewViewports) {
    has_VK_QCOM_multiview_per_view_viewports =
        enable(VK_QCOM_MULTIVIEW_PER_VIEW_VIEWPORTS_EXTENSION_NAME, ExtensionType::Device);
    IGL_SOFT_ASSERT(has_VK_QCOM_multiview_per_view_viewports,
                    "VK_QCOM_multiview_per_view_viewports is not supported");
  }

  has_VK_EXT_mesh_shader = enable(VK_EXT_MESH_SHADER_EXTENSION_NAME, ExtensionType::Device);

  // VK_EXT_extended_dynamic_state / _state2 (promoted to core in Vulkan 1.3).
  // Only meaningful when device apiVersion < 1.3; on >= 1.3 core entry points are used
  // directly. Enabling here as extensions also loads the *EXT-suffixed function pointers.
  has_VK_EXT_extended_dynamic_state =
      enable(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, ExtensionType::Device);
  has_VK_EXT_extended_dynamic_state2 =
      enable(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, ExtensionType::Device);

  // Enable fragment shading rate extension (required when primitiveFragmentShadingRateMeshShader is
  // used)
  enable(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, ExtensionType::Device);
}

bool VulkanFeatures::enabled(const char* extensionName) const {
  return enabled(extensionName, ExtensionType::Instance) ||
         enabled(extensionName, ExtensionType::Device);
}

bool VulkanFeatures::enabled(const char* extensionName, ExtensionType extensionType) const {
  return enabledExtensions_[static_cast<size_t>(extensionType)].count(extensionName) > 0;
}

std::vector<const char*> VulkanFeatures::allEnabled(ExtensionType extensionType) const {
  const size_t vectorIndex = static_cast<size_t>(extensionType);
  std::vector<const char*> returnList;
  for (const auto& extension : enabledExtensions_[vectorIndex]) {
    returnList.emplace_back(extension.c_str());
  }
  return returnList;
}

} // namespace igl::vulkan
